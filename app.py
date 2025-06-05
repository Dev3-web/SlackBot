import os
import json
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging
from dotenv import load_dotenv
import re
import random
from datetime import datetime
import numpy as np
import faiss
import pickle
from typing import List, Dict, Optional
import openai
from transformers import pipeline
import google.generativeai as genai
from pymongo.errors import PyMongoError
import gridfs
from bson import ObjectId
from pymongo import MongoClient
import PyPDF2
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Configure Gemini API Key
# IMPORTANT: It's best practice to load this from .env directly, not hardcode a placeholder.
# If you hardcode it here (even a placeholder), it will override the .env value.
# I'm reverting to load from .env first, with a placeholder fallback if missing.
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    GEMINI_API_KEY = "YOUR_GOOGLE_API_KEY_HERE" # Placeholder for missing key
    app.logger.error(
        "CRITICAL: GOOGLE_API_KEY is not set in environment variables. Gemini features will not work."
    )
elif GEMINI_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
    app.logger.warning(
        "WARNING: Gemini API Key is a placeholder. Please set GOOGLE_API_KEY in your .env file."
    )

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GOOGLE_API_KEY_HERE":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        app.logger.info("Gemini API Key configured.")
    except Exception as e:
        app.logger.critical(f"Failed to configure Gemini API: {e}. Please check your key.")
else:
    app.logger.error(
        "CRITICAL: Gemini API Key is not configured. The application will likely fail to get embeddings."
    )

# Initialize Flask app
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "rag_knowledge_base")

# Initialize Slack client
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
if not SLACK_BOT_TOKEN:
    app.logger.error(
        "CRITICAL: SLACK_BOT_TOKEN not found in environment variables. Slack integration will fail."
    )
slack_client = WebClient(token=SLACK_BOT_TOKEN)


# Initialize OpenAI (optional - for better response generation)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    app.logger.info("OpenAI API Key configured (optional feature).")
else:
    app.logger.info(
        "OpenAI API Key not found. OpenAI dependent features will be unavailable."
    )

class EnhancedRAGKnowledgeBase:
    def __init__(self):
        # MongoDB setup
        try:
            self.mongo_client = MongoClient(MONGODB_URI)
            self.db = self.mongo_client[MONGODB_DB_NAME]
            self.documents_collection = self.db.documents
            self.chunks_collection = self.db.document_chunks
            self.gridfs = gridfs.GridFS(self.db)
            
            # Create indexes for better performance
            self.documents_collection.create_index("doc_id", unique=True)
            self.chunks_collection.create_index("doc_id")
            self.chunks_collection.create_index("chunk_id", unique=True)
            self.chunks_collection.create_index([("doc_id", 1), ("chunk_index", 1)])
            
            # Test connection
            self.mongo_client.admin.command('ping') 
            app.logger.info(f"✅ Connected to MongoDB: {MONGODB_URI}")
            self.mongodb_connected = True
        except Exception as e:
            app.logger.error(f"❌ Failed to connect to MongoDB at {MONGODB_URI}: {e}")
            self.mongodb_connected = False
            # We don't raise here, but mark connection as failed
            # This allows the app to start even without DB, but functionalities will be limited.

        # Gemini configuration
        self.gemini_embedding_model_id = "models/text-embedding-004"
        self.model_name = self.gemini_embedding_model_id
        self.embedding_dim = 768 # Standard for text-embedding-004

        app.logger.info(
            f"Initializing RAG with Gemini model: {self.gemini_embedding_model_id} (Dimension: {self.embedding_dim})"
        )

        # FAISS setup
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.doc_ids = [] # Stores chunk_ids
        self.doc_texts = [] # Stores chunk_texts
        self.doc_metadatas = [] # Stores parent document metadata for each chunk

        # QA Pipeline setup
        try:
            # Check if a model is specified in env or use default
            qa_model_name = os.environ.get("QA_MODEL", "distilbert-base-cased-distilled-squad")
            self.qa_pipeline = pipeline(
                "question-answering", model=qa_model_name
            )
            app.logger.info(
                f"QA pipeline '{qa_model_name}' loaded successfully."
            )
        except Exception as e:
            self.qa_pipeline = None
            app.logger.warning(
                f"QA pipeline not loaded. Error: {e}. QA features will be limited."
            )

        # Only attempt to load embeddings if MongoDB is connected and Gemini API key is set
        if self.mongodb_connected and GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GOOGLE_API_KEY_HERE":
            self.load_existing_embeddings()
        elif not self.mongodb_connected:
            app.logger.warning("Skipping load_existing_embeddings: MongoDB connection failed.")
        else:
            app.logger.warning("Skipping load_existing_embeddings: Gemini API key not configured.")


    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        if not text or len(text.strip()) == 0:
            return []
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            current_chunk_text = text[start:end]
            if end >= len(text):
                chunks.append(text[start:])
                break
            # Find a natural breakpoint within the chunk, but not too close to the start
            last_period = current_chunk_text.rfind(".")
            last_newline = current_chunk_text.rfind("\n")
            
            # Prioritize a breakpoint if it's within the last half of the chunk
            # or if it's the only option near the end.
            break_point = -1
            if last_period > chunk_size * 0.7: # If a period is in the last 30% of chunk
                break_point = last_period
            elif last_newline > chunk_size * 0.7: # If a newline is in the last 30%
                break_point = last_newline
            elif last_period != -1: # Otherwise, take the last period
                break_point = last_period
            elif last_newline != -1: # Or last newline
                break_point = last_newline

            if break_point != -1 and break_point > chunk_size // 2: # Ensure it's past mid-point for good breaks
                end = start + break_point + 1 # +1 to include the delimiter
            
            chunks.append(text[start:end].strip()) # Add stripped chunk
            
            next_start = end - overlap
            if next_start <= start: # Prevent infinite loop if overlap is too large or chunk is small
                start = end
            else:
                start = next_start
        return [c for c in chunks if c.strip()] # Filter out empty chunks

    def get_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """Generate embedding for text using Gemini."""
        if (
            not GEMINI_API_KEY
            or GEMINI_API_KEY == "YOUR_GOOGLE_API_KEY_HERE"
        ):
            app.logger.error("Gemini API key not configured. Cannot generate embeddings.")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if not text or not text.strip():
            app.logger.warning("Attempted to get embedding for empty text. Returning zero vector.")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            # Gemini has a 3072 token limit for embed_content.
            # Truncate if necessary, though chunking should handle most cases.
            if len(text) > 2048: # Rough estimate, actual token count can vary
                text = text[:2048] 
                app.logger.warning("Truncated text for embedding due to length.")

            result = genai.embed_content(
                model=self.gemini_embedding_model_id,
                content=text,
                task_type=task_type,
            )
            embedding = np.array(result["embedding"], dtype=np.float32)
            return embedding
        except Exception as e:
            app.logger.error(f"Error generating Gemini embedding for text (first 50 chars): '{text[:50]}...': {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        if not self.mongodb_connected:
            app.logger.error(f"Cannot add document {doc_id}: MongoDB is not connected.")
            return

        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
            app.logger.error(f"Cannot add document {doc_id}: Gemini API key not configured.")
            return

        try:
            # Store document in MongoDB
            doc_data = {
                "doc_id": doc_id,
                "text": text, # Store full text for retrieval if needed
                "metadata": metadata or {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Use upsert to update if exists. This will replace the entire document and its chunks.
            # Store full text in documents collection, not embedding here, as chunks have embeddings
            self.documents_collection.replace_one(
                {"doc_id": doc_id}, 
                doc_data, 
                upsert=True
            )

            # Process chunks
            chunks = self.chunk_text(text)
            if not chunks:
                app.logger.warning(f"No chunks generated for document {doc_id}. Document added but no chunks for search.")
                return

            # Remove existing chunks for this document
            self.chunks_collection.delete_many({"doc_id": doc_id})

            # Add new chunks
            chunk_documents = []
            for i, chunk_text_content in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_embedding = self.get_embedding(
                    chunk_text_content, task_type="RETRIEVAL_DOCUMENT"
                )
                
                chunk_doc = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_text": chunk_text_content,
                    "chunk_embedding": chunk_embedding.tolist(), # Store as list for MongoDB
                    "chunk_index": i,
                    "created_at": datetime.utcnow()
                }
                chunk_documents.append(chunk_doc)

            if chunk_documents:
                self.chunks_collection.insert_many(chunk_documents)

            app.logger.info(
                f"Added/Updated document: {doc_id} with {len(chunks)} chunks in MongoDB."
            )
            
            # Reload FAISS index after adding/updating documents
            self.load_existing_embeddings()

        except PyMongoError as e:
            app.logger.error(f"MongoDB error adding document {doc_id}: {e}")
            raise # Re-raise to indicate failure to the caller
        except Exception as e:
            app.logger.error(f"Error adding document {doc_id}: {e}")
            raise # Re-raise

    def load_existing_embeddings(self):
        """Load embeddings from MongoDB into FAISS index."""
        if not self.mongodb_connected:
            app.logger.warning("Cannot load embeddings: MongoDB is not connected.")
            return
        
        app.logger.info("Loading existing embeddings from MongoDB into FAISS index...")
        
        try:
            # Reset FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.doc_ids, self.doc_texts, self.doc_metadatas = [], [], []
            
            # Get all chunks with their parent document metadata
            pipeline = [
                {
                    "$lookup": {
                        "from": "documents", # The collection to join with
                        "localField": "doc_id", # Field from the input documents (chunks)
                        "foreignField": "doc_id", # Field from the "documents" collection
                        "as": "document_details" # Output array field
                    }
                },
                {
                    "$unwind": "$document_details" # Deconstructs the document_details array field from the input documents to output a document for each element.
                },
                {
                    "$project": { # Selects which fields to return
                        "chunk_id": 1,
                        "chunk_text": 1,
                        "chunk_embedding": 1,
                        "document_metadata": "$document_details.metadata" # Get metadata from the joined document
                    }
                },
                {
                    "$sort": {"chunk_id": 1} # Consistent order
                }
            ]
            
            all_chunks_data = list(self.chunks_collection.aggregate(pipeline))
            
            if not all_chunks_data:
                app.logger.info("No chunks found in MongoDB to load into FAISS index.")
                return

            embeddings_to_add = []
            
            for chunk_data in all_chunks_data:
                try:
                    chunk_embedding = chunk_data.get("chunk_embedding")
                    if not chunk_embedding:
                        app.logger.warning(f"Skipping chunk {chunk_data['chunk_id']}: no embedding found.")
                        continue
                    
                    embedding = np.array(chunk_embedding, dtype=np.float32)
                    if embedding.shape[0] != self.embedding_dim:
                        app.logger.warning(
                            f"Skipping chunk {chunk_data['chunk_id']}: embedding dimension mismatch (expected {self.embedding_dim}, got {embedding.shape[0]})."
                        )
                        continue
                    
                    embeddings_to_add.append(embedding)
                    self.doc_ids.append(chunk_data["chunk_id"])
                    self.doc_texts.append(chunk_data["chunk_text"])
                    self.doc_metadatas.append(chunk_data.get("document_metadata", {})) # Use .get with default for safety
                    
                except Exception as e:
                    app.logger.warning(f"Error processing chunk {chunk_data.get('chunk_id', 'unknown')}: {e}")

            if embeddings_to_add:
                embeddings_array = np.array(embeddings_to_add, dtype=np.float32)
                self.index.add(embeddings_array)
                app.logger.info(f"Successfully loaded {self.index.ntotal} embeddings into FAISS from MongoDB.")
            else:
                app.logger.info("No compatible embeddings found to load into FAISS.")
                
        except PyMongoError as e:
            app.logger.error(f"MongoDB error loading embeddings: {e}")
        except Exception as e:
            app.logger.error(f"Error loading embeddings into FAISS: {e}")

    def semantic_search(self, query: str, n_results: int = 5, score_threshold: float = 0.5) -> Dict:
        try:
            if self.index.ntotal == 0:
                app.logger.warning("FAISS index empty. Cannot search.")
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}
            
            query_embedding = self.get_embedding(query, task_type="RETRIEVAL_QUERY")
            if np.all(query_embedding == 0): # Check if embedding failed (all zeros)
                app.logger.warning("Query embedding failed (returned zero vector). Cannot search.")
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}
            
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            scores, indices = self.index.search(
                query_embedding, k=min(n_results, self.index.ntotal)
            )

            results_data = {"documents": [], "scores": [], "metadatas": [], "ids": []}
            for score, idx in zip(scores[0], indices[0]):
                # Ensure index is valid and score is above threshold
                if (
                    score >= score_threshold
                    and idx != -1
                    and 0 <= idx < len(self.doc_texts) # Check bounds of our lists
                ):
                    results_data["documents"].append(self.doc_texts[idx])
                    results_data["scores"].append(float(score))
                    results_data["metadatas"].append(self.doc_metadatas[idx])
                    results_data["ids"].append(self.doc_ids[idx])
            
            return results_data
        except Exception as e:
            app.logger.error(f"Error in semantic search for query '{query[:50]}...': {e}")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    def hybrid_search(self, query: str, n_results: int = 5) -> Dict:
        # Get more semantic results initially to combine with keyword
        semantic_results = self.semantic_search(query, n_results * 2, score_threshold=0.3) # Lower threshold for more candidates
        combined_results = {}
        query_words_kw = set(re.findall(r"\w+", query.lower()))

        for doc_chunk, score, metadata, chunk_id in zip(
            semantic_results["documents"],
            semantic_results["scores"],
            semantic_results["metadatas"],
            semantic_results["ids"],
        ):
            # Semantic score is directly from FAISS (cosine similarity)
            semantic_component = score 
            
            keyword_score_for_chunk = 0
            if query_words_kw:
                chunk_text_words = set(re.findall(r"\w+", doc_chunk.lower()))
                common_words = query_words_kw.intersection(chunk_text_words)
                if common_words:
                    # Jaccard index for keyword similarity
                    keyword_score_for_chunk = len(common_words) / len(
                        query_words_kw.union(chunk_text_words)
                    )

            # Combine scores (adjust weights as needed)
            # A simple weighted sum: 0.7 for semantic, 0.3 for keyword
            final_score = (semantic_component * 0.7) + (keyword_score_for_chunk * 0.3)

            combined_results[chunk_id] = {
                "document": doc_chunk,
                "semantic_score": score,
                "keyword_score": keyword_score_for_chunk,
                "metadata": metadata,
                "id": chunk_id,
                "final_score": final_score,
            }

        sorted_results_list = sorted(
            combined_results.values(), key=lambda x: x["final_score"], reverse=True
        )[:n_results]
        
        return {
            "documents": [r["document"] for r in sorted_results_list],
            "scores": [r["final_score"] for r in sorted_results_list],
            "metadatas": [r["metadata"] for r in sorted_results_list],
            "ids": [r["id"] for r in sorted_results_list],
            "semantic_scores": [r["semantic_score"] for r in sorted_results_list],
            "keyword_scores": [r["keyword_score"] for r in sorted_results_list],
        }

    def simple_keyword_search(self, query: str, n_results: int = 3) -> Dict:
        try:
            query_words = set(re.findall(r"\w+", query.lower()))
            if not query_words:
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}

            # Build regex for each word to ensure full word matching if possible
            # Or use $text search if full text index is enabled on 'text' field in 'documents'
            # For simplicity, using $or with regex for now, which is less efficient than $text index.
            # To use $text, ensure you have a text index on your 'text' field in 'documents' collection:
            # db.documents.createIndex({"text": "text"})

            # Let's do a simple phrase search if possible, then fallback to word search
            regex_query = "(?i)" + "|".join(re.escape(word) for word in query_words)
            
            # Find documents containing any of the query words.
            # Fetch entire documents and then score them. This is less efficient than if chunks
            # were indexed for keyword search directly, but fits the current model.
            
            # Using $text search requires a text index: db.documents.createIndex({"text": "text"})
            # If you don't have a text index, comment out the $text search part.
            # Example using $text:
            # documents = list(self.documents_collection.find(
            #     {"$text": {"$search": query}}, 
            #     {"score": {"$meta": "textScore"}}
            # ).sort([("score", {"$meta": "textScore"})]).limit(n_results * 5)) # Fetch more for scoring

            # Fallback to regex search on document text if $text not used or desired
            documents_from_db = list(self.documents_collection.find(
                {"text": {"$regex": regex_query, "$options": "i"}}
            ).limit(n_results * 5)) # Fetch more than n_results to allow for internal scoring
            
            if not documents_from_db:
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}

            scored_docs = []
            for doc in documents_from_db:
                text_words = set(re.findall(r"\w+", doc["text"].lower()))
                common_words = query_words.intersection(text_words)
                
                # Simple keyword score based on Jaccard index
                score = 0
                if query_words.union(text_words): # Avoid division by zero
                    score = len(common_words) / len(query_words.union(text_words))
                
                if score > 0: # Only add if there's a match
                    # For keyword search, we return the full document text, not just chunks
                    scored_docs.append((score, doc["text"], doc["metadata"], doc["doc_id"]))

            scored_docs.sort(reverse=True, key=lambda x: x[0])
            top_docs = scored_docs[:n_results]
            
            return {
                "documents": [d[1] for d in top_docs],
                "scores": [d[0] for d in top_docs],
                "metadatas": [d[2] for d in top_docs],
                "ids": [d[3] for d in top_docs],
            }
        except PyMongoError as e:
            app.logger.error(f"MongoDB error in keyword search: {e}")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}
        except Exception as e:
            app.logger.error(f"Error in keyword search: {e}")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    def generate_answer_with_qa_model(self, question: str, context: str) -> str:
        if not self.qa_pipeline:
            return context[:500] + "..." if len(context) > 500 else context
        try:
            # The QA model has a context limit. Truncate context if too long.
            # distilbert-base-cased-distilled-squad has a max sequence length of 384
            # We'll use a larger buffer for safety.
            max_context_len = 500 # Characters, not tokens, as a rough measure
            if len(context) > max_context_len:
                context = context[:max_context_len]

            result = self.qa_pipeline(question=question, context=context)
            answer, confidence = result["answer"], result["score"]
            
            # Confidence threshold and answer length heuristic
            if confidence < 0.2 or len(answer) < 10: # Increased confidence threshold slightly
                app.logger.info(f"QA model confidence ({confidence:.2f}) or answer length low ({len(answer)}). Fallback to context.")
                return context[:500] + "..." if len(context) > 500 else context
            
            return answer
        except Exception as e:
            app.logger.error(f"Error in QA generation: {e}")
            return context[:500] + "..." if len(context) > 500 else context

    def search(self, query: str, n_results: int = 3, use_hybrid: bool = True) -> Dict:
        if use_hybrid:
            return self.hybrid_search(query, n_results)
        results = self.semantic_search(query, n_results)
        results["semantic_scores"] = results.get("scores", [])
        results["keyword_scores"] = [0.0] * len(results.get("documents", []))
        return results

    def get_all_documents(self):
        if not self.mongodb_connected:
            app.logger.error("Cannot get all documents: MongoDB is not connected.")
            return {"ids": [], "documents": [], "metadatas": []}
            
        try:
            # Fetch all documents, but only return necessary fields for summary
            documents = list(self.documents_collection.find({}, {"doc_id": 1, "text": 1, "metadata": 1}))
            return {
                "ids": [doc["doc_id"] for doc in documents],
                "documents": [doc["text"] for doc in documents],
                "metadatas": [doc["metadata"] for doc in documents],
            }
        except PyMongoError as e:
            app.logger.error(f"MongoDB error getting all documents: {e}")
            return {"ids": [], "documents": [], "metadatas": []}
        except Exception as e:
            app.logger.error(f"Error getting all documents: {e}")
            return {"ids": [], "documents": [], "metadatas": []}

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks from MongoDB."""
        if not self.mongodb_connected:
            app.logger.error(f"Cannot delete document {doc_id}: MongoDB is not connected.")
            return False
            
        try:
            # Delete document
            doc_result = self.documents_collection.delete_one({"doc_id": doc_id})
            # Delete chunks
            chunk_result = self.chunks_collection.delete_many({"doc_id": doc_id})
            
            if doc_result.deleted_count > 0:
                app.logger.info(f"Deleted document {doc_id} and {chunk_result.deleted_count} chunks")
                self.load_existing_embeddings()  # Reload FAISS index
                return True
            else:
                app.logger.warning(f"Document {doc_id} not found for deletion.")
                return False
        except PyMongoError as e:
            app.logger.error(f"MongoDB error deleting document {doc_id}: {e}")
            return False
        except Exception as e:
            app.logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def get_database_stats(self):
        """Get statistics about the MongoDB database."""
        if not self.mongodb_connected:
            app.logger.error("Cannot get database stats: MongoDB is not connected.")
            return {"error": "MongoDB not connected", "mongodb_connected": False}
            
        try:
            doc_count = self.documents_collection.count_documents({})
            chunk_count = self.chunks_collection.count_documents({})
            
            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "faiss_index_size": self.index.ntotal if self.index else 0,
                "embedding_dimension": self.embedding_dim,
                "database_name": MONGODB_DB_NAME,
                "mongodb_connected": self.mongodb_connected # Added for health check
            }
        except Exception as e:
            app.logger.error(f"Error getting database stats: {e}")
            return {"error": str(e), "mongodb_connected": self.mongodb_connected}

# --- Initialize KB globally ---
kb = EnhancedRAGKnowledgeBase() 
app.logger.info("✅ RAG Knowledge Base initialization attempted.") 

# --- PDF Processing Utility Function ---
def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from a file-like object assumed to be a PDF."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""
    except Exception as e:
        app.logger.error(f"Error extracting text from PDF: {e}")
        raise
    return text

# --- Greeting and other conversational functions ---
def get_time_based_greeting():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return random.choice(["Good morning! ☀️", "Morning! 🌅"])
    elif 12 <= current_hour < 17:
        return random.choice(["Good afternoon! ☀️", "Afternoon! 👋"])
    elif 17 <= current_hour < 21:
        return random.choice(["Good evening! 🌆", "Evening! 👋"])
    else:
        return random.choice(["Hello! Working late? 🌙", "Hi there! 👋"])


def detect_greeting(text):
    text_lower = text.lower().strip()
    greetings = [
        "hello",
        "hi",
        "hey",
        "hiya",
        "howdy",
        "sup",
        "what's up",
        "good morning",
        "morning",
        "good afternoon",
        "afternoon",
        "good evening",
        "evening",
        "yo",
        "hola",
    ]
    for greeting in greetings:
        if text_lower.startswith(greeting):
            return True
    if any(p in text_lower for p in ["how are you", "how's it going"]):
        return True
    return False


def generate_greeting_response():
    time_greeting = get_time_based_greeting()
    follow_ups = [
        "How can I help you today?",
        "What can I assist you with?",
        "Ready to help! What do you need?",
    ]
    return f"{time_greeting} {random.choice(follow_ups)} 😊"


def detect_thanks(text):
    text_lower = text.lower().strip()
    thanks_phrases = [
        "thank you",
        "thanks",
        "thx",
        "ty",
        "appreciate it",
        "grateful",
        "cheers",
        "awesome",
        "perfect",
        "great",
        "helpful",
    ]
    if (
        any(phrase in text_lower for phrase in thanks_phrases)
        and not "no thanks" in text_lower
    ):
        return True
    return False


def generate_thanks_response():
    responses = [
        "You're very welcome! 😊",
        "Happy to help! 👍",
        "Glad I could assist! 🤖",
        "No problem at all!",
    ]
    return random.choice(responses)


def detect_help_request(text):
    text_lower = text.lower().strip()
    help_phrases = [
        "help",
        "what can you do",
        "how can you help",
        "capabilities",
        "assistance",
        "guide me",
    ]
    if any(phrase in text_lower for phrase in help_phrases):
        return True
    return False


def generate_help_response():
    return (
        f"I'm an AI assistant using RAG with Gemini embeddings ({kb.model_name}) 🤖. "
        "I can answer questions based on my knowledge base about support, billing, passwords, or product features. Ask me anything!"
    )


def enhanced_generate_response(
    question: str,
    context_docs: List[str],
    scores: List[float] = None,
    doc_ids: List[str] = None,
) -> str:
    question_lower = question.lower().strip()

    if detect_greeting(question_lower):
        return generate_greeting_response()
    if detect_thanks(question_lower):
        return generate_thanks_response()
    if detect_help_request(question_lower):
        return generate_help_response()

    farewell_words = [
        "bye",
        "goodbye",
        "see you",
        "farewell",
        "take care",
        "later",
        "cya",
        "good night",
    ]
    if any(word in question_lower for word in farewell_words):
        return random.choice(["Goodbye! 👋", "See you later!", "Take care!"])

    if not context_docs or not any(context_docs):
        return "I couldn't find relevant information for your question. 🤔 Could you rephrase or ask about something else?"

    best_context = context_docs[0]
    confidence_score = scores[0] if scores and scores[0] is not None else 0.0

    # Gemini scores (cosine similarity) are typically 0.5-0.9 for relevant docs
    confidence_emoji = (
        "🎯" if confidence_score > 0.7 else "📈" if confidence_score > 0.55 else "💡"
    )

    if kb.qa_pipeline and best_context and len(best_context.strip()) > 20:
        generated_answer = kb.generate_answer_with_qa_model(question, best_context)
        # Check if QA model returned something different from truncated context and is meaningful
        is_qa_meaningful = (
            generated_answer
            != (best_context[:500] + "..." if len(best_context) > 500 else best_context)
            and len(generated_answer) >= 15
        )
        if is_qa_meaningful:
            return (
                f"{confidence_emoji} Here's what I found:\n\n_{generated_answer}_\n\n"
                f"Is this helpful? (Source relevance: {confidence_score:.2f})"
            )

    response_text = best_context
    if len(response_text) > 700:
        response_text = response_text[:700] + "..."
    return (
        f"{confidence_emoji}here's some information:\n\n{response_text}\n\n"
        f"Is this helpful?"
    )


# --- Slack event handling ---
def handle_message(event):
    try:
        channel = event["channel"]
        text = event.get("text", "")
        if not text:
            return

        bot_user_id = os.environ.get("SLACK_BOT_USER_ID")
        if bot_user_id:
            text = text.replace(f"<@{bot_user_id}>", "").strip()
        if not text:
            return

        app.logger.info(f"Processing question from channel {channel}: '{text}'")

        # Check for conversational cues first
        if (
            detect_greeting(text)
            or detect_thanks(text)
            or detect_help_request(text)
            or any(
                word in text.lower().strip()
                for word in ["bye", "goodbye", "good night"]
            )
        ):
            response = enhanced_generate_response(text, [], None, None)
        else:
            # Ensure KB is ready before performing search
            if not kb.mongodb_connected or not (GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GOOGLE_API_KEY_HERE"):
                response = "Sorry, my knowledge base is not fully operational right now. Please check the server status."
            else:
                search_results = kb.search(text, n_results=3, use_hybrid=True)
                context_docs = search_results.get("documents", [])
                scores = search_results.get("scores")
                doc_ids = search_results.get("ids") # Pass along if needed by response fn
                if context_docs:
                    app.logger.info(
                        f"Found {len(context_docs)} relevant chunks. Top score: {scores[0] if scores else 'N/A'}"
                    )
                else:
                    app.logger.info(f"No relevant documents found for query: '{text}'")
                response = enhanced_generate_response(text, context_docs, scores, doc_ids)

        if SLACK_BOT_TOKEN: # Ensure token is available before trying to post
            slack_client.chat_postMessage(
                channel=channel, text=response, thread_ts=event.get("ts")
            )
            app.logger.info(f"Sent response (first 100 chars): {response[:100]}...")
        else:
            app.logger.error(
                "Cannot send Slack message: SLACK_BOT_TOKEN not configured."
            )

    except SlackApiError as e:
        app.logger.error(f"Slack API Error: {e.response['error']}")
    except Exception as e:
        app.logger.error(f"Error handling message: {e}", exc_info=True)
        if SLACK_BOT_TOKEN:
            try:
                slack_client.chat_postMessage(
                    channel=event.get("channel"),
                    text="Sorry, an error occurred. 😅 Please try again.",
                    thread_ts=event.get("ts"),
                )
            except Exception as slack_err:
                app.logger.error(f"Failed to send error message to Slack: {slack_err}")


# --- Flask Routes ---
# IMPORTANT: Only ONE @app.route("/slack/events") definition allowed!
@app.route("/slack/events", methods=["POST"]) 
def slack_events_route(): # Renamed to avoid confusion with the handler function
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data["challenge"]})
    if data.get("type") == "event_callback":
        event = data.get("event")
        if not event:
            return jsonify({"status": "ok", "message": "No event payload"}), 200
        # Ignore messages from bots to prevent infinite loops
        if event.get("subtype") == "bot_message" or event.get("bot_id"):
            app.logger.info("Ignoring bot message.")
            return jsonify({"status": "ok", "message": "Ignored bot message"})

        # Process app_mention or direct messages
        if event.get("type") == "app_mention" or (
            event.get("type") == "message" and event.get("channel_type") == "im"
        ):
            handle_message(event)
    return jsonify({"status": "ok"})


@app.route("/delete_knowledge/<doc_id>", methods=["DELETE"])
def delete_knowledge(doc_id):
    try:
        success = kb.delete_document(doc_id)
        if success:
            return jsonify({"status": "success", "message": f"Document '{doc_id}' deleted."})
        else:
            return jsonify({"error": "Document not found or deletion failed."}), 404
    except Exception as e:
        app.logger.error(f"Error deleting knowledge: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/db_stats", methods=["GET"])
def database_stats_route(): # Renamed to avoid clash if db_stats used internally
    try:
        stats = kb.get_database_stats()
        if stats.get("error") and stats.get("mongodb_connected") is False: # Check specifically for mongo connection error
             return jsonify({"status": "error", "message": stats["error"]}), 503 # Service Unavailable if MongoDB is down
        elif stats.get("error"): # Other errors
            return jsonify({"status": "error", "message": stats["error"]}), 500
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error getting database stats via route: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/add_knowledge", methods=["POST"])
def add_knowledge():
    try:
        data = request.json
        if not data or not data.get("id") or not data.get("text"):
            return jsonify({"error": "Missing 'id' or 'text' in request body."}), 400
        
        # Ensure MongoDB is connected
        if not kb.mongodb_connected:
            return jsonify({"error": "Knowledge base not ready: MongoDB not connected."}), 503
        
        # Ensure Gemini API is configured
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
            return jsonify({"error": "Knowledge base not ready: Gemini API key not configured."}), 503

        kb.add_document(data["id"], data["text"], data.get("metadata", {}))
        return jsonify(
            {
                "status": "success",
                "message": f"Document '{data['id']}' added/updated with Gemini embeddings.",
            }
        )
    except Exception as e:
        app.logger.error(f"Error adding knowledge: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if not file or not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Invalid file type. Only PDF files are supported."}), 400

        doc_id = request.form.get('doc_id')
        if not doc_id:
            # Generate a unique ID if not provided
            doc_id = f"pdf_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}"
        
        metadata_str = request.form.get('metadata', '{}')
        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            metadata = {} # Default to empty if invalid JSON
            app.logger.warning(f"Invalid metadata JSON provided for PDF upload: {metadata_str}")

        pdf_content = file.read()
        text = extract_text_from_pdf(io.BytesIO(pdf_content))
        
        if not text.strip():
            return jsonify({"error": "No text extracted from PDF. PDF might be image-based or empty."}), 400
        
        # Ensure KB is ready before attempting to add document
        if not kb.mongodb_connected:
            return jsonify({"error": "Knowledge base not ready: MongoDB not connected."}), 503
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
            return jsonify({"error": "Knowledge base not ready: Gemini API key not configured."}), 503

        kb.add_document(doc_id, text, metadata)
        
        return jsonify({
            "status": "success",
            "message": f"PDF '{file.filename}' processed and added as document '{doc_id}'.",
            "doc_id": doc_id,
            "extracted_text_preview": text[:200] + "..." if len(text) > 200 else text
        })

    except PyPDF2.errors.PdfReadError as e:
        app.logger.error(f"PDF Read Error: {e}")
        return jsonify({"error": f"Failed to read PDF: {e}. It might be corrupted or encrypted."}), 400
    except Exception as e:
        app.logger.error(f"Error uploading PDF: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/test_search", methods=["POST"])
def test_search():
    try:
        data = request.json
        if not data or not data.get("query"):
            return jsonify({"error": "Missing query"}), 400
        query, search_type, n_results = (
            data["query"],
            data.get("search_type", "hybrid").lower(),
            data.get("n_results", 3),
        )

        # Ensure KB is ready before attempting search
        if not kb.mongodb_connected or not (GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GOOGLE_API_KEY_HERE"):
            return jsonify({"error": "Knowledge base not ready: MongoDB not connected or Gemini API key not configured."}), 503

        if search_type == "semantic":
            results = kb.semantic_search(query, n_results)
        elif search_type == "keyword":
            results = kb.simple_keyword_search(query, n_results)
        elif search_type == "hybrid":
            results = kb.hybrid_search(query, n_results)
        else:
            return jsonify({"error": f"Invalid search_type: {search_type}"}), 400

        formatted_results = []
        if results and results.get("documents"):
            for i in range(len(results["documents"])):
                formatted_results.append(
                    {
                        "document": results["documents"][i],
                        "score": (
                            results["scores"][i]
                            if "scores" in results and i < len(results["scores"])
                            else None
                        ),
                        "id": (
                            results["ids"][i]
                            if "ids" in results and i < len(results["ids"])
                            else None
                        ),
                        "metadata": (
                            results["metadatas"][i]
                            if "metadatas" in results and i < len(results["metadatas"])
                            else {}
                        ),
                        "semantic_score": (
                            results.get("semantic_scores", [])[i]
                            if "semantic_scores" in results
                            and i < len(results.get("semantic_scores", []))
                            else None
                        ),
                        "keyword_score": (
                            results.get("keyword_scores", [])[i]
                            if "keyword_scores" in results
                            and i < len(results.get("keyword_scores", []))
                            else None
                        ),
                    }
                )
        return jsonify(
            {"query": query, "search_type": search_type, "results": formatted_results}
        )
    except Exception as e:
        app.logger.error(f"Error in test_search: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/list_knowledge", methods=["GET"])
def list_knowledge():
    try:
        if not kb.mongodb_connected:
            return jsonify({"error": "Knowledge base not ready: MongoDB not connected."}), 503
            
        docs_summary = kb.get_all_documents()
        return jsonify(
            {
                "total_documents": len(docs_summary["ids"]),
                "embedding_model": kb.model_name,
                "vector_dimension": kb.embedding_dim,
                "total_chunks_in_faiss": kb.index.ntotal if kb.index else 0,
                "documents": [
                    {
                        "id": docs_summary["ids"][i],
                        "text_preview": docs_summary["documents"][i][:200] + "..." if len(docs_summary["documents"][i]) > 200 else docs_summary["documents"][i],
                        "metadata": docs_summary["metadatas"][i],
                    }
                    for i in range(len(docs_summary["ids"]))
                ],
            }
        )
    except Exception as e:
        app.logger.error(f"Error in list_knowledge: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    # Use the kb.mongodb_connected status directly
    is_mongodb_connected = kb.mongodb_connected 
    
    db_stats = {}
    if is_mongodb_connected:
        try:
            db_stats = kb.get_database_stats()
        except Exception as e:
            app.logger.error(f"Failed to get DB stats during health check: {e}")
            db_stats = {"error": str(e)} # Indicate error in stats if it occurs during fetch
            
    # Check if a Gemini API key is set AND it's not the placeholder
    gemini_configured = bool(
        GEMINI_API_KEY
        and GEMINI_API_KEY not in [
            "YOUR_GOOGLE_API_KEY_HERE"
        ]
    )

    return jsonify({
        "status": "healthy" if is_mongodb_connected and gemini_configured else "unhealthy", # Overall status depends on critical components
        "rag_enabled": is_mongodb_connected and gemini_configured,
        "database_type": "MongoDB",
        "database_name": MONGODB_DB_NAME,
        "embedding_model": kb.model_name,
        "vector_dimension": kb.embedding_dim,
        "total_documents_in_db": db_stats.get("total_documents", 0),
        "total_chunks_in_db": db_stats.get("total_chunks", 0),
        "total_chunks_in_faiss": kb.index.ntotal if kb.index else 0,
        "qa_model_loaded": kb.qa_pipeline is not None,
        "gemini_api_configured": gemini_configured,
        "mongodb_connected": is_mongodb_connected,
        "slack_bot_token_configured": bool(SLACK_BOT_TOKEN) # New check for Slack
    })


@app.route("/")
def home():
    total_docs = kb.get_all_documents()["total_documents"] if kb.mongodb_connected else "N/A"
    total_chunks = kb.index.ntotal if kb.index and kb.mongodb_connected else "N/A"
    qa_status = "✅ Loaded (DistilBERT)" if kb.qa_pipeline else "❌ Not loaded"
    
    gemini_ok_display = bool(
        GEMINI_API_KEY
        and GEMINI_API_KEY
        not in ["YOUR_GOOGLE_API_KEY_HERE"]
    )
    
    return f"""
    <h1>🚀 RAG Slack AI Bot (Gemini Embeddings)</h1>
    <p>Status: {'✅ Running' if kb.mongodb_connected and gemini_ok_display else '⚠️ Critical Configuration Issue!'}</p>
    <h2>🔧 RAG Features:</h2>
    <ul>
        <li><strong>Embedding Service:</strong> Google Gemini ('{kb.model_name}')</li>
        <li><strong>Vector DB:</strong> FAISS (Inner Product for Cosine Similarity)</li>
        <li><strong>QA Model:</strong> {'DistilBERT' if kb.qa_pipeline else 'Not Loaded'}</li>
    </ul>
    <h2>📊 Stats:</h2>
    <ul>
        <li><strong>Embedding Model:</strong> {kb.model_name} (Dim: {kb.embedding_dim})</li>
        <li><strong>Total Docs in DB:</strong> {total_docs}</li>
        <li><strong>Total Chunks in FAISS:</strong> {total_chunks}</li>
        <li><strong>QA Model:</strong> {qa_status}</li>
        <li><strong>Gemini API Key:</strong> {'✅ Configured' if gemini_ok_display else '❌ NOT CONFIGURED OR PLACEHOLDER!'}</li>
        <li><strong>MongoDB Connection:</strong> {'✅ Connected' if kb.mongodb_connected else '❌ NOT CONNECTED!'}</li>
        <li><strong>Slack Token:</strong> {'✅ Configured' if SLACK_BOT_TOKEN else '❌ NOT CONFIGURED!'}</li>
    </ul>
    <p><em>See code/logs for API details. Remember to set GOOGLE_API_KEY, SLACK_BOT_TOKEN, and MONGODB_URI in your .env file.</em></p>
    """


# --- Sample Data Initialization ---
def initialize_sample_data():
    sample_docs = [
        {
            "id": "support_policy",
            "text": "Our support team is available Monday through Friday, from 9 AM to 5 PM EST. For immediate assistance, please use our live chat feature on the website or email support@example.com. We aim to respond to all inquiries within 24 business hours.",
            "metadata": {"category": "support", "source": "Website FAQ"},
        },
        {
            "id": "password_reset_guide",
            "text": "To reset your password, visit the login page and click on 'Forgot Password'. Enter your registered email address, and a password reset link will be sent to you. If you don't receive the email within a few minutes, please check your spam folder.",
            "metadata": {"category": "account", "source": "User Manual"},
        },
        {
            "id": "billing_information",
            "text": "All invoices are generated monthly and sent to your registered billing email address. Payments are due within 30 days of the invoice date. For any billing inquiries or to update your payment method, please contact our billing department at billing@example.com or call us at (123) 456-7890 during business hours.",
            "metadata": {"category": "billing", "source": "Terms and Conditions"},
        },
        {
            "id": "product_features_overview",
            "text": "Our product includes robust data analytics dashboards, real-time reporting, and customizable user roles. Key features also include secure data encryption, cloud storage integration, and API access for developers. New features are rolled out quarterly.",
            "metadata": {"category": "product", "source": "Product Brochure"},
        },
        {
            "id": "slack_integration_details",
            "text": "The Slack integration allows you to receive instant notifications, automate task assignments, and interact with our bot directly within your Slack channels. Set up the integration by visiting your profile settings on our platform and connecting your Slack workspace.",
            "metadata": {"category": "integration", "source": "Help Center"},
        },
    ]

    if not kb.mongodb_connected:
        app.logger.warning("Skipping sample data initialization: MongoDB not connected.")
        return

    # Check if any documents already exist
    existing_doc_count = kb.documents_collection.count_documents({})
    
    gemini_configured_for_embeddings = bool(
        GEMINI_API_KEY
        and GEMINI_API_KEY != "YOUR_GOOGLE_API_KEY_HERE"
    )

    if existing_doc_count == 0 and gemini_configured_for_embeddings:
        app.logger.info(
            "🔄 Initializing RAG with sample data (using Gemini embeddings)..."
        )
        for doc in sample_docs:
            try:
                kb.add_document(
                    doc["id"], doc["text"], doc.get("metadata", {})
                )
                app.logger.info(f"Successfully added sample document: {doc['id']}")
            except Exception as e:
                app.logger.error(f"Failed to add sample document {doc['id']}: {e}")
        app.logger.info(
            f"✅ Attempted to add {len(sample_docs)} sample docs. Current FAISS chunks: {kb.index.ntotal if kb.index else 'N/A'}"
        )
    elif existing_doc_count > 0:
        app.logger.info(
            f"📚 MongoDB already contains {existing_doc_count} documents. Skipping sample data initialization."
        )
        # Ensure FAISS index is loaded even if sample data was skipped but DB has data
        if (kb.index is None or kb.index.ntotal == 0) and gemini_configured_for_embeddings:
            app.logger.info("Attempting to load existing embeddings into FAISS as it seems empty...")
            kb.load_existing_embeddings()
    elif not gemini_configured_for_embeddings:
        app.logger.warning(
            "Skipping sample data initialization because Gemini API key is not properly configured."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    app.logger.info("🚀 Starting RAG-Powered Slack AI Bot with Gemini Embeddings")

    # Re-evaluate critical configs for startup message
    gemini_ok_startup = bool(
        GEMINI_API_KEY
        and GEMINI_API_KEY
        not in ["YOUR_GOOGLE_API_KEY_HERE"]
    )
    slack_ok_startup = bool(SLACK_BOT_TOKEN)
    mongodb_ok_startup = kb.mongodb_connected # Check after KB initialization

    if not gemini_ok_startup:
        app.logger.critical(
            "FATAL: Gemini API Key is not configured correctly. Embedding generation will fail. Please set GOOGLE_API_KEY."
        )
    if not slack_ok_startup:
        app.logger.critical(
            "FATAL: SLACK_BOT_TOKEN is not configured. Bot cannot connect to Slack. Please set SLACK_BOT_TOKEN."
        )
    if not mongodb_ok_startup:
        app.logger.critical(
            "FATAL: MongoDB is not connected. Knowledge base functionality will be severely limited. Please check MONGODB_URI."
        )

    # Initialize sample data only if core components are okay
    if gemini_ok_startup and mongodb_ok_startup:
        initialize_sample_data()
    else:
        app.logger.warning(
            "Skipping sample data initialization due to missing Gemini API key or MongoDB connection issues."
        )

    app.logger.info("\n" + "=" * 80)
    app.logger.info("🎯 Gemini RAG Bot is ready (or attempting to be)!")
    app.logger.info(
        f"   • Embeddings by: Google Gemini ('{kb.model_name}') {'✅' if gemini_ok_startup else '❌ (KEY ISSUE!)'}"
    )
    app.logger.info(
        f"   • Slack Integration: {'✅' if slack_ok_startup else '❌ (TOKEN ISSUE!)'}"
    )
    app.logger.info(
        f"   • MongoDB Connection: {'✅' if mongodb_ok_startup else '❌ (CONNECTION ISSUE!)'}"
    )
    app.logger.info(
        f"   • QA Model: {'DistilBERT (loaded)' if kb.qa_pipeline else 'Not loaded'}"
    )
    app.logger.info(
        f"🌐 Web interface: http://localhost:{os.environ.get('PORT', 3000)}"
    ) # Corrected default port to 3000 for consistency
    app.logger.info("=" * 80 + "\n")

    if not gemini_ok_startup or not slack_ok_startup or not mongodb_ok_startup:
        app.logger.warning(
            "CRITICAL CONFIGURATION ISSUES DETECTED. BOT MAY NOT FUNCTION CORRECTLY."
        )

    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)