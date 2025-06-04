import os
import json
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging
from dotenv import load_dotenv
import sqlite3
import re
from collections import Counter
import math
import random
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Dict, Tuple, Optional
import openai
from transformers import pipeline

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Initialize Slack client
slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

# Initialize OpenAI (optional - for better response generation)
openai.api_key = os.environ.get("OPENAI_API_KEY")

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# ^-- This line was causing the OSError in the traceback.
# It also appears to be an unused global variable, as EnhancedRAGKnowledgeBase initializes its own.
# If the OSError persists when EnhancedRAGKnowledgeBase initializes its model,
# you may need to clear your Hugging Face model cache for "sentence-transformers/all-MiniLM-L6-v2".
# Cache location is typically:
# - Linux/macOS: ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
# - Windows: C:\Users\<YourUsername>\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2
# Delete that specific model folder and re-run.


class EnhancedRAGKnowledgeBase:
    def __init__(
        self,
        db_path="knowledge_base.db",
        model_name="mixedbread-ai/mxbai-embed-large-v1",
    ):
        self.db_path = db_path
        self.model_name = model_name
        # The following line might also raise the OSError if the cache is corrupted.
        # If so, ensure you clear the cache as described in the comment above.
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize FAISS index for vector similarity search
        self.index = faiss.IndexFlatIP(
            self.embedding_dim
        )  # Inner product for cosine similarity
        self.doc_ids = []
        self.doc_texts = []
        self.doc_metadatas = []

        # Initialize question-answering pipeline (optional)
        try:
            self.qa_pipeline = pipeline(
                "question-answering", model="distilbert-base-cased-distilled-squad"
            )
        except Exception as e:  # Catch more general exceptions during pipeline loading
            self.qa_pipeline = None
            print(
                f"Warning: QA pipeline not loaded (model 'distilbert-base-cased-distilled-squad'). Error: {e}"
            )
            print("Using simple context retrieval instead.")

        self.init_database()
        self.load_existing_embeddings()

    def init_database(self):
        """Initialize SQLite database with embeddings support"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create chunks table for better retrieval
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                chunk_text TEXT NOT NULL,
                chunk_embedding BLOB,
                chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents (id)
            )
        """
        )

        conn.commit()
        conn.close()

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks for better retrieval"""
        if not text or len(text) == 0:  # Handle empty string case
            return []
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            current_chunk_text = text[
                start:end
            ]  # Define current_chunk_text for clarity

            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to break at sentence boundary
            # Ensure slice indices are valid
            last_period = current_chunk_text.rfind(".")
            last_newline = current_chunk_text.rfind("\n")

            break_point_in_chunk = max(last_period, last_newline)

            # If we found a good break point within the current chunk (not at the very beginning)
            # and it's past a certain portion of the chunk (e.g., half), use it.
            if break_point_in_chunk > chunk_size // 2:
                end = start + break_point_in_chunk + 1

            chunks.append(text[start:end])

            # Ensure overlap doesn't cause start to go negative or create infinite loop with small chunks
            if end - overlap > start:
                start = end - overlap
            else:  # If overlap is too large or chunk is too small, advance by a minimal amount or break
                start = end  # Or handle differently, e.g., make non-overlapping in this case

        # Post-process to remove very small or empty chunks if any resulted from logic
        chunks = [c for c in chunks if c.strip()]
        return chunks

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.embedding_model.encode(text, normalize_embeddings=True)

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        """Add a document with embeddings and chunking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            metadata_json = json.dumps(metadata or {})

            # Generate embedding for full document
            doc_embedding = self.get_embedding(text)
            doc_embedding_blob = pickle.dumps(doc_embedding)

            # Store main document
            cursor.execute(
                """
                INSERT OR REPLACE INTO documents (id, text, metadata, embedding)
                VALUES (?, ?, ?, ?)
            """,
                (doc_id, text, metadata_json, doc_embedding_blob),
            )

            # Create and store chunks
            chunks = self.chunk_text(text)
            if not chunks:  # If text was empty or too short to chunk meaningfully
                print(
                    f"Warning: No chunks generated for document {doc_id}. Text might be too short or empty."
                )

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_embedding = self.get_embedding(chunk)
                chunk_embedding_blob = pickle.dumps(chunk_embedding)

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO document_chunks
                    (chunk_id, doc_id, chunk_text, chunk_embedding, chunk_index)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (chunk_id, doc_id, chunk, chunk_embedding_blob, i),
                )

            conn.commit()
            conn.close()

            # Update in-memory index
            self.load_existing_embeddings()  # This ensures FAISS is up-to-date

            print(f"Added document: {doc_id} with {len(chunks)} chunks")

        except Exception as e:
            print(f"Error adding document {doc_id}: {e}")

    def load_existing_embeddings(self):
        """Load existing embeddings into FAISS index"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Load document chunks for better retrieval
            cursor.execute(
                """
                SELECT chunk_id, doc_id, chunk_text, chunk_embedding,
                       documents.metadata
                FROM document_chunks
                JOIN documents ON document_chunks.doc_id = documents.id
                ORDER BY doc_id, chunk_index
            """
            )

            chunks_data = (
                cursor.fetchall()
            )  # Renamed to avoid conflict with `chunks` variable in add_document
            conn.close()

            # Reset index and related lists before loading
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.doc_ids = []
            self.doc_texts = []
            self.doc_metadatas = []

            if chunks_data:
                embeddings_list = []  # Renamed for clarity
                for (
                    chunk_id,
                    doc_id,
                    chunk_text,
                    embedding_blob,
                    metadata_json,
                ) in chunks_data:
                    embedding = pickle.loads(embedding_blob)
                    embeddings_list.append(embedding)  # Corrected variable name

                    self.doc_ids.append(chunk_id)
                    self.doc_texts.append(chunk_text)
                    self.doc_metadatas.append(json.loads(metadata_json))

                # Add to FAISS index
                if embeddings_list:  # Ensure there's something to add
                    embeddings_array = np.array(embeddings_list, dtype=np.float32)
                    # Normalize embeddings before adding to IndexFlatIP if they weren't already
                    # faiss.normalize_L2(embeddings_array) # Already done by normalize_embeddings=True in get_embedding
                    self.index.add(embeddings_array)
                    print(
                        f"Loaded {len(chunks_data)} chunks into FAISS index. Index total: {self.index.ntotal}"
                    )
                else:
                    print("No embeddings found in the database to load into FAISS.")
            else:
                print("No chunks found in database to load into FAISS index.")

        except Exception as e:
            print(f"Error loading embeddings: {e}")
            # Ensure index is reset even on error to avoid inconsistent state
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.doc_ids = []
            self.doc_texts = []
            self.doc_metadatas = []

    def semantic_search(
        self, query: str, n_results: int = 5, score_threshold: float = 0.1
    ) -> Dict:
        """Perform semantic search using embeddings"""
        try:
            if self.index.ntotal == 0:
                print("Warning: FAISS index is empty. Cannot perform semantic search.")
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}

            # Generate query embedding
            query_embedding = self.get_embedding(query)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            # faiss.normalize_L2(query_embedding) # Already done by normalize_embeddings=True

            # Search in FAISS index
            scores, indices = self.index.search(
                query_embedding, k=min(n_results, self.index.ntotal)
            )  # Ensure k <= ntotal

            # Filter by score threshold
            valid_results = [
                (score, idx)
                for score, idx in zip(scores[0], indices[0])
                if score >= score_threshold and idx != -1  # idx can be -1 if k > ntotal
            ]

            if not valid_results:
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}

            # Extract results
            documents = []
            result_scores = []
            metadatas = []
            ids = []

            for score, idx in valid_results:
                if 0 <= idx < len(self.doc_texts):  # Boundary check
                    documents.append(self.doc_texts[idx])
                    result_scores.append(float(score))
                    metadatas.append(self.doc_metadatas[idx])
                    ids.append(self.doc_ids[idx])
                else:
                    print(f"Warning: Invalid index {idx} from FAISS search.")

            return {
                "documents": documents,
                "scores": result_scores,
                "metadatas": metadatas,
                "ids": ids,
            }

        except Exception as e:
            print(f"Error in semantic search: {e}")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    def hybrid_search(self, query: str, n_results: int = 5) -> Dict:
        """Combine semantic search with keyword matching"""
        # Get semantic search results
        semantic_results = self.semantic_search(
            query, n_results * 2
        )  # Get more results initially

        # Get keyword search results (from original method)
        # This keyword search looks in full documents, not chunks.
        # For consistency, it might be better to keyword search through self.doc_texts (chunks)
        # or ensure semantic results refer to full doc IDs if needed for keyword merging.
        # Current implementation of simple_keyword_search uses the 'documents' table.
        keyword_results_raw = self.simple_keyword_search(query, n_results * 2)

        # Combine and rank results
        combined_results = {}  # Use chunk_id or a unique doc identifier as key

        # Add semantic results with higher weight (these are chunk-based)
        for i, (doc_chunk, score, metadata, chunk_id) in enumerate(
            zip(
                semantic_results["documents"],
                semantic_results["scores"],
                semantic_results["metadatas"],
                semantic_results["ids"],
            )
        ):
            combined_results[chunk_id] = {
                "document": doc_chunk,  # This is a chunk
                "semantic_score": score,
                "keyword_score": 0,  # Will be updated if keyword search matches this chunk
                "metadata": metadata,  # Metadata is from the parent document
                "id": chunk_id,
                "final_score": score * 0.7,  # 70% weight for semantic
            }

        # Add keyword results (these are full document-based from simple_keyword_search)
        # This part is tricky because simple_keyword_search returns full docs, not chunks.
        # We need a strategy to map these to chunks or re-evaluate.
        # For simplicity, let's assume keyword results are distinct unless we refine this.
        # A better hybrid search would run keyword search on the same corpus (chunks)
        # or use document IDs from chunks to boost scores of related full docs.

        # Let's re-think keyword integration.
        # Option 1: Keyword search on chunks (self.doc_texts)
        # Option 2: Use keyword results as separate items and then re-rank.

        # For now, let's keep it simple and add keyword scores if the *query* matches the *chunk* text.
        # This is not what simple_keyword_search does.
        # Let's adjust `simple_keyword_search` or create a chunk-based one.
        # Given the current structure, `simple_keyword_search` works on full documents.
        # A true hybrid on chunks would require `simple_keyword_search_on_chunks`.

        # Sticking to the provided structure: simple_keyword_search returns full docs.
        # This makes direct merging by ID difficult if semantic search gives chunk IDs.
        # Let's make semantic_results store original doc_id in metadata if possible, or retrieve it.
        # The current metadata is per document, so that's fine.

        # For now, the hybrid approach will be somewhat naive:
        # It will add semantic chunk scores.
        # It will add keyword full-doc scores.
        # If a full doc from keyword search happens to contain a chunk from semantic search,
        # it's hard to reconcile scores without more complex ID mapping.

        # Let's assume for now that keyword results are treated as separate entries,
        # and their "id" is distinct from chunk_id.
        query_words_kw = set(re.findall(r"\w+", query.lower()))

        for doc_id_sem, result_data in list(
            combined_results.items()
        ):  # Iterate over a copy
            chunk_text_words = set(re.findall(r"\w+", result_data["document"].lower()))
            common_words = query_words_kw.intersection(chunk_text_words)
            if common_words and query_words_kw:  # Check query_words_kw is not empty
                kw_score_for_chunk = len(common_words) / len(
                    query_words_kw.union(chunk_text_words)
                )
                result_data["keyword_score"] = kw_score_for_chunk
                result_data["final_score"] = (
                    result_data["semantic_score"] * 0.7 + kw_score_for_chunk * 0.3
                )

        # Sort by final score and return top results
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["final_score"], reverse=True
        )[:n_results]

        return {
            "documents": [r["document"] for r in sorted_results],
            "scores": [r["final_score"] for r in sorted_results],
            "metadatas": [r["metadata"] for r in sorted_results],
            "ids": [r["id"] for r in sorted_results],  # Return chunk_ids
            "semantic_scores": [r["semantic_score"] for r in sorted_results],
            "keyword_scores": [r["keyword_score"] for r in sorted_results],
        }

    def simple_keyword_search(self, query: str, n_results: int = 3) -> Dict:
        """Original keyword-based search on full documents as fallback"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT id, text, metadata FROM documents")
            all_docs = cursor.fetchall()
            conn.close()

            if not all_docs:
                return {
                    "documents": [],
                    "scores": [],
                    "metadatas": [],
                    "ids": [],
                }  # Added ids

            query_words = set(re.findall(r"\w+", query.lower()))
            if not query_words:  # Handle empty query
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}

            scored_docs = []

            for doc_id, text, metadata_json in all_docs:
                text_words = set(re.findall(r"\w+", text.lower()))
                common_words = query_words.intersection(text_words)

                if common_words:
                    # Jaccard similarity or simple count based
                    score = len(common_words) / len(query_words.union(text_words))
                    scored_docs.append(
                        (score, text, json.loads(metadata_json), doc_id)
                    )  # Added doc_id

            scored_docs.sort(reverse=True, key=lambda x: x[0])
            top_docs = scored_docs[:n_results]

            return {
                "documents": [doc[1] for doc in top_docs],
                "scores": [doc[0] for doc in top_docs],
                "metadatas": [doc[2] for doc in top_docs],
                "ids": [doc[3] for doc in top_docs],  # Added ids
            }

        except Exception as e:
            print(f"Error in keyword search: {e}")
            return {
                "documents": [],
                "scores": [],
                "metadatas": [],
                "ids": [],
            }  # Added ids

    def generate_answer_with_qa_model(self, question: str, context: str) -> str:
        """Use QA model to generate specific answers"""
        if not self.qa_pipeline:
            return context[:500] + "..." if len(context) > 500 else context

        try:
            # Limit context length for QA model (DistilBERT has max sequence length typically 512 tokens)
            # It's better to use model's tokenizer to truncate properly if being precise.
            # For simplicity, character limit.
            max_context_char_length = (
                3000  # Approx 500-700 words, should be fine for most QA models.
            )
            # Transformers pipeline handles truncation if context + question > max_seq_len

            # The pipeline itself should handle truncation of context if it's too long.
            # We provide the full context chunk.
            result = self.qa_pipeline(question=question, context=context)
            answer = result["answer"]
            confidence = result["score"]

            # If confidence is low, or answer is very short/generic, consider returning context.
            if (
                confidence < 0.1 or len(answer) < 10
            ):  # Adjusted threshold and added length check
                return context[:500] + "..." if len(context) > 500 else context

            return answer

        except Exception as e:
            print(f"Error in QA generation: {e}")
            # Fallback if QA model fails
            return context[:500] + "..." if len(context) > 500 else context

    def search(self, query: str, n_results: int = 3, use_hybrid: bool = True) -> Dict:
        """Main search method - can use hybrid or semantic search"""
        if use_hybrid:
            return self.hybrid_search(query, n_results)
        else:
            # Ensure semantic search also returns 'ids'
            results = self.semantic_search(query, n_results)
            # Ensure consistent return structure if hybrid adds more keys
            results["semantic_scores"] = results[
                "scores"
            ]  # if semantic_only, sem_score is the main score
            results["keyword_scores"] = [0.0] * len(
                results["documents"]
            )  # No keyword score in pure semantic
            return results

    def get_all_documents(self):
        """Get all documents in the knowledge base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT id, text, metadata FROM documents")
            docs = cursor.fetchall()
            conn.close()

            return {
                "ids": [doc[0] for doc in docs],
                "documents": [doc[1] for doc in docs],
                "metadatas": [
                    json.loads(doc[2]) if doc[2] else {} for doc in docs
                ],  # Handle null metadata
            }

        except Exception as e:
            print(f"Error getting documents: {e}")
            return {"ids": [], "documents": [], "metadatas": []}


# Initialize enhanced RAG knowledge base
print("Initializing Enhanced RAG Knowledge Base...")
print("Loading embedding model (this may take a moment)...")
kb = (
    EnhancedRAGKnowledgeBase()
)  # This is where the OSError might occur if cache is problematic
print("✅ RAG Knowledge Base initialized!")


# Keep all the existing greeting and response functions...
def get_time_based_greeting():
    """Get time-appropriate greeting"""
    current_hour = datetime.now().hour

    if 5 <= current_hour < 12:
        return random.choice(
            [
                "Good morning! ☀️",
                "Morning! 🌅",
                "Good morning! Hope you're having a great start to your day!",
                "Morning! Ready to tackle the day?",
            ]
        )
    elif 12 <= current_hour < 17:
        return random.choice(
            [
                "Good afternoon! ☀️",
                "Afternoon! 👋",
                "Good afternoon! Hope your day is going well!",
                "Hey there! How's your afternoon going?",
            ]
        )
    elif 17 <= current_hour < 21:
        return random.choice(
            [
                "Good evening! 🌆",
                "Evening! 👋",
                "Good evening! Hope you had a productive day!",
                "Hey! How was your day?",
            ]
        )
    else:
        return random.choice(
            [
                "Hello! Working late? 🌙",
                "Hi there! Burning the midnight oil?",
                "Hey! Hope you're not working too hard!",
                "Hello! 👋",
            ]
        )


def detect_greeting(text):
    """Enhanced greeting detection"""
    text_lower = text.lower().strip()
    # More robust: allow for punctuation directly after greeting
    greetings = [
        "hello",
        "hi",
        "hey",
        "hiya",
        "howdy",
        "sup",
        "what's up",
        "whats up",
        "good morning",
        "morning",
        "good afternoon",
        "afternoon",
        "good evening",
        "evening",  # "good night", "goodnight", # Good night is often a farewell
        "yo",
        "hola",
        "bonjour",
        "guten tag",
        "namaste",
        "how are you",
        "how's it going",
        "hows it going",
        "how do you do",
        "nice to meet you",
        "pleased to meet you",
    ]

    # Check for exact match or start with greeting + space/punctuation
    for greeting in greetings:
        if (
            text_lower == greeting
            or text_lower.startswith(greeting + " ")
            or text_lower.startswith(greeting + ",")
            or text_lower.startswith(greeting + "!")
            or text_lower.startswith(greeting + ".")
        ):
            return True

    # Regex for more flexible patterns
    greeting_patterns = [
        r"^(hi|hello|hey|yo)[\s!,.]*$",  # Matches "hi", "hello!", "hey," etc.
        r"^(good\s*(morning|afternoon|evening))[\s!,.]*$",
        r"^(how\s*(are\s*you|'s\s*it\s*going|s\s*it\s*going|do\s*you\s*do))[\s!,.?]*$",
        r"^\w*(morning|afternoon|evening)\w*[\s!,.]*$",  # Catches "mornin", "evenin"
    ]

    for pattern in greeting_patterns:
        if re.match(pattern, text_lower):
            return True

    return False


def generate_greeting_response():
    """Generate a friendly greeting response"""
    time_greeting = get_time_based_greeting()
    follow_ups = [
        "How can I help you today?",
        "What can I assist you with?",
        "I'm here to help! What do you need?",
        "Ready to help with any questions you have!",
        "What would you like to know?",
        "I can help answer questions from my knowledge base. What's on your mind?",
        "Feel free to ask me anything!",
    ]
    follow_up = random.choice(follow_ups)
    extras = ["😊", "👋", "🤖", "✨", "", "", ""]  # Increased chance of no emoji
    extra = random.choice(extras)

    if extra:
        return f"{time_greeting} {follow_up} {extra}"
    else:
        return f"{time_greeting} {follow_up}"


def detect_thanks(text):
    """Enhanced thanks detection"""
    text_lower = text.lower().strip()
    # More comprehensive list, including phrases that imply satisfaction
    thanks_phrases = [
        "thank you",
        "thanks",
        "thx",
        "ty",
        "thank u",
        "thankyou",
        "much appreciated",
        "appreciate it",
        "appreciate that",
        "i appreciate that",
        "grateful",
        "cheers",
        "awesome",
        "perfect",
        "great",
        "wonderful",
        "fantastic",
        "excellent",
        "that helps",
        "that's helpful",
        "very helpful",
        "super helpful",
        "exactly what i needed",
        "that's perfect",
        "that was it",
        "got it",
        "ok thanks",
        "alright thanks",
        "cool thanks",
    ]
    # Check if any phrase is a substring. For some, ensure they are not part of a larger negative context.
    # For simplicity, direct substring check is often good enough for a chatbot.
    for phrase in thanks_phrases:
        if phrase in text_lower:
            # Avoid cases like "no thanks" being detected as thanks
            if "no " + phrase == text_lower or "not " + phrase in text_lower:
                continue
            return True
    return False


def generate_thanks_response():
    """Generate varied responses to thanks"""
    responses = [
        "You're very welcome! 😊",
        "Happy to help! 👍",
        "Glad I could assist! 🤖",
        "No problem at all!",
        "You're welcome! Anything else I can help with?",
        "My pleasure! Feel free to ask if you need anything else.",
        "Anytime! Let me know if you have more questions.",
        "You're welcome! That's what I'm here for! ✨",
        "Glad that helped! 👋",
        "You bet! Hope that solved your question!",
        "No worries! Let me know if there's more I can do.",
    ]
    return random.choice(responses)


def detect_help_request(text):
    """Enhanced help request detection"""
    text_lower = text.lower().strip()
    # Broader patterns for help
    help_phrases = [
        "help",
        "what can you do",
        "what do you do",
        "how can you help",
        "what are you",
        "who are you",
        "what's your purpose",
        "what is your purpose",
        "what can i ask",
        "how does this work",
        "what are your capabilities",
        "what kind of questions",
        "what do you know",
        "tell me about yourself",
        "i need assistance",
        "can you assist me",
        "guide me",
        "instructions",
    ]
    # Using "in" for substring matching which is usually fine for these phrases
    for phrase in help_phrases:
        if phrase in text_lower:
            # Avoid "this is helpful"
            if (
                "helpful" in phrase and "this is helpful" in text_lower
            ):  # "helpful" is not a help request trigger
                continue
            return True
    return False


def generate_help_response():
    """Generate helpful response about bot capabilities"""
    responses = [
        "I'm your friendly AI assistant! 🤖 I can help answer questions based on my knowledge base using advanced semantic search. Try asking about support, billing, passwords, or product features!",
        "I'm here to help! ✨ I use RAG (Retrieval-Augmented Generation) to find the most relevant information from my knowledge base. This includes topics like:\n• Technical support and contact info\n• Password reset procedures\n• Billing and payment questions\n• Product features and capabilities\n\nJust ask me anything!",
        "Hi! I'm an AI bot that uses semantic search to understand and answer your questions. 👋 I can find information about support hours, account issues, billing, and product features. What would you like to know?",
        "I can help you find information quickly using AI-powered search! 🔍 I understand context and meaning, not just keywords. Ask me about support, billing, passwords, and product features!",
        "I'm an AI designed to provide information from our knowledge base. You can ask me about various topics. For example: 'How do I reset my password?' or 'What are the support hours?'",
    ]
    return random.choice(responses)


def enhanced_generate_response(
    question: str,
    context_docs: List[str],
    scores: List[float] = None,
    doc_ids: List[str] = None,
) -> str:
    """Enhanced response generation with RAG capabilities"""
    question_lower = question.lower().strip()

    # Handle greetings first
    if detect_greeting(question):  # Make sure greeting detection is robust
        return generate_greeting_response()

    # Handle thanks
    if detect_thanks(question):  # Make sure thanks detection is robust
        return generate_thanks_response()

    # Handle help requests
    if detect_help_request(question):  # Make sure help detection is robust
        return generate_help_response()

    # Handle goodbye/farewell
    farewell_words = [
        "bye",
        "goodbye",
        "see you",
        "farewell",
        "take care",
        "later",
        "cya",
        "good night",
        "goodnight",
    ]
    # Using `any` with `in` for substring check is fine for farewells
    if any(word in question_lower for word in farewell_words):
        farewell_responses = [
            "Goodbye! Have a great day! 👋",
            "See you later! Feel free to come back anytime! 😊",
            "Take care! I'll be here whenever you need help! ✨",
            "Bye! Hope I was helpful! 🤖",
            "Farewell! Don't hesitate to reach out if you need anything!",
            "Good night! Sleep well! 🌙",
        ]
        return random.choice(farewell_responses)

    # Enhanced knowledge-based response
    if not context_docs or not any(
        context_docs
    ):  # Ensure context_docs itself is not None
        return (
            "I couldn't find relevant information in my knowledge base to answer your question. "
            "Could you please rephrase or ask about something else? 🤔\n\n"
            "💡 Try asking about: support hours, password reset, billing, or product features!"
        )

    # Use the best matching context
    best_context = context_docs[0] if context_docs else ""
    confidence_score = (
        scores[0] if scores and scores[0] is not None else 0
    )  # Handle None score

    # Try to use QA model for better answers
    if (
        hasattr(kb, "qa_pipeline")
        and kb.qa_pipeline
        and best_context
        and len(best_context.strip()) > 20
    ):  # Ensure context is substantial
        generated_answer = kb.generate_answer_with_qa_model(question, best_context)

        # Add confidence indicator based on retrieval score
        confidence_emoji = (
            "🎯"
            if confidence_score > 0.7
            else "📈" if confidence_score > 0.5 else "💡"  # Adjusted thresholds/emojis
        )

        # Check if QA model returned the context or a short/unhelpful answer
        if (
            generated_answer
            == (best_context[:500] + "..." if len(best_context) > 500 else best_context)
            or len(generated_answer) < 20
        ):
            # Fallback to context if QA answer is not good
            response_text = best_context
            if len(response_text) > 700:  # Truncate long contexts a bit more
                response_text = response_text[:700] + "..."
            return (
                f"{confidence_emoji} Here's some information from my knowledge base that might be relevant:\n\n{response_text}\n\n"
                f"Is this helpful? Let me know if you need more specific information! 😊"
            )
        else:
            return (
                f"{confidence_emoji} Based on my knowledge base, here's what I found:\n\n_{generated_answer}_\n\n"  # Emphasize QA answer
                f"Is this helpful? Let me know if you need more specific information! 😊 (Source relevance: {confidence_score:.2f})"
            )
    else:
        # Fallback to context-based response (if QA model not available or context too short)
        response_text = best_context
        if len(response_text) > 700:
            response_text = response_text[:700] + "..."

        confidence_emoji = (
            "🎯" if confidence_score > 0.7 else "📈" if confidence_score > 0.5 else "💡"
        )

        return (
            f"{confidence_emoji} Here's some information from my knowledge base:\n\n{response_text}\n\n"
            f"Is this helpful? Let me know if you need more specific information! 😊 (Relevance: {confidence_score:.2f})"
        )


def handle_message(event):
    """Process incoming message with enhanced RAG"""
    try:
        channel = event["channel"]
        # user = event["user"] # User variable not used currently
        text = event.get("text", "")  # Use .get for safety

        if not text:  # Ignore empty messages
            return

        # Remove bot mention from text
        # Ensure SLACK_BOT_USER_ID is fetched correctly and is not None
        bot_user_id = os.environ.get("SLACK_BOT_USER_ID")
        if bot_user_id:
            text = text.replace(f"<@{bot_user_id}>", "").strip()

        if not text:  # Ignore messages that only contained a mention
            # Optionally, post a generic "Hi, how can I help?" if only mentioned.
            # For now, just return.
            return

        app.logger.info(f"Processing question from channel {channel}: '{text}'")

        # Check if it's a greeting, thanks, or help request first
        # These don't require context_docs, so pass empty list or None
        if (
            detect_greeting(text)
            or detect_thanks(text)
            or detect_help_request(text)
            or any(
                word in text.lower().strip()
                for word in ["bye", "goodbye", "good night"]
            )
        ):  # also check farewell here
            response = enhanced_generate_response(text, [], None, None)
        else:
            # Use enhanced RAG search
            search_results = kb.search(text, n_results=3, use_hybrid=True)

            if search_results and search_results.get(
                "documents"
            ):  # Check search_results is not None
                context_docs = search_results["documents"]
                scores = search_results.get("scores")  # Use .get for safety
                doc_ids = search_results.get("ids")
                app.logger.info(
                    f"Found {len(context_docs)} relevant document chunks. Top score: {scores[0] if scores else 'N/A'}"
                )
                response = enhanced_generate_response(
                    text, context_docs, scores, doc_ids
                )
            else:
                app.logger.info(f"No relevant documents found for query: '{text}'")
                response = enhanced_generate_response(
                    text, [], None, None
                )  # Pass empty context

        # Send response to Slack
        slack_client.chat_postMessage(
            channel=channel, text=response, thread_ts=event.get("ts")  # Use .get for ts
        )

        app.logger.info(f"Sent response (first 100 chars): {response[:100]}...")

    except SlackApiError as e:
        app.logger.error(f"Slack API Error handling message: {e.response['error']}")
    except Exception as e:
        app.logger.error(
            f"Error handling message: {e}", exc_info=True
        )  # Log full traceback
        try:
            slack_client.chat_postMessage(
                channel=event.get("channel"),  # Use .get for safety
                text="Sorry, I encountered an error processing your request. Please try again. 😅",
                thread_ts=event.get("ts"),  # Use .get for safety
            )
        except Exception as slack_err:  # Catch error during error reporting
            app.logger.error(f"Failed to send error message to Slack: {slack_err}")


# Keep all existing Flask routes but update some for enhanced features...


@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle Slack events"""
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    if data.get("type") == "url_verification":
        return jsonify({"challenge": data["challenge"]})

    if data.get("type") == "event_callback":
        event = data.get("event")
        if not event:
            return jsonify({"status": "ok", "message": "No event payload"}), 200

        # Ignore bot messages and subtypes like message_changed, message_deleted
        if event.get("subtype") or event.get("bot_id"):
            return jsonify(
                {"status": "ok", "message": "Ignored bot message or subtype"}
            )

        # Handle app_mention (bot is tagged) or direct messages
        # For DMs, event["type"] is "message" and event["channel_type"] is "im"
        # For channel messages where bot is mentioned, event["type"] is "app_mention"
        if event.get("type") == "app_mention":
            handle_message(event)
        elif (
            event.get("type") == "message" and event.get("channel_type") == "im"
        ):  # Direct Message
            handle_message(event)
        # Add handling for messages in channels where bot is a member but not mentioned (if desired)
        # elif event.get("type") == "message" and event.get("channel_type") == "channel":
        # This would make the bot respond to all messages in channels it's in.
        # Usually, you only want it to respond to mentions or DMs.
        # pass

    return jsonify({"status": "ok"})


@app.route("/add_knowledge", methods=["POST"])
def add_knowledge():
    """Enhanced endpoint to add documents with embeddings"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        doc_id = data.get("id")
        text = data.get("text")
        metadata = data.get("metadata", {})  # Default to empty dict

        if not doc_id or not text:
            return jsonify({"error": "Missing id or text"}), 400

        kb.add_document(doc_id, text, metadata)
        return jsonify(
            {
                "status": "success",
                "message": f"Document '{doc_id}' added to RAG knowledge base with embeddings.",
                "doc_id": doc_id,
            }
        )

    except Exception as e:
        app.logger.error(f"Error adding knowledge: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/test_search", methods=["POST"])
def test_search():
    """Test enhanced search functionality"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        query = data.get("query", "")
        search_type = data.get(
            "search_type", "hybrid"
        ).lower()  # Normalize to lowercase

        if not query:
            return jsonify({"error": "Missing query"}), 400

        if search_type == "semantic":
            results = kb.semantic_search(query, n_results=3)
        elif search_type == "keyword":
            results = kb.simple_keyword_search(query, n_results=3)
        elif search_type == "hybrid":
            results = kb.hybrid_search(query, n_results=3)
        else:
            return (
                jsonify(
                    {
                        "error": f"Invalid search_type: {search_type}. Choose hybrid, semantic, or keyword."
                    }
                ),
                400,
            )

        # Standardize result format for consistent API response
        formatted_results = []
        if results and results.get("documents"):
            for i in range(len(results["documents"])):
                formatted_results.append(
                    {
                        "document": results["documents"][i],
                        "score": (
                            results["scores"][i]
                            if results.get("scores") and i < len(results["scores"])
                            else None
                        ),
                        "id": (
                            results["ids"][i]
                            if results.get("ids") and i < len(results["ids"])
                            else None
                        ),
                        "metadata": (
                            results["metadatas"][i]
                            if results.get("metadatas")
                            and i < len(results["metadatas"])
                            else {}
                        ),
                        "semantic_score": (
                            results.get("semantic_scores", [])[i]
                            if results.get("semantic_scores")
                            and i < len(results["semantic_scores"])
                            else (
                                results["scores"][i]
                                if search_type == "semantic"
                                else None
                            )
                        ),
                        "keyword_score": (
                            results.get("keyword_scores", [])[i]
                            if results.get("keyword_scores")
                            and i < len(results["keyword_scores"])
                            else (
                                results["scores"][i]
                                if search_type == "keyword"
                                else None
                            )
                        ),
                    }
                )

        return jsonify(
            {
                "query": query,
                "search_type": search_type,
                "results": formatted_results,
            }
        )

    except Exception as e:
        app.logger.error(f"Error in test_search: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/list_knowledge", methods=["GET"])
def list_knowledge():
    """List all documents in knowledge base"""
    try:
        docs_data = kb.get_all_documents()  # Renamed to avoid conflict
        return jsonify(
            {
                "total_documents": len(docs_data["ids"]),
                "embedding_model": kb.model_name,
                "vector_dimension": kb.embedding_dim,
                "total_chunks_in_faiss": (
                    kb.index.ntotal if kb.index else 0
                ),  # Check if index exists
                "documents": [
                    {
                        "id": docs_data["ids"][i],
                        "text_preview": (
                            docs_data["documents"][i][:200] + "..."
                            if len(docs_data["documents"][i]) > 200
                            else docs_data["documents"][i]
                        ),
                        "metadata": (
                            docs_data["metadatas"][i]
                            if i < len(docs_data["metadatas"])
                            else {}
                        ),
                    }
                    for i in range(len(docs_data["ids"]))
                ],
            }
        )
    except Exception as e:
        app.logger.error(f"Error in list_knowledge: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Enhanced health check endpoint"""
    all_docs_data = kb.get_all_documents()  # Fetch once
    return jsonify(
        {
            "status": "healthy",
            "rag_enabled": True,
            "embedding_model": kb.model_name,
            "vector_dimension": kb.embedding_dim,
            "total_documents_in_db": len(all_docs_data["ids"]),
            "total_chunks_in_faiss": kb.index.ntotal if kb.index else 0,
            "qa_model_loaded": kb.qa_pipeline is not None,
        }
    )


@app.route("/")
def home():
    """Enhanced home page"""
    # Fetch dynamic data carefully, handle potential None for kb.index
    total_docs = len(kb.get_all_documents()["ids"])
    total_chunks = kb.index.ntotal if kb.index else "N/A (Index not initialized)"
    qa_status = "✅ Loaded" if kb.qa_pipeline else "❌ Not loaded (or failed to load)"

    return f"""
    <h1>🚀 Enhanced RAG-Powered Slack AI Bot</h1>
    <p>Bot is running with advanced RAG capabilities! 🤖✨</p>
    
    <h2>🔧 RAG Features:</h2>
    <ul>
        <li><strong>Semantic Search:</strong> Using SentenceTransformers ('{kb.model_name}') for understanding context</li>
        <li><strong>Vector Embeddings:</strong> FAISS index for fast similarity search (Inner Product)</li>
        <li><strong>Text Chunking:</strong> Intelligent document splitting for better retrieval</li>
        <li><strong>Hybrid Search:</strong> Combines semantic + keyword matching (TF-IDF like on chunks)</li>
        <li><strong>QA Model:</strong> {'DistilBERT (distilbert-base-cased-distilled-squad)' if kb.qa_pipeline else 'Not Loaded'} for generating specific answers</li>
        <li><strong>Confidence Scoring:</strong> Shows relevance scores for results</li>
    </ul>
    
    <h2>📊 Current Stats:</h2>
    <ul>
        <li><strong>Embedding Model:</strong> {kb.model_name}</li>
        <li><strong>Vector Dimension:</strong> {kb.embedding_dim}</li>
        <li><strong>Total Documents in DB:</strong> {total_docs}</li>
        <li><strong>Total Chunks in FAISS:</strong> {total_chunks}</li>
        <li><strong>QA Model Status:</strong> {qa_status}</li>
    </ul>
    
    <h2>🛠️ Available Endpoints:</h2>
    <ul>
        <li><strong>POST /slack/events</strong> - Slack events webhook (for app_mention and DMs)</li>
        <li><strong>POST /add_knowledge</strong> - Add documents with embeddings to the knowledge base</li>
        <li><strong>GET /list_knowledge</strong> - View all documents stored in the knowledge base</li>
        <li><strong>POST /test_search</strong> - Test search functionality (types: hybrid, semantic, keyword)</li>
        <li><strong>GET /health</strong> - Health check endpoint with RAG statistics</li>
    </ul>
    
    <h2>🎯 Search Types:</h2>
    <ul>
        <li><strong>Hybrid Search:</strong> Combines semantic similarity on chunks with keyword scores on chunks (default).</li>
        <li><strong>Semantic Search:</strong> Uses vector embeddings to find semantically similar chunks.</li>
        <li><strong>Keyword Search:</strong> Traditional TF-IDF like keyword matching on full documents (from DB).</li>
    </ul>
    
    <h2>💡 Example Usage (cURL):</h2>
    <pre>
    # Test hybrid search (default)
    curl -X POST http://localhost:3000/test_search \\
      -H "Content-Type: application/json" \\
      -d '{{"query": "how to reset password"}}'
    
    # Test semantic search
    curl -X POST http://localhost:3000/test_search \\
      -H "Content-Type: application/json" \\
      -d '{{"query": "information about billing", "search_type": "semantic"}}'
    
    # Add a new document to the knowledge base
    curl -X POST http://localhost:3000/add_knowledge \\
      -H "Content-Type: application/json" \\
      -d '{{
        "id": "new_doc_example_1",
        "text": "This is a new sample document about cloud services and their benefits.",
        "metadata": {{"category": "cloud", "priority": "medium"}}
      }}'
    </pre>
    
    <p><em>🔍 In Slack, try asking: "@[YourBotName] How do I reset my password?", "What are your support hours?", "Tell me about billing"</em></p>
    """


def initialize_sample_data():
    """Initialize with enhanced sample knowledge base data if DB is empty"""
    sample_docs = [
        {
            "id": "support_hours_detailed",
            "text": "Our comprehensive technical support team is available to assist you Monday through Friday, 9:00 AM to 5:00 PM Eastern Standard Time. You can reach our support specialists by emailing enquiery@nervesparks.in or calling our dedicated support line at 88000-10366. For urgent technical issues that occur outside of our standard business hours, please email our emergency support address at varun@nervesparks.in. We also offer premium 24/7 support for enterprise customers. Our support team specializes in troubleshooting, account management, technical guidance, and product assistance.",
            "metadata": {
                "category": "support",
                "type": "hours",
                "priority": "high",
                "keywords": [
                    "support",
                    "hours",
                    "contact",
                    "email",
                    "phone",
                    "EST",
                    "emergency",
                ],
            },
        },
        {
            "id": "password_reset_comprehensive",
            "text": "To reset your account password, please follow these detailed steps: 1) Navigate to the login page on our website 2) Click the 'Forgot Password' link located below the login form 3) Enter your registered email address in the provided field 4) Check your email inbox for a password reset message (also check spam folder) 5) Click the secure reset link in the email within 24 hours 6) Create a new strong password with at least 8 characters, including uppercase, lowercase, numbers, and special characters 7) Confirm your new password and save changes. The reset link expires after 24 hours for security purposes. If you don't receive the email, please contact support. For additional security, we recommend enabling two-factor authentication (2FA) after resetting your password.",
            "metadata": {
                "category": "account",
                "type": "password",
                "priority": "high",
                "keywords": [
                    "password",
                    "reset",
                    "forgot",
                    "login",
                    "security",
                    "authentication",
                    "2FA",
                    "credentials",
                ],
            },
        },
        {
            "id": "billing_comprehensive",
            "text": "For all billing inquiries, account charges, payment issues, and invoice questions, please contact our dedicated billing department. You can reach them via email at billing@company.com or by calling our billing hotline at 1-800-123-4568 during business hours (Monday-Friday, 9 AM - 5 PM Local Time). Our billing team handles subscription management, payment processing, refund requests, and account upgrades. Monthly invoices are automatically generated and sent to your registered email address. Payment is due within 30 days of the invoice date. We accept various payment methods including major credit cards (Visa, MasterCard, American Express), bank transfers (ACH/Wire), and PayPal for select plans. For enterprise customers, we offer flexible payment terms and consolidated billing options. You can also access your billing history, download invoices, and update payment methods through your account dashboard on our website.",
            "metadata": {
                "category": "billing",
                "type": "general",
                "priority": "medium",
                "keywords": [
                    "billing",
                    "payment",
                    "invoice",
                    "charges",
                    "subscription",
                    "refund",
                    "credit card",
                    "ACH",
                    "invoice",
                ],
            },
        },
        {
            "id": "product_features_detailed",
            "text": "Our comprehensive software platform includes a wide range of powerful features designed to enhance your productivity and streamline your workflow. Key features include: Advanced Analytics Dashboard with real-time data visualization and custom reporting capabilities; Collaborative Team Tools with shared workspaces, comment systems, and task management; Robust API Access with RESTful endpoints and webhook integrations for developers; Custom Integration Support for popular third-party applications and services (e.g., Salesforce, Slack, Google Workspace); 24/7 System Monitoring with uptime tracking and performance metrics; Automated Backup and Recovery systems ensuring data integrity; Role-based Access Control (RBAC) with granular permissions for enhanced security; Mobile Applications for iOS and Android for on-the-go access; Advanced Search and Filtering capabilities across all modules; Data Export and Import tools for seamless data management; White-label Solutions for enterprise clients requiring custom branding; and Dedicated Account Management for premium customers. Our platform is designed to be scalable and adapt to your growing business needs.",
            "metadata": {
                "category": "product",
                "type": "features",
                "priority": "medium",
                "keywords": [
                    "features",
                    "analytics",
                    "API",
                    "integration",
                    "monitoring",
                    "mobile",
                    "enterprise",
                    "collaboration",
                    "RBAC",
                ],
            },
        },
        {
            "id": "security_privacy",
            "text": "We take security and privacy extremely seriously. Our platform implements enterprise-grade security measures including end-to-end SSL/TLS encryption for all data in transit, and AES-256 encryption for data at rest. We support two-factor authentication (2FA/MFA) for all user accounts. Regular independent security audits and penetration tests are conducted. Our infrastructure is compliant with major standards like SOC 2 Type II, ISO 27001, and GDPR. All data is stored in secure, geographically redundant data centers. We perform regular automated backups, and have robust disaster recovery procedures in place. Our comprehensive privacy policy, available on our website, outlines how we collect, use, store, and protect your personal information. We are committed to data minimization and never sell your data to third parties. For security concerns or to report vulnerabilities, please contact our dedicated security team at security@company.com.",
            "metadata": {
                "category": "security",
                "type": "privacy",
                "priority": "high",
                "keywords": [
                    "security",
                    "privacy",
                    "encryption",
                    "GDPR",
                    "data protection",
                    "compliance",
                    "2FA",
                    "SOC 2",
                    "ISO 27001",
                ],
            },
        },
        {
            "id": "troubleshooting_common",
            "text": "If you encounter common technical issues with our platform, try these troubleshooting steps first: 1) Clear your web browser's cache and cookies. 2) Temporarily disable any browser extensions or add-ons, as they can sometimes interfere. 3) Try accessing the platform using an incognito/private browsing window. 4) Check your internet connection stability and speed. 5) Ensure you are using a supported web browser (latest versions of Chrome, Firefox, Safari, or Edge are recommended). 6) Update your browser to its latest version. 7) Restart your computer or device. 8) Check our official status page (status.company.com) for any announcements regarding ongoing service disruptions or maintenance. If these steps do not resolve your issue, please contact our support team with detailed information about the problem. Include any error messages, the browser type and version you are using, your operating system, and the specific steps you took that led to the issue. Screenshots can also be very helpful.",
            "metadata": {
                "category": "troubleshooting",
                "type": "common_issues",
                "priority": "medium",
                "keywords": [
                    "troubleshooting",
                    "browser",
                    "cache",
                    "cookies",
                    "connection",
                    "error",
                    "technical issues",
                    "support",
                    "fix",
                ],
            },
        },
    ]

    # Check if knowledge base is empty by checking the documents table
    conn = sqlite3.connect(kb.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    conn.close()

    if doc_count == 0:
        app.logger.info(
            "🔄 Initializing RAG knowledge base with enhanced sample data as DB is empty..."
        )
        for doc_data in sample_docs:  # Renamed to avoid conflict
            kb.add_document(doc_data["id"], doc_data["text"], doc_data["metadata"])
        app.logger.info(
            f"✅ Added {len(sample_docs)} enhanced documents to RAG knowledge base."
        )
        app.logger.info(
            f"📊 Total chunks created in FAISS: {kb.index.ntotal if kb.index else 'N/A'}"
        )
    else:
        app.logger.info(
            f"📚 Knowledge base already contains {doc_count} documents. Skipping sample data initialization."
        )
        # Ensure FAISS is loaded even if not adding new sample data
        if kb.index is None or kb.index.ntotal == 0:
            app.logger.info(
                "Attempting to load existing embeddings into FAISS as it seems empty/uninitialized..."
            )
            kb.load_existing_embeddings()
        app.logger.info(
            f"🔍 Total searchable chunks in FAISS: {kb.index.ntotal if kb.index else 'N/A'}"
        )


if __name__ == "__main__":
    # Set up basic logging to console for app messages
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    app.logger.info("🚀 Starting Enhanced RAG-Powered Slack AI Bot")
    app.logger.info(
        "🤖 Advanced capabilities: Semantic search + QA model + Hybrid retrieval"
    )
    app.logger.info(
        f"🔧 Using SentenceTransformer: {kb.model_name} + FAISS + QA: {'DistilBERT' if kb.qa_pipeline else 'None'}"
    )
    # app.logger.info("📊 Initializing embedding models...") # Already done by kb instantiation

    # Initialize sample data if DB is empty
    initialize_sample_data()

    app.logger.info("\n" + "=" * 80)
    app.logger.info("🎯 Enhanced RAG Bot is ready!")
    app.logger.info("💡 Key Features:")
    app.logger.info("   • Semantic search with vector embeddings on text chunks.")
    app.logger.info("   • Intelligent text chunking for improved retrieval accuracy.")
    app.logger.info(
        "   • Hybrid search combining semantic scores with keyword-based scores on chunks."
    )
    app.logger.info(
        f"   • QA model ({'DistilBERT' if kb.qa_pipeline else 'Not loaded'}) for generating specific answers from retrieved context."
    )
    app.logger.info("   • Confidence scoring for relevance of retrieved results.")
    app.logger.info(
        "👋 In Slack, try asking: '@[YourBotName] How do I reset my password?', 'What are your support hours?' (if bot user ID is set)"
    )
    app.logger.info("📝 Add documents via API: POST /add_knowledge")
    app.logger.info(
        '🔍 Test search via API: POST /test_search (body: {"query": "your question", "search_type": "hybrid"})'
    )
    app.logger.info(
        f"🌐 Web interface with stats and examples: http://localhost:{os.environ.get('PORT', 3000)}"
    )
    app.logger.info("=" * 80 + "\n")

    # Run Flask app
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True)  # debug=True is fine for development
