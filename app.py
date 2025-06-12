import os
import json
from flask import Flask, request, jsonify, send_from_directory
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging
from dotenv import load_dotenv
import re
import random
from datetime import datetime
import numpy as np
import faiss # Keep direct faiss import if needed, but LangChain wrapper handles most use
import pickle # Needed by FAISS wrapper persistence
from typing import List, Dict, Optional
import openai # Used by langchain-openai
from pymongo.errors import PyMongoError, ConnectionFailure
from pymongo import MongoClient
import PyPDF2
import io
import time

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser # Helpful for chains

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)
app.logger.setLevel(logging.INFO) # Set Flask's specific logger level

# --- Configuration ---
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "rag_knowledge_base")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_BOT_USER_ID = os.environ.get("SLACK_BOT_USER_ID") # Need this for mention filtering
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # Primary API Key for LLM and Embeddings

# --- Initialize LLM and Embeddings (OpenAI) ---
llm = None
embeddings = None
openai_available = False

if not OPENAI_API_KEY:
    app.logger.critical(
        "CRITICAL: OPENAI_API_KEY is not set in environment variables. OpenAI features will not work."
    )
else:
    try:
        # Initialize OpenAI Chat Model for Generation
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=OPENAI_API_KEY)
        app.logger.info("OpenAI ChatModel initialized.")

        # Initialize OpenAI Embeddings Model
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
        app.logger.info("OpenAI Embeddings initialized.")
        
        # Test embedding model (optional but good practice)
        try:
            dummy_embedding = embeddings.embed_query("test query")
            if len(dummy_embedding) > 0:
                 app.logger.info(f"OpenAI embedding model seems functional (dim: {len(dummy_embedding)}).")
                 openai_available = True
                 # Update embedding dimension based on the model response
                 # text-embedding-ada-002 is 1536, text-embedding-3-small is 1536, text-embedding-3-large is 3072
                 # Let's assume text-embedding-ada-002 for simplicity unless specified
                 # The actual dimension is returned by the API call
                 FAISS_EMBEDDING_DIMENSION = len(dummy_embedding)
            else:
                 app.logger.error("OpenAI embedding model returned empty embedding.")
                 openai_available = False
                 FAISS_EMBEDDING_DIMENSION = 1536 # Default/fallback
        except Exception as e:
            app.logger.critical(f"Failed to test OpenAI embedding model: {e}. OpenAI features disabled.")
            openai_available = False
            FAISS_EMBEDDING_DIMENSION = 1536 # Default/fallback

    except Exception as e:
        app.logger.critical(f"Failed to initialize OpenAI LLM or Embeddings: {e}. Please check your key and model names.")
        openai_available = False
        FAISS_EMBEDDING_DIMENSION = 1536 # Default/fallback


# Initialize Slack client
if not SLACK_BOT_TOKEN:
    app.logger.critical(
        "CRITICAL: SLACK_BOT_TOKEN not found in environment variables. Slack integration will fail."
    )
slack_client = WebClient(token=SLACK_BOT_TOKEN or "dummy_token") # Use dummy if none to avoid crash, but warn

# Initialize External Search Tool (from langchain-community)
search_tool = None
try:
    search_tool = DuckDuckGoSearchRun()
    app.logger.info("DuckDuckGo search tool initialized.")
except Exception as e:
    app.logger.error(f"Failed to initialize DuckDuckGo search tool: {e}")


# --- Knowledge Base Class using LangChain FAISS & MongoDB ---
class LangChainRAGKnowledgeBase:
    def __init__(self, mongodb_uri: str, db_name: str, embeddings_model: OpenAIEmbeddings, faiss_dir: str = "./faiss_index_openai"):
        self.mongodb_connected = False
        self.mongo_client = None
        self.db = None
        self.documents_collection = None
        self.chunks_collection = None
        
        self.embeddings = embeddings_model
        self.faiss_dir = faiss_dir
        self.vectorstore = None # LangChain FAISS vector store
        
        self.openai_available = bool(self.embeddings) # Flag based on if embeddings model was initialized

        # MongoDB setup
        try:
            self.mongo_client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
            self.mongo_client.admin.command('ping') # Test connection
            self.db = self.mongo_client[db_name]
            self.documents_collection = self.db.documents
            self.chunks_collection = self.db.document_chunks
            
            # Create indexes for chunks (important for fetching text/metadata when rebuilding FAISS)
            self.chunks_collection.create_index("doc_id")
            self.chunks_collection.create_index("chunk_id", unique=True)
            self.chunks_collection.create_index([("doc_id", 1), ("chunk_index", 1)])

            # Create index for documents collection (optional, but good for listing/deleting)
            self.documents_collection.create_index("doc_id", unique=True)
            
            app.logger.info(f"✅ Connected to MongoDB: {mongodb_uri}")
            self.mongodb_connected = True
        except ConnectionFailure as e:
            app.logger.critical(f"❌ CRITICAL: Failed to connect to MongoDB at {mongodb_uri}: {e}")
            self.mongodb_connected = False
        except Exception as e:
            app.logger.error(f"❌ Failed during MongoDB setup: {e}")
            self.mongodb_connected = False

        # Load or build FAISS index
        if self.mongodb_connected and self.openai_available:
            self.load_or_build_faiss_index()
        elif not self.mongodb_connected:
            app.logger.warning("Skipping FAISS load/build: MongoDB not connected.")
        elif not self.openai_available:
             app.logger.warning("Skipping FAISS load/build: OpenAI Embeddings not available.")


    def load_or_build_faiss_index(self):
        """Loads FAISS index from disk or builds it from MongoDB if not found."""
        if not self.mongodb_connected or not self.openai_available:
            app.logger.warning("Cannot load/build FAISS index: Requirements not met.")
            return

        app.logger.info(f"Attempting to load FAISS index from {self.faiss_dir}...")
        try:
            # Try loading from disk
            if os.path.exists(self.faiss_dir):
                 # Ensure the index.faiss file exists within the directory
                 if os.path.exists(os.path.join(self.faiss_dir, "index.faiss")):
                    self.vectorstore = FAISS.load_local(
                        self.faiss_dir, self.embeddings, allow_dangerous_deserialization=True # Necessary for loading pickle file
                    )
                    app.logger.info(f"✅ Loaded FAISS index from {self.faiss_dir}.")
                 else:
                     app.logger.warning(f"FAISS directory {self.faiss_dir} exists, but index.faiss not found. Will attempt to rebuild.")
                     self._build_and_save_faiss_index_from_mongo()
            else:
                app.logger.info(f"FAISS directory {self.faiss_dir} not found. Will attempt to build from MongoDB.")
                self._build_and_save_faiss_index_from_mongo()

        except Exception as e:
            app.logger.error(f"Error loading FAISS index, attempting to rebuild: {e}", exc_info=True)
            self._build_and_save_faiss_index_from_mongo() # Attempt to rebuild if loading fails

    def _build_and_save_faiss_index_from_mongo(self):
        """Builds FAISS index from chunks in MongoDB and saves it."""
        if not self.mongodb_connected or not self.openai_available:
            app.logger.warning("Cannot build/save FAISS index: Requirements not met.")
            self.vectorstore = None # Ensure vectorstore is None if build fails
            return

        app.logger.info("Building FAISS index from MongoDB chunks...")
        try:
            # Fetch all chunks from MongoDB
            # We only need chunk_text and parent doc's metadata for LangChain Documents
            pipeline = [
                 {"$project": {
                     "_id": 0, # Exclude _id
                     "chunk_text": "$chunk_text",
                     "doc_id": "$doc_id", # Keep doc_id to fetch parent metadata
                     "chunk_index": "$chunk_index",
                     "chunk_id": "$chunk_id" # Keep chunk_id
                 }}
            ]
            
            # Fetch chunks and documents in parallel for better efficiency
            chunks_cursor = self.chunks_collection.find({})
            documents_cursor = self.documents_collection.find({}, {"doc_id": 1, "metadata": 1})
            
            # Map doc_id to metadata
            doc_metadata_map = {doc["doc_id"]: doc.get("metadata", {}) for doc in documents_cursor}

            documents_for_faiss = []
            for chunk in chunks_cursor:
                # Construct LangChain Document from chunk data
                # Include chunk_id and chunk_index in metadata if useful, plus parent doc metadata
                metadata = {
                    "doc_id": chunk.get("doc_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    **doc_metadata_map.get(chunk.get("doc_id"), {}) # Add parent doc metadata
                }
                documents_for_faiss.append(
                    Document(page_content=chunk.get("chunk_text", ""), metadata=metadata)
                )

            if not documents_for_faiss:
                app.logger.info("No documents found in MongoDB to build FAISS index from.")
                self.vectorstore = None # Explicitly set to None
                # Ensure FAISS directory is empty if there are no docs
                if os.path.exists(self.faiss_dir):
                    try:
                        import shutil
                        shutil.rmtree(self.faiss_dir)
                        app.logger.info(f"Removed empty FAISS directory {self.faiss_dir}.")
                    except Exception as rm_e:
                         app.logger.warning(f"Failed to remove empty FAISS directory {self.faiss_dir}: {rm_e}")

                return

            app.logger.info(f"Building FAISS index with {len(documents_for_faiss)} documents...")
            # Use LangChain's FAISS.from_documents which handles embedding and indexing
            self.vectorstore = FAISS.from_documents(documents_for_faiss, self.embeddings)

            # Save the built index to disk
            self.vectorstore.save_local(self.faiss_dir)
            app.logger.info(f"✅ Successfully built and saved FAISS index to {self.faiss_dir}.")

        except PyMongoError as e:
            app.logger.error(f"MongoDB error while building FAISS index: {e}", exc_info=True)
            self.vectorstore = None
        except Exception as e:
            app.logger.error(f"Error building/saving FAISS index: {e}", exc_info=True)
            self.vectorstore = None


    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """
        Splits text into chunks using LangChain's TextSplitter logic (simulated)
        and returns LangChain Document objects.
        This custom version directly prepares Documents for FAISS.
        """
        if not text or len(text.strip()) == 0:
            return []

        text = text.replace('\r\n', '\n').replace('\r', '\n') # Normalize newlines
        # Use regex to find natural breaks, prioritizing larger breaks
        separators = ["\n\n", "\n", ". ", "? ", "! ", " "]
        chunks = []
        last_idx = 0

        while last_idx < len(text):
            chunk_start = last_idx
            chunk_end = min(last_idx + chunk_size, len(text))

            # Find a split point within the overlap region or at the end of the chunk size
            split_point = chunk_end
            search_end = min(len(text), last_idx + chunk_size)
            search_start = max(last_idx, search_end - chunk_overlap)

            found_break = False
            for sep in separators:
                 idx = text.rfind(sep, search_start, search_end)
                 if idx != -1:
                     split_point = idx + len(sep)
                     found_break = True
                     break # Found the highest priority break

            chunk_text_content = text[last_idx:split_point].strip()
            if chunk_text_content:
                # Note: We don't add metadata like original doc_id/chunk_index here.
                # That's added when creating the final Document objects *after* chunking.
                chunks.append(chunk_text_content)

            if split_point >= len(text): # Reached end of text
                 break
            
            last_idx = split_point # Start next chunk after the split point

        return [c for c in chunks if c] # Return list of chunk strings


    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        """Adds or updates a document, chunks it, stores chunks in MongoDB, and rebuilds FAISS."""
        if not self.mongodb_connected:
            app.logger.error(f"Cannot add document {doc_id}: MongoDB is not connected.")
            raise ConnectionFailure("MongoDB not connected")

        if not self.openai_available:
            app.logger.error(f"Cannot add document {doc_id}: OpenAI API not available.")
            raise Exception("OpenAI API not available for embeddings or LLM.")

        try:
            # Store document record in MongoDB
            doc_data = {
                "doc_id": doc_id,
                "text": text, # Store full text
                "metadata": metadata or {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Use upsert to update if exists. This will replace the entire document and its chunks.
            self.documents_collection.replace_one(
                {"doc_id": doc_id}, 
                doc_data, 
                upsert=True
            )

            # Process chunks
            chunk_texts = self.chunk_text(text) # Get list of chunk strings
            
            if not chunk_texts:
                app.logger.warning(f"No chunks generated for document {doc_id}. Document added but no chunks for search.")
                # If updating, remove previous chunks for this doc
                self.chunks_collection.delete_many({"doc_id": doc_id})
                self.load_or_build_faiss_index() # Rebuild FAISS after deletion
                return

            # Remove existing chunks for this document before adding new ones
            delete_result = self.chunks_collection.delete_many({"doc_id": doc_id})
            app.logger.info(f"Deleted {delete_result.deleted_count} existing chunks for document {doc_id}")

            # Prepare chunk documents for MongoDB
            chunk_documents_mongo = []
            for i, chunk_text_content in enumerate(chunk_texts):
                chunk_id = f"{doc_id}_chunk_{i}" # Unique ID for the chunk
                chunk_doc = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id, # Reference to the parent document
                    "chunk_text": chunk_text_content,
                    "chunk_index": i,
                    "created_at": datetime.utcnow()
                }
                chunk_documents_mongo.append(chunk_doc)

            if chunk_documents_mongo:
                self.chunks_collection.insert_many(chunk_documents_mongo)
                app.logger.info(
                    f"Added {len(chunk_documents_mongo)} new chunks for document: {doc_id} in MongoDB."
                )
            else:
                 app.logger.warning(f"No chunks prepared for document {doc_id}. Document added but chunks skipped.")

            # Rebuild FAISS index from the updated state of MongoDB
            self.load_or_build_faiss_index()
            app.logger.info(f"Finished processing document {doc_id} and rebuilt FAISS index.")

        except PyMongoError as e:
            app.logger.error(f"MongoDB error adding document {doc_id}: {e}")
            raise
        except Exception as e:
            app.logger.error(f"Error adding document {doc_id}: {e}")
            raise


    def search(self, query: str, n_results: int = 5) -> Dict:
        """Performs semantic search using the LangChain FAISS vector store."""
        if self.vectorstore is None:
            app.logger.warning("FAISS vector store not initialized or empty. Cannot search.")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}
        
        if not self.openai_available:
             app.logger.error("OpenAI API not available for embeddings. Cannot search.")
             return {"documents": [], "scores": [], "metadatas": [], "ids": []}

        app.logger.info(f"Performing semantic search for query: '{query[:50]}...'")

        try:
            # Use similarity_search_with_score to get documents and their relevance scores
            results_with_scores = self.vectorstore.similarity_search_with_score(query, k=n_results)
            
            # Format results into the desired dictionary structure
            results_data = {"documents": [], "scores": [], "metadatas": [], "ids": []}
            for doc, score in results_with_scores:
                results_data["documents"].append(doc.page_content)
                results_data["scores"].append(float(score)) # Score is cosine distance, lower is better. RAG prompt needs context, not score.
                results_data["metadatas"].append(doc.metadata) # Contains parent doc metadata, chunk_id, etc.
                results_data["ids"].append(doc.metadata.get("chunk_id")) # Use chunk_id as item ID

            app.logger.info(f"Semantic search found {len(results_data['documents'])} results.")
            return results_data
        except Exception as e:
            app.logger.error(f"Error in semantic search for query '{query[:50]}...': {e}")
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    # Keyword search is less critical with LangChain's RAG chain, but can keep a simplified version
    # if you ever need pure keyword lookup. For RAG, semantic search is the primary mechanism.
    # We'll simplify the RAG flow to primarily use the RetrievalQA chain which relies on the retriever.
    # The hybrid search logic could be reimplemented within the RAG chain setup if needed.
    # For now, stick to the basic LangChain RetrievalQA using the semantic retriever.

    def get_retriever(self, search_kwargs: Dict = {"k": 5}):
         """Gets a LangChain retriever from the vector store."""
         if self.vectorstore is None:
             app.logger.warning("Vector store not initialized. Cannot get retriever.")
             return None
         return self.vectorstore.as_retriever(search_kwargs=search_kwargs)


    def get_all_documents(self):
        """Retrieves a summary of all documents from MongoDB."""
        if not self.mongodb_connected:
            app.logger.error("Cannot get all documents: MongoDB is not connected.")
            return {"documents": [], "total_documents": 0}
            
        try:
            # Fetch all documents, but only return necessary fields for summary/listing
            documents = list(self.documents_collection.find({}, {"doc_id": 1, "text": 1, "metadata": 1}))
            
            # Fetch chunk count per document
            chunk_counts_agg = list(self.chunks_collection.aggregate([
                {"$group": {"_id": "$doc_id", "count": {"$sum": 1}}}
            ]))
            chunk_counts_map = {item["_id"]: item["count"] for item in chunk_counts_agg}

            document_summaries = []
            for doc in documents:
                 document_summaries.append({
                     "id": doc.get("doc_id", "N/A"),
                     "text_preview": doc.get("text", "")[:200] + "..." if len(doc.get("text", "")) > 200 else doc.get("text", ""),
                     "metadata": doc.get("metadata", {}),
                     "chunk_count": chunk_counts_map.get(doc.get("doc_id"), 0)
                 })

            return {
                "documents": document_summaries,
                "total_documents": len(documents)
            }
        except PyMongoError as e:
            app.logger.error(f"MongoDB error getting all documents: {e}")
            return {"documents": [], "total_documents": 0}
        except Exception as e:
            app.logger.error(f"Error getting all documents: {e}")
            return {"documents": [], "total_documents": 0}


    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks from MongoDB and rebuild FAISS."""
        if not self.mongodb_connected:
            app.logger.error(f"Cannot delete document {doc_id}: MongoDB is not connected.")
            return False
            
        try:
            # Delete document record
            doc_result = self.documents_collection.delete_one({"doc_id": doc_id})
            
            # Delete chunks associated with the document
            chunk_result = self.chunks_collection.delete_many({"doc_id": doc_id})
            
            if doc_result.deleted_count > 0:
                app.logger.info(f"Deleted document {doc_id} and {chunk_result.deleted_count} chunks from MongoDB.")
                
                # Rebuild FAISS index from the remaining data in MongoDB
                self.load_or_build_faiss_index()  
                return True
            else:
                app.logger.warning(f"Document {doc_id} not found for deletion in MongoDB.")
                return False
        except PyMongoError as e:
            app.logger.error(f"MongoDB error deleting document {doc_id}: {e}")
            return False
        except Exception as e:
            app.logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def get_database_stats(self):
        """Get statistics about the MongoDB database and FAISS index."""
        stats = {
            "mongodb_connected": self.mongodb_connected,
            "database_name": MONGODB_DB_NAME,
            "total_documents_in_db": 0,
            "total_chunks_in_db": 0,
            "faiss_index_size": 0, # Number of vectors in FAISS
            "embedding_dimension": self.embeddings.client.embeddings[0].embedding_dimensions if self.embeddings and hasattr(self.embeddings, 'client') and hasattr(self.embeddings.client, 'embeddings') and self.embeddings.client.embeddings else FAISS_EMBEDDING_DIMENSION,
            "error": None
        }
        
        if not self.mongodb_connected:
            stats["error"] = "MongoDB not connected"
            return stats
            
        try:
            stats["total_documents_in_db"] = self.documents_collection.count_documents({})
            stats["total_chunks_in_db"] = self.chunks_collection.count_documents({})
            stats["faiss_index_size"] = self.vectorstore.index.ntotal if self.vectorstore and self.vectorstore.index else 0
            
        except Exception as e:
            app.logger.error(f"Error getting database stats: {e}")
            stats["error"] = str(e)
        
        return stats

# --- Initialize KB globally ---
# The KB needs embeddings, which depend on OPENAI_API_KEY. Check openai_available flag.
kb = None
if openai_available:
    kb = LangChainRAGKnowledgeBase(MONGODB_URI, MONGODB_DB_NAME, embeddings)
    app.logger.info("✅ LangChain RAG Knowledge Base initialization attempted.")
else:
    app.logger.critical("❌ LangChain RAG Knowledge Base initialization skipped due to OpenAI API unavailability.")


# --- Setup LangChain RAG Chain ---
rag_chain = None
if llm and kb and kb.vectorstore:
    # Define the RAG prompt template
    rag_prompt_template = """You are a helpful AI assistant. Answer the following question based *only* on the provided context. 
If the context doesn't contain enough information to answer the question, say so politely and state that you couldn't find the information in your knowledge base. Do not make up information.

Context:
{context}

Question: {question}

Answer:"""
    RAG_PROMPT = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"])

    # Create RetrievalQA chain
    # Uses 'stuff' chain type by default, which puts all retrieved docs in context
    # Get retriever from the initialized vectorstore
    retriever = kb.get_retriever(search_kwargs={"k": 5}) # Retrieve top 5 documents for RAG context

    if retriever:
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": RAG_PROMPT},
            return_source_documents=False, # Set to True if you want source docs in the output
        )
        app.logger.info("✅ LangChain RetrievalQA chain initialized.")
    else:
        app.logger.critical("❌ Failed to get retriever from vectorstore. RAG chain not initialized.")
        rag_chain = None # Ensure it's None if retriever failed
else:
     app.logger.critical("❌ LangChain RetrievalQA chain initialization skipped due to missing LLM, KB, or Vectorstore.")


# --- PDF Processing Utility Function ---
def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from a file-like object assumed to be a PDF."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        if reader.is_encrypted:
             app.logger.warning("PDF is encrypted, attempting to decrypt with empty password.")
             try:
                 reader.decrypt('') # Try decrypting with empty password
             except PyPDF2.errors.PasswordIncorrectError:
                 app.logger.error("PDF is encrypted and password provided is incorrect or empty.")
                 raise ValueError("Encrypted PDF requires a password.")
             except Exception as e:
                  app.logger.error(f"Error during PDF decryption attempt: {e}")
                  raise ValueError("Failed to decrypt PDF.")

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                 text += page_text + "\n\n" # Add newlines between pages
    except Exception as e:
        app.logger.error(f"Error extracting text from PDF: {e}")
        raise
    return text.strip() # Return stripped text

# --- External Search Result Parsing ---
def parse_duckduckgo_search_results(raw_results: str, max_links_per_type: int = 3, video_site_filter: str = "youtube.com") -> Dict[str, List[Dict]]:
    """
    Parses raw search results string from DuckDuckGoSearchRun into formatted links,
    categorizing broadly into web links and video links.
    """
    results = {"web": [], "videos": []}
    if not raw_results or not isinstance(raw_results, str):
        return results

    # DuckDuckGoSearchRun output format can vary, often comma-separated entries
    # or simple list-like format. This parsing is heuristic.
    # Example: "[snippet title: link, snippet title: link]" or "snippet title: link\nsnippet title: link"
    
    # Split by common separators like comma or newline followed by a potential title/link pattern
    entries = re.split(r',\s*(?=\w+: http)', raw_results) # Split by comma followed by title: http
    if len(entries) <= 1: # If comma split didn't work well, try splitting by newline
         entries = raw_results.strip().split('\n')

    link_pattern = r'http[s]?://(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?' # More specific URL pattern

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        match = re.search(rf'(.*):\s*({link_pattern})', entry) # Look for "Title: URL" pattern
        
        if match:
            title = match.group(1).strip()
            link = match.group(2).strip()
            
            # Basic cleaning of title (remove leading "[", trailing ",")
            title = title.lstrip('["').rstrip(',"]') 

            # Check if it's a video link (e.g., YouTube)
            if video_site_filter.lower() in link.lower() and len(results['videos']) < max_links_per_type:
                results['videos'].append({"title": title or "Video Link", "url": link})
            # Otherwise, add as a general web link
            elif len(results['web']) < max_links_per_type:
                 results['web'].append({"title": title or "Web Link", "url": link})

        else:
             # If "Title: URL" pattern not found, just try to find a raw URL
             urls = re.findall(link_pattern, entry)
             if urls:
                 link = urls[0]
                 # Use a snippet of the text as a fallback title
                 title = entry.replace(link, "").strip()[:50] + "..." if len(entry.replace(link, "").strip()) > 50 else entry.replace(link, "").strip() or f"Link {len(results['web']) + len(results['videos']) + 1}"

                 if video_site_filter.lower() in link.lower() and len(results['videos']) < max_links_per_type:
                    results['videos'].append({"title": title or "Video Link", "url": link})
                 elif len(results['web']) < max_links_per_type:
                     results['web'].append({"title": title or "Web Link", "url": link})


    return results

# --- Conversational Helpers ---
def get_time_based_greeting():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return random.choice(["Good morning!", "Morning!"])
    elif 12 <= current_hour < 17:
        return random.choice(["Good afternoon!", "Afternoon!"])
    elif 17 <= current_hour < 21:
        return random.choice(["Good evening!", "Evening!"])
    else:
        return random.choice(["Hello!", "Hi there!"])

def detect_greeting(text):
    text_lower = text.lower().strip()
    greetings = [
        "hello", "hi", "hey", "hiya", "howdy", "sup", "what's up", 
        "good morning", "morning", "good afternoon", "afternoon", 
        "good evening", "evening", "yo", "hola", "namaste" # Added more
    ]
    # Check if the text *starts with or primarily consists of* a greeting
    # Avoid matching greetings within a question like "Hello, what is X?" vs "Hello"
    if any(text_lower == g for g in greetings):
         return True
    # Also check if the text contains a greeting and is relatively short (e.g., just a greeting + maybe a question mark)
    if len(text_lower.split()) <= 3 and any(g in text_lower for g in greetings):
         return True
    return False

def generate_greeting_response():
    time_greeting = get_time_based_greeting()
    return f"{time_greeting} How may I assist you? 😊"

def detect_thanks(text):
    text_lower = text.lower().strip()
    thanks_phrases = [
        "thank you", "thanks", "thx", "ty", "cheers", "appreciate it"
    ]
    if (
        any(phrase in text_lower for phrase in thanks_phrases)
        and not "no thanks" in text_lower # Exclude "no thanks"
    ):
        return True
    return False

def generate_thanks_response():
    responses = [
        "You're very welcome! 😊",
        "Happy to help! 👍",
        "Glad I could assist! 🤖",
        "No problem at all!",
        "Anytime!",
    ]
    return random.choice(responses)

def detect_farewell(text):
    text_lower = text.lower().strip()
    farewell_words = [
        "bye", "goodbye", "see you", "farewell", "take care", "later", "cya", "good night", "gtg"
    ]
    if any(word in text_lower for word in farewell_words):
        return True
    return False

def generate_farewell_response():
     return random.choice(["Goodbye! 👋", "See you later!", "Take care!", "Bye for now!"])


def format_search_results_for_slack(search_results: Dict[str, List[Dict]]) -> str:
    """Formats parsed DuckDuckGo search results into a Slack-friendly string."""
    output = []
    if search_results.get("web"):
        output.append("*Relevant Web Links:*")
        for item in search_results["web"]:
            output.append(f"• <{item['url']}|{item['title']}>")

    if search_results.get("videos"):
        output.append("\n*Relevant Videos:*")
        for item in search_results["videos"]:
             # Add a YouTube icon if the link is from YouTube
             icon = ":youtube:" if "youtube.com" in item['url'].lower() else ""
             output.append(f"• {icon} <{item['url']}|{item['title']}>")

    return "\n".join(output)


# --- Slack event handling ---
def handle_message(event):
    """Processes incoming Slack message, determines response type, and sends response."""
    try:
        channel = event["channel"]
        text = event.get("text", "")
        thread_ts = event.get("ts") # Get thread timestamp

        # Ensure Slack Bot User ID is available
        if not SLACK_BOT_USER_ID:
             app.logger.error("SLACK_BOT_USER_ID environment variable not set. Cannot process mentions correctly.")
             try:
                  slack_client.chat_postMessage(
                       channel=channel,
                       text="Sorry, my configuration is incomplete (Bot User ID is missing). I can't respond right now.",
                       thread_ts=thread_ts,
                  )
             except Exception as post_error:
                  app.logger.error(f"Failed to post configuration error to Slack: {post_error}")
             return # Stop processing if critical config is missing


        # Remove bot mention from text if it exists (important for app_mention events)
        mention_string = f"<@{SLACK_BOT_USER_ID}>"
        if mention_string in text:
             text = text.replace(mention_string, "").strip()

        if not text:
            # Ignore empty messages after removing mention or if message was initially empty
            app.logger.debug("Ignoring empty message after processing.")
            return

        app.logger.info(f"Processing message from user {event.get('user')} in channel {channel}: '{text}'")

        # --- Handle Conversational Cues ---
        text_lower = text.lower().strip()
        if detect_greeting(text_lower):
            response_text = generate_greeting_response()
        elif detect_thanks(text_lower):
            response_text = generate_thanks_response()
        elif detect_farewell(text_lower):
            response_text = generate_farewell_response()
        # Add other simple conversational handlers here if needed
        elif text_lower in ["who are you", "what are your capabilities", "what can you do"]:
             response_text = (
                f"I am an AI assistant powered by OpenAI embeddings and generation. "
                f"My knowledge base is stored in MongoDB ({MONGODB_DB_NAME}) and indexed using FAISS. "
                "I can answer questions based on the documents I've been trained on, and I can also search the web for links and videos."
                "\n\nAsk me about topics covered in the knowledge base, or just ask a general question."
            )
        else:
             # --- Handle RAG and External Search ---
             # Ensure KB and RAG chain are available
             if not kb or not kb.mongodb_connected or not openai_available or not rag_chain:
                response_text = "Sorry, my core RAG system is not fully operational right now. Please check the server status."
                app.logger.error("KB, MongoDB, OpenAI, or RAG chain not available. Cannot perform RAG.")
             else:
                try:
                    # Perform RAG query using the chain
                    app.logger.info(f"Invoking RAG chain for query: '{text}'")
                    # The RetrievalQA chain handles retrieval and generation internally
                    rag_response_obj = rag_chain.invoke({"query": text})
                    rag_answer = rag_response_obj.get('result', "I couldn't find relevant information in my knowledge base to answer that.")

                    # Perform external search using the tool
                    external_search_links = ""
                    if search_tool:
                        try:
                            app.logger.info(f"Performing external search for: {text}")
                            # Perform two searches: one general, one targeted at videos
                            general_search_raw = search_tool.run(f"{text} links")
                            video_search_raw = search_tool.run(f"site:youtube.com {text}")

                            # Combine and parse results
                            combined_raw_results = general_search_raw + "\n" + video_search_raw
                            parsed_external_results = parse_duckduckgo_search_results(combined_raw_results, max_links_per_type=3)

                            external_search_links = format_search_results_for_slack(parsed_external_results)

                        except Exception as search_error:
                            app.logger.error(f"Error during external search: {search_error}")
                            external_search_links = "\n\n_Error fetching external links._"
                    else:
                         external_search_links = "\n\n_External search tool not available._"


                    # Combine RAG answer and external links
                    response_text = f"{rag_answer}"
                    if external_search_links.strip(): # Only add if there are links
                         response_text += f"\n\n---\n{external_search_links}"


                except Exception as rag_error:
                    app.logger.error(f"Error during RAG or external search process: {rag_error}", exc_info=True)
                    response_text = "Sorry, I encountered an error while trying to find or generate information. Please try again."


        # Send response to Slack
        if SLACK_BOT_TOKEN:
            try:
                 slack_client.chat_postMessage(
                     channel=channel,
                     text=response_text,
                     thread_ts=thread_ts, # Reply in thread if message was in a thread
                 )
                 app.logger.info(f"Sent response (first 100 chars): {response_text[:100]}...")
            except SlackApiError as e:
                 app.logger.error(f"Slack API Error sending message: {e.response['error']}")
            except Exception as post_error:
                 app.logger.error(f"Failed to send Slack message: {post_error}")

        else:
            app.logger.error("Cannot send Slack message: SLACK_BOT_TOKEN not configured.")


    except Exception as e:
        app.logger.error(f"Error handling message: {e}", exc_info=True)
        # Attempt to send a generic error message back to the user in Slack
        if SLACK_BOT_TOKEN:
            try:
                slack_client.chat_postMessage(
                    channel=event.get("channel"),
                    text="Oops! Something went wrong while processing your request. My apologies! 😵",
                    thread_ts=event.get("ts"),
                )
            except Exception as slack_err:
                app.logger.error(f"Failed to send error message to Slack: {slack_err}")


# --- Flask Routes ---
@app.route("/slack/events", methods=["POST"])
def slack_events_route():
    """Handle Slack events endpoint."""
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    # Handle URL verification challenge
    if data.get("type") == "url_verification":
        app.logger.info("Received URL verification challenge")
        return jsonify({"challenge": data["challenge"]})

    # Handle event callback
    if data.get("type") == "event_callback":
        event = data.get("event")
        if not event:
            app.logger.warning("Received event_callback with no event payload.")
            return jsonify({"status": "ok", "message": "No event payload"}), 200

        # Ignore specific event types or messages from bots
        if event.get("subtype") == "bot_message" or event.get("bot_id"):
            app.logger.debug("Ignoring bot message subtype or bot_id.")
            return jsonify({"status": "ok", "message": "Ignored bot message"})

        # Process app_mention or direct messages
        # Ensure SLACK_BOT_USER_ID is set for proper mention handling if type is app_mention
        if event.get("type") == "app_mention":
            if not SLACK_BOT_USER_ID:
                app.logger.error("Cannot process app_mention: SLACK_BOT_USER_ID not set.")
                return jsonify({"status": "error", "message": "Bot user ID not configured"}), 500 # Or return ok, but log
            handle_message(event)
            return jsonify({"status": "ok"}) # Respond quickly to Slack API

        if event.get("type") == "message" and event.get("channel_type") == "im":
             # This is a direct message to the bot
             handle_message(event)
             return jsonify({"status": "ok"}) # Respond quickly to Slack API
        
        # Ignore other message types (like channel messages not mentioning the bot)
        app.logger.debug(f"Ignoring message type '{event.get('type')}' or channel type '{event.get('channel_type')}'")
        return jsonify({"status": "ok", "message": "Ignored message type"})

    # Ignore other event types (e.g., team_join, reaction_added)
    app.logger.debug(f"Ignoring top-level event type: {data.get('type')}")
    return jsonify({"status": "ok", "message": "Ignored top-level event type"})


@app.route("/add_knowledge", methods=["POST"])
def add_knowledge_route():
    """Endpoint to add documents to knowledge base via JSON."""
    try:
        data = request.json
        if not data or not data.get("id") or not data.get("text"):
            return jsonify({"error": "Missing 'id' or 'text' in request body."}), 400

        doc_id = data["id"]
        text = data["text"]
        metadata = data.get("metadata", {})
        
        # Basic input validation
        if not isinstance(doc_id, str) or not doc_id.strip():
             return jsonify({"error": "Invalid 'id' provided."}), 400
        if not isinstance(text, str) or not text.strip():
             return jsonify({"error": "Invalid or empty 'text' provided."}), 400
        if not isinstance(metadata, dict):
            metadata = {} # Default to empty dict if invalid type

        app.logger.info(f"Received request to add document with ID: {doc_id}")

        # Ensure KB is ready before attempting to add
        if not kb or not kb.mongodb_connected:
            return jsonify({"error": "Knowledge base not ready: MongoDB not connected or KB init failed."}), 503
        if not openai_available:
            return jsonify({"error": "Knowledge base not ready: OpenAI API not available."}), 503


        kb.add_document(doc_id, text, metadata)
        app.logger.info(f"Successfully added/updated document '{doc_id}'.")
        
        # Re-initialize RAG chain if KB or vectorstore was just created/rebuilt
        global rag_chain
        if llm and kb and kb.vectorstore:
             if not rag_chain: # Only create if it was None before
                  try:
                      retriever = kb.get_retriever(search_kwargs={"k": 5})
                      if retriever:
                          RAG_PROMPT = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"])
                          rag_chain = RetrievalQA.from_chain_type(
                              llm=llm,
                              chain_type="stuff",
                              retriever=retriever,
                              chain_type_kwargs={"prompt": RAG_PROMPT},
                              return_source_documents=False,
                          )
                          app.logger.info("✅ RAG chain re-initialized after adding document.")
                      else:
                           app.logger.error("Failed to get retriever after adding document. RAG chain not re-initialized.")
                  except Exception as chain_init_e:
                      app.logger.error(f"Error re-initializing RAG chain after adding document: {chain_init_e}")


        return jsonify(
            {
                "status": "success",
                "message": f"Document '{doc_id}' added/updated and FAISS index rebuilt.",
            }
        )
    except Exception as e:
        app.logger.error(f"Error adding knowledge: {e}", exc_info=True)
        # Return more specific error if it's a known issue like MongoDB connection or API key
        if isinstance(e, ConnectionFailure):
             return jsonify({"error": f"Database connection error: {str(e)}"}), 503
        if "OpenAI API not available" in str(e):
             return jsonify({"error": f"OpenAI API not available: {str(e)}"}), 503
        return jsonify({"error": str(e)}), 500


@app.route("/upload_pdf", methods=["POST"])
def upload_pdf_route():
    """Endpoint to upload a PDF file, extract text, and add to knowledge base."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file."}), 400
        
        # Check file extension
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Invalid file type. Only PDF files are supported."}), 400

        # Get optional doc_id and metadata from form data
        doc_id = request.form.get('doc_id')
        metadata_str = request.form.get('metadata', '{}')
        
        # Generate a unique ID if not provided
        if not doc_id or not doc_id.strip():
            doc_id = f"pdf_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}_{os.path.basename(file.filename).replace('.', '_')}"[:64] # Ensure ID is not too long

        try:
            metadata = json.loads(metadata_str)
            if not isinstance(metadata, dict):
                metadata = {}
                app.logger.warning(f"Invalid metadata JSON format provided for PDF upload: {metadata_str}")
        except json.JSONDecodeError:
            metadata = {} # Default to empty if invalid JSON
            app.logger.warning(f"Invalid metadata JSON provided for PDF upload: {metadata_str}")

        app.logger.info(f"Received PDF upload request for file '{file.filename}', assigning doc_id: {doc_id}")

        # Ensure KB is ready before attempting to add
        if not kb or not kb.mongodb_connected:
            return jsonify({"error": "Knowledge base not ready: MongoDB not connected or KB init failed."}), 503
        if not openai_available:
            return jsonify({"error": "Knowledge base not ready: OpenAI API not available."}), 503

        # Read PDF content and extract text
        pdf_content = file.read()
        text = extract_text_from_pdf(io.BytesIO(pdf_content))
        
        if not text.strip():
            return jsonify({"error": "No text extracted from PDF. PDF might be image-based, empty, or encrypted with a password."}), 400
        
        app.logger.info(f"Extracted {len(text)} characters from PDF.")

        # Add the extracted text as a document
        # Add original filename to metadata
        metadata['original_filename'] = file.filename
        kb.add_document(doc_id, text, metadata)
        
        app.logger.info(f"Successfully processed PDF '{file.filename}' and added as document '{doc_id}'.")

        # Re-initialize RAG chain if KB or vectorstore was just created/rebuilt
        global rag_chain
        if llm and kb and kb.vectorstore:
             if not rag_chain: # Only create if it was None before
                  try:
                      retriever = kb.get_retriever(search_kwargs={"k": 5})
                      if retriever:
                          RAG_PROMPT = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"])
                          rag_chain = RetrievalQA.from_chain_type(
                              llm=llm,
                              chain_type="stuff",
                              retriever=retriever,
                              chain_type_kwargs={"prompt": RAG_PROMPT},
                              return_source_documents=False,
                          )
                          app.logger.info("✅ RAG chain re-initialized after adding document.")
                      else:
                           app.logger.error("Failed to get retriever after adding document. RAG chain not re-initialized.")
                  except Exception as chain_init_e:
                      app.logger.error(f"Error re-initializing RAG chain after adding document: {chain_init_e}")

        
        return jsonify({
            "status": "success",
            "message": f"PDF '{file.filename}' processed and added as document '{doc_id}'.",
            "doc_id": doc_id,
            "extracted_text_preview": text[:200] + "..." if len(text) > 200 else text # Show a preview
        })

    except PyPDF2.errors.PdfReadError as e:
        app.logger.error(f"PDF Read Error during upload: {e}")
        return jsonify({"error": f"Failed to read PDF: {e}. It might be corrupted, encrypted, or an invalid format."}), 400
    except ValueError as e: # Catch specific ValueErrors raised (like encrypted PDF)
         app.logger.error(f"ValueError during PDF upload: {e}")
         return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Error uploading PDF: {e}", exc_info=True)
        # Return more specific error if it's a known issue like MongoDB connection or API key
        if isinstance(e, ConnectionFailure):
             return jsonify({"error": f"Database connection error: {str(e)}"}), 503
        if "OpenAI API not available" in str(e):
             return jsonify({"error": f"OpenAI API not available: {str(e)}"}), 503
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/delete_knowledge/<doc_id>", methods=["DELETE"])
def delete_knowledge_route(doc_id):
    """Endpoint to delete a document by its ID."""
    app.logger.info(f"Received request to delete document with ID: {doc_id}")
    try:
        if not kb or not kb.mongodb_connected:
            return jsonify({"error": "Knowledge base not ready: MongoDB not connected or KB init failed."}), 503
        if not openai_available: # Rebuilding FAISS requires embeddings
             return jsonify({"error": "Knowledge base cannot rebuild after deletion: OpenAI API not available."}), 503


        success = kb.delete_document(doc_id)
        if success:
            app.logger.info(f"Successfully deleted document: {doc_id}")
            # Re-initialize RAG chain after deleting document
            global rag_chain
            if llm and kb and kb.vectorstore:
                try:
                    retriever = kb.get_retriever(search_kwargs={"k": 5})
                    if retriever:
                        RAG_PROMPT = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"])
                        rag_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            chain_type_kwargs={"prompt": RAG_PROMPT},
                            return_source_documents=False,
                        )
                        app.logger.info("✅ RAG chain re-initialized after deleting document.")
                    else:
                        # If retriever is none (e.g., no docs left), the chain might become unusable
                        rag_chain = None 
                        app.logger.warning("Retriever returned None after deletion (maybe no docs left). RAG chain set to None.")

                except Exception as chain_init_e:
                    app.logger.error(f"Error re-initializing RAG chain after deleting document: {chain_init_e}")
                    rag_chain = None # Set to None on error

            return jsonify({"status": "success", "message": f"Document '{doc_id}' deleted."})
        else:
            app.logger.warning(f"Document {doc_id} not found for deletion.")
            return jsonify({"error": "Document not found or deletion failed."}), 404
    except Exception as e:
        app.logger.error(f"Error deleting knowledge: {e}", exc_info=True)
        if isinstance(e, ConnectionFailure):
             return jsonify({"error": f"Database connection error: {str(e)}"}), 503
        if "OpenAI API not available" in str(e):
             return jsonify({"error": f"OpenAI API not available: {str(e)}"}), 503
        return jsonify({"error": str(e)}), 500


@app.route("/db_stats", methods=["GET"])
def database_stats_route():
    """Endpoint to get database statistics."""
    app.logger.info("Received request for database stats.")
    try:
        if not kb:
             return jsonify({"status": "error", "message": "Knowledge base not initialized."}), 503

        stats = kb.get_database_stats()
        if stats.get("error") and stats.get("mongodb_connected") is False:
             return jsonify({"status": "error", "message": stats["error"]}), 503 # Service Unavailable if MongoDB is down
        elif stats.get("error"): # Other errors
            return jsonify({"status": "error", "message": stats["error"]}), 500
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error getting database stats via route: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/test_search", methods=["POST"])
def test_search_route():
    """Endpoint to test semantic search using the FAISS index."""
    try:
        data = request.json
        if not data or not data.get("query"):
            return jsonify({"error": "Missing query"}), 400
        
        query = data["query"]
        n_results = data.get("n_results", 3)
        # Remove search_type option as this is strictly semantic via FAISS now

        app.logger.info(f"Received test semantic search request: query='{query[:50]}...', n={n_results}")

        # Ensure KB and vectorstore are ready
        if not kb or not kb.vectorstore or not kb.mongodb_connected or not openai_available:
            return jsonify({"error": "Semantic search not ready: KB, Vectorstore, MongoDB, or OpenAI unavailable."}), 503


        # Perform the search using the KB instance's semantic_search method
        results = kb.search(query, n_results) # kb.search defaults to semantic now

        # Format results for JSON output
        formatted_results = []
        if results and results.get("documents"):
            for i in range(len(results["documents"])):
                formatted_results.append(
                    {
                        "chunk_text": results["documents"][i],
                        "score": (results["scores"][i] if "scores" in results and i < len(results["scores"]) else None), # Cosine distance from FAISS
                        "chunk_id": (results["ids"][i] if "ids" in results and i < len(results["ids"]) else None),
                        "metadata": (results["metadatas"][i] if "metadatas" in results and i < len(results["metadatas"]) else {}), # This is parent doc metadata + chunk info
                    }
                )
        app.logger.info(f"Test search returned {len(formatted_results)} results.")
        return jsonify(
            {"query": query, "search_type": "semantic", "n_results_requested": n_results, "results": formatted_results}
        )
    except Exception as e:
        app.logger.error(f"Error in test_search: {e}", exc_info=True)
        if isinstance(e, ConnectionFailure):
             return jsonify({"error": f"Database connection error: {str(e)}"}), 503
        if "OpenAI API not available" in str(e) or not openai_available:
             return jsonify({"error": f"OpenAI API not available: {str(e)}"}), 503
        return jsonify({"error": str(e)}), 500

@app.route("/test_rag_gen", methods=["POST"])
def test_rag_gen_route():
    """Endpoint to test RAG generation with provided context and question using the RAG chain."""
    try:
        data = request.json
        if not data or not data.get("question") or not data.get("context"):
            # User can provide context directly for testing the LLM part
            # Or they can rely on the chain's retrieval if context is missing
            # Let's make context optional for this test route, if missing, it performs retrieval first
            question = data.get("question")
            provided_context = data.get("context")

            if not question:
                 return jsonify({"error": "Missing 'question' in request body."}), 400

        app.logger.info(f"Received test RAG gen request: question='{question[:50]}...'")

        # Ensure RAG chain is available
        if not rag_chain:
             return jsonify({"error": "RAG Generation chain not initialized. Check OpenAI API, MongoDB, and FAISS index status."}), 503

        try:
            # If context is provided, use it directly in the chain's LLM call
            # If not provided, the chain will perform retrieval based on the question
            if provided_context:
                 # This mimics the stuffing process, but with user-provided context
                 # We can't directly inject context into the RetrievalQA chain's internal LLM call easily
                 # A better approach is to bypass the retriever and call the LLM with a custom prompt
                 # Or, just require context for this test route
                 
                 # Let's refine the route: require question and context to test the LLM generation based on *given* context.
                 # To test retrieval + generation, use the handle_message flow or a dedicated endpoint.
                 
                 # Use the base LLM with a RAG-like prompt
                 if not llm:
                      return jsonify({"error": "OpenAI LLM not initialized."}), 503

                 rag_prompt_template_llm = """You are a helpful AI assistant. Answer the following question based *only* on the provided context. 
If the context doesn't contain enough information to answer the question, say so politely and state that you couldn't find the information in the context. Do not make up information.

Context:
{context}

Question: {question}

Answer:"""
                 prompt = PromptTemplate(template=rag_prompt_template_llm, input_variables=["context", "question"]).format(context=provided_context, question=question)

                 # Use LLM directly
                 response = llm.invoke(prompt)
                 generated_answer = response.content.strip() # Use .content for BaseMessage

                 app.logger.info("Test RAG generation (with provided context) completed.")
                 return jsonify({
                    "question": question,
                    "context_preview": provided_context[:200] + "..." if len(provided_context) > 200 else provided_context,
                    "generated_answer": generated_answer
                 })

            else:
                 # If no context provided, use the RAG chain to test retrieval + generation end-to-end
                 app.logger.info(f"Performing retrieval + generation test for query: '{question}'")
                 rag_response_obj = rag_chain.invoke({"query": question})
                 generated_answer = rag_response_obj.get('result', "RAG chain could not generate an answer.")

                 # Optionally, get source documents if return_source_documents=True in chain config
                 # source_docs = rag_response_obj.get('source_documents', [])
                 # source_previews = [{"id": doc.metadata.get("chunk_id"), "text_preview": doc.page_content[:100] + "..."} for doc in source_docs]

                 app.logger.info("Test retrieval + generation completed.")
                 return jsonify({
                     "query": question,
                     "generated_answer": generated_answer,
                     # "source_documents": source_previews # Add if return_source_documents is True
                 })


        except Exception as e:
            app.logger.error(f"Error during RAG generation test: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        app.logger.error(f"Error in test_rag_gen route setup: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/list_knowledge", methods=["GET"])
def list_knowledge_route():
    """Endpoint to list all documents in the knowledge base."""
    app.logger.info("Received request to list knowledge.")
    try:
        if not kb or not kb.mongodb_connected:
             return jsonify({"error": "Knowledge base not ready: MongoDB not connected or KB init failed."}), 503
            
        docs_info = kb.get_all_documents()
        
        # Get FAISS stats for context
        faiss_size = kb.vectorstore.index.ntotal if kb.vectorstore and kb.vectorstore.index else 0 # Use index.ntotal for FAISS size

        return jsonify(
            {
                "total_documents": docs_info["total_documents"],
                "total_chunks_in_faiss_index": faiss_size,
                "embedding_model": getattr(embeddings, 'model_name', 'Unknown'), # Get model name from embeddings object
                "vector_dimension": getattr(embeddings, 'embedding_ctx_length', 'Unknown'), # Get dimension from embeddings object (might be different for different models)
                "documents": docs_info["documents"] # This already contains id, preview, metadata, chunk_count
            }
        )
    except Exception as e:
        app.logger.error(f"Error in list_knowledge: {e}", exc_info=True)
        if isinstance(e, ConnectionFailure):
             return jsonify({"error": f"Database connection error: {str(e)}"}), 503
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check_route():
    """Health check endpoint."""
    app.logger.debug("Received health check request.")
    
    # Check core component statuses directly
    is_mongodb_connected = kb.mongodb_connected if kb else False
    is_openai_available = openai_available
    is_rag_chain_initialized = rag_chain is not None
    is_search_tool_available = search_tool is not None

    db_stats = {}
    if is_mongodb_connected:
        try:
            # Get stats without logging errors unless they are severe
            db_stats_raw = kb.get_database_stats()
            if db_stats_raw.get("error"):
                app.logger.warning(f"Health check DB stats fetch warning: {db_stats_raw['error']}")
            db_stats = db_stats_raw
            db_stats.pop("error", None) # Remove error field if present
        except Exception as e:
            app.logger.warning(f"Failed to get DB stats during health check: {e}")
            db_stats = {"error": str(e)} # Indicate error in stats if it occurs during fetch


    # Overall status: healthy only if critical components are confirmed working
    overall_status = "healthy" 
    if not is_mongodb_connected or not is_openai_available or not is_rag_chain_initialized:
        overall_status = "unhealthy"
    # External search and Slack User ID are less critical, might make status 'warning'
    # if not is_search_tool_available or not SLACK_BOT_USER_ID:
    #     overall_status = "warning"


    return jsonify({
        "status": overall_status,
        "components": {
            "mongodb": {"status": "ok" if is_mongodb_connected else "error", "details": db_stats},
            "openai_api": {"status": "ok" if is_openai_available else "error"},
            "openai_embedding_model": {"status": "ok" if embeddings else "error", "model": getattr(embeddings, 'model_name', 'Unknown'), "dimension": getattr(embeddings, 'embedding_ctx_length', 'Unknown')},
            "openai_chat_model": {"status": "ok" if llm else "error", "model": getattr(llm, 'model_name', 'Unknown')},
            "faiss_index": {"status": "ok" if kb and kb.vectorstore else "error", "size": kb.vectorstore.index.ntotal if kb and kb.vectorstore and kb.vectorstore.index else 0, "description": "In-memory vector index"},
            "rag_chain": {"status": "ok" if is_rag_chain_initialized else "error"},
            "external_search_tool": {"status": "ok" if is_search_tool_available else "warning"},
            "slack_token": {"status": "ok" if SLACK_BOT_TOKEN else "error"},
            "slack_bot_user_id": {"status": "ok" if SLACK_BOT_USER_ID else "warning", "details": "Needed for mentions"}
        }
    })


@app.route("/")
def home_route():
    """Simple homepage route displaying status."""
    # Use status attributes checked in health_check
    is_mongodb_connected = kb.mongodb_connected if kb else False
    is_openai_available = openai_available
    is_rag_chain_initialized = rag_chain is not None
    is_search_tool_available = search_tool is not None
    
    db_stats = {}
    if is_mongodb_connected and kb:
        try:
             db_stats = kb.get_database_stats()
        except Exception as e:
             db_stats["error"] = str(e) # Don't log aggressively for home page

    total_docs = db_stats.get("total_documents_in_db", "N/A")
    total_chunks_db = db_stats.get("total_chunks_in_db", "N/A")
    faiss_size = db_stats.get("faiss_index_size", "N/A")

    openai_status = '✅ Configured' if is_openai_available else '❌ NOT CONFIGURED OR UNAVAILABLE!'
    rag_chain_status = '✅ Initialized' if is_rag_chain_initialized else '❌ Not Initialized (Check dependencies!)'
    mongodb_status = '✅ Connected' if is_mongodb_connected else '❌ NOT CONNECTED!'
    slack_status = '✅ Configured' if SLACK_BOT_TOKEN else '❌ NOT CONFIGURED!'
    slack_user_status = '✅ Configured' if SLACK_BOT_USER_ID else '⚠️ Missing (Mentions might fail)!'
    search_tool_status = '✅ Available' if is_search_tool_available else '⚠️ Unavailable'

    emb_model_name = getattr(embeddings, 'model_name', 'Unknown') if embeddings else 'Unknown'
    emb_dim = getattr(embeddings, 'embedding_ctx_length', 'Unknown') if embeddings else 'Unknown' # Or hardcode if specific model
    llm_model_name = getattr(llm, 'model_name', 'Unknown') if llm else 'Unknown'

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Slack AI Bot Status (OpenAI + FAISS)</title>
        <style>
            body {{ font-family: sans-serif; line-height: 1.6; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .status {{ font-weight: bold; }}
            .status.ok {{ color: green; }}
            .status.warning {{ color: orange; }}
            .status.error {{ color: red; }}
            code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 4px; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>🚀 RAG Slack AI Bot Status</h1>
        <p>Overall Status: <span class="status {'ok' if is_mongodb_connected and is_openai_available and is_rag_chain_initialized else 'error'}">{'✅ Running' if is_mongodb_connected and is_openai_available and is_rag_chain_initialized else '❌ CRITICAL CONFIGURATION ISSUE!'}</span></p>
        
        <h2>🔧 RAG Configuration:</h2>
        <ul>
            <li><strong>LLM (Generation):</strong> OpenAI (<code id="llm_model">{llm_model_name}</code>) <span class="status {'ok' if llm else 'error'}">{openai_status}</span></li>
            <li><strong>Embedding Service:</strong> OpenAI (<code id="emb_model">{emb_model_name}</code>, Dim: {emb_dim}) <span class="status {'ok' if embeddings else 'error'}">{openai_status}</span></li>
            <li><strong>Vector DB:</strong> FAISS (In-memory Index)</li>
            <li><strong>Persistence:</strong> MongoDB (<code id="mongo_uri">{MONGODB_URI}</code>, DB: <code id="mongo_db">{MONGODB_DB_NAME}</code>) <span class="status {'ok' if is_mongodb_connected else 'error'}">{mongodb_status}</span></li>
            <li><strong>RAG Chain:</strong> LangChain RetrievalQA <span class="status {'ok' if is_rag_chain_initialized else 'error'}">{rag_chain_status}</span></li>
             <li><strong>External Search:</strong> DuckDuckGo Tool <span class="status {'ok' if is_search_tool_available else 'warning'}">{search_tool_status}</span></li>
        </ul>

        <h2>📊 Knowledge Base Stats:</h2>
        <ul>
            <li><strong>Total Docs in DB:</strong> {total_docs}</li>
            <li><strong>Total Chunks in DB:</strong> {total_chunks_db}</li>
            <li><strong>Total Chunks in FAISS Index:</strong> {faiss_size}</li>
        </ul>

        <h2>💬 Slack Integration:</h2>
        <ul>
            <li><strong>Slack Bot Token:</strong> <span class="status {'ok' if SLACK_BOT_TOKEN else 'error'}">{slack_status}</span></li>
            <li><strong>Slack Bot User ID:</strong> <span class="status {'ok' if SLACK_BOT_USER_ID else 'warning'}">{slack_user_status}</span></li>
        </ul>

        <h2>🔗 API Endpoints:</h2>
        <ul>
            <li><code>POST /slack/events</code>: For Slack event subscriptions.</li>
            <li><code>POST /add_knowledge</code>: Add text document. Body: <code>{{ "id": "doc1", "text": "...", "metadata": {{}} }}</code></li>
            <li><code>POST /upload_pdf</code>: Upload PDF file. Use form-data with 'file' field. Optional 'doc_id', 'metadata'.</li>
            <li><code>DELETE /delete_knowledge/<doc_id></code>: Delete document and chunks.</li>
            <li><code>GET /list_knowledge</code>: List documents summary.</li>
            <li><code>GET /db_stats</code>: Get database statistics.</li>
            <li><code>POST /test_search</code>: Test *semantic* search (FAISS). Body: <code>{{ "query": "...", "n_results": 3 }}</code></li>
             <li><code>POST /test_rag_gen</code>: Test RAG generation (LLM + optional provided context). Body: <code>{{ "question": "...", "context": "..." }}</code> (context optional)</li>
            <li><code>GET /health</code>: Basic health check.</li>
        </ul>

        <p><em>Remember to set <code>OPENAI_API_KEY</code>, <code>SLACK_BOT_TOKEN</code>, <code>SLACK_BOT_USER_ID</code>, and <code>MONGODB_URI</code> in your <code>.env</code> file.</em></p>
    </body>
    </html>
    """


# --- Main Execution Block ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    app.logger.info("🚀 Starting RAG-Powered Slack AI Bot with OpenAI + FAISS")

    # Check critical configurations
    is_mongodb_connected_startup = kb.mongodb_connected if kb else False
    is_openai_available_startup = openai_available
    is_rag_chain_initialized_startup = rag_chain is not None
    slack_token_ok_startup = bool(SLACK_BOT_TOKEN)
    slack_user_id_ok_startup = bool(SLACK_BOT_USER_ID) # Check for user ID too


    if not is_openai_available_startup:
        app.logger.critical(
            "FATAL: OpenAI API is not available. Embedding and generation will fail. Please check OPENAI_API_KEY."
        )
    if not slack_token_ok_startup:
        app.logger.critical(
            "FATAL: SLACK_BOT_TOKEN is not configured. Bot cannot connect to Slack. Please set SLACK_BOT_TOKEN."
        )
    if not slack_user_id_ok_startup:
        app.logger.warning(
             "WARNING: SLACK_BOT_USER_ID is not configured. Bot may not respond correctly to channel mentions. Please set SLACK_BOT_USER_ID."
        )
    if not is_mongodb_connected_startup:
        app.logger.critical(
            "FATAL: MongoDB is not connected. Knowledge base persistence will be unavailable. Please check MONGODB_URI."
        )
    if not is_rag_chain_initialized_startup and is_mongodb_connected_startup and is_openai_available_startup and kb and kb.vectorstore:
         # Log warning if rag_chain *should* have initialized but didn't,
         # assuming dependencies were met
         app.logger.warning(
             "WARNING: RAG Chain failed to initialize despite dependencies (MongoDB, OpenAI API, KB, Vectorstore) seeming okay. Check chain creation logic/errors above."
         )


    app.logger.info("\n" + "=" * 80)
    app.logger.info("🎯 OpenAI + FAISS RAG Bot Startup Summary:")

    # Access status flags and model names safely
    app.logger.info(
        f"   • OpenAI API: {'✅' if is_openai_available_startup else '❌ (KEY/CONNECT ISSUE!)'}"
    )
    app.logger.info(
        f"   • LLM (Generation): {'OpenAI (' + getattr(llm, 'model_name', 'Unknown') + ')' if llm else 'Unavailable'} {'✅' if llm else '❌'}"
    )
    app.logger.info(
        f"   • Embeddings: {'OpenAI (' + getattr(embeddings, 'model_name', 'Unknown') + ', Dim: ' + str(getattr(embeddings, 'embedding_ctx_length', 'Unknown')) + ')' if embeddings else 'Unavailable'} {'✅' if embeddings else '❌'}"
    )
    app.logger.info(
        f"   • RAG Chain (RetrievalQA): {'✅ Initialized' if is_rag_chain_initialized_startup else '❌ Not Initialized'}"
    )
    app.logger.info(
        f"   • External Search Tool: {'✅ Available' if search_tool else '⚠️ Unavailable'}"
    )
    app.logger.info(
        f"   • Slack Integration: {'✅ Configured' if slack_token_ok_startup else '❌ (TOKEN ISSUE!)'} (User ID {'✅' if slack_user_id_ok_startup else '⚠️ MISSING!'})"
    )
    app.logger.info(
        f"   • MongoDB Connection: {'✅ Connected' if is_mongodb_connected_startup else '❌ (CONNECTION ISSUE!)'}"
    )
    app.logger.info(
        f"   • FAISS Index Size: {kb.vectorstore.index.ntotal if kb and kb.vectorstore and kb.vectorstore.index else 0} chunks"
    )
    
    port = int(os.environ.get("PORT", 3000))
    app.logger.info(f"🌐 Web interface: http://localhost:{port}")
    app.logger.info("=" * 80 + "\n")

    if not is_mongodb_connected_startup or not is_openai_available_startup or not slack_token_ok_startup or not is_rag_chain_initialized_startup:
         app.logger.warning("CRITICAL CONFIGURATION ISSUES DETECTED. BOT MAY NOT FUNCTION CORRECTLY.")

    # Running with debug=True and use_reloader=False is good for development
    # In production, disable debug and use a production-ready WSGI server (like Gunicorn)
    # and consider running handle_message asynchronously.
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)