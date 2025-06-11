import os
import json
from flask import Flask, request, jsonify, render_template_string
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging
from dotenv import load_dotenv
import re
import random
from datetime import datetime
import numpy as np # Still useful if interacting with raw embeddings
import faiss # LangChain uses this, but we might interact less directly
import pickle # For FAISS persistence
from typing import List, Dict, Optional, Any
import openai # Keep if needed elsewhere, but RAG will use Gemini via LangChain
import google.generativeai as genai
from pymongo.errors import PyMongoError
# GridFS removed as we store text, not files, in MongoDB now
# from bson import ObjectId # Keep if ObjectId is used for MongoDB docs, but doc_id str is fine
from pymongo import MongoClient
import PyPDF2
import io

# --- LangChain Imports ---
from langchain.docstore.document import Document # Represents a piece of text and metadata
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # Gemini integration
from langchain.text_splitter import RecursiveCharacterTextSplitter # Split text into chunks
from langchain.vectorstores import FAISS # FAISS vector store wrapper
from langchain.chains import RetrievalQA # Standard RAG chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# --- Configuration from Environment Variables ---
# Use .get for safer access with defaults
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "rag_knowledge_base_lc") # Use a different DB name
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "./faiss_index") # Path for FAISS persistence

# --- Critical Dependency Checks ---
if not GEMINI_API_KEY:
    app.logger.critical(
        "FATAL: GOOGLE_API_KEY is not set in environment variables. Embedding and generation will fail."
    )
    # Optionally, exit or raise an exception here if this is absolutely required for startup
    # import sys; sys.exit("GOOGLE_API_KEY not set.")

if not SLACK_BOT_TOKEN:
    app.logger.critical(
        "FATAL: SLACK_BOT_TOKEN not found in environment variables. Slack integration will fail."
    )
    # Optionally, exit or raise an exception here
    # import sys; sys.exit("SLACK_BOT_TOKEN not set.")

# --- Initialize Google Generative AI (for raw calls or direct usage) ---
# This is separate from LangChain's wrapper but good for initial checks
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        app.logger.info("Gemini API Key configured via genai.")
        # Optional: Test a simple call to verify key
        # genai.list_models()
    except Exception as e:
        app.logger.critical(f"Failed to configure Gemini API via genai: {e}. Please check your key.")
else:
     app.logger.warning("Gemini API Key not configured. Direct genai calls may fail.")

# --- Initialize OpenAI (optional - if you still want it for non-RAG things) ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    app.logger.info("OpenAI API Key configured (optional).")
else:
    app.logger.info("OpenAI API Key not found. OpenAI dependent features will be unavailable.")


# --- LangChain RAG Setup ---
class LangchainRAGKnowledgeBase:
    def __init__(self):
        self.mongodb_connected = False
        self.mongo_client = None
        self.db = None
        self.documents_collection = None

        # MongoDB Setup (only for storing original docs and metadata)
        try:
            self.mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            self.db = self.mongo_client[MONGODB_DB_NAME]
            self.documents_collection = self.db.original_documents # New collection name
            
            # Create index for doc_id
            self.documents_collection.create_index("doc_id", unique=True)
            
            self.mongo_client.admin.command('ping')
            app.logger.info(f"✅ Connected to MongoDB: {MONGODB_URI} (DB: {MONGODB_DB_NAME})")
            self.mongodb_connected = True
        except Exception as e:
            app.logger.error(f"❌ Failed to connect to MongoDB at {MONGODB_URI}: {e}")
            self.mongodb_connected = False

        # LangChain Component Setup
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )

        # Ensure Gemini API key is available for LangChain embeddings and LLM
        if not GEMINI_API_KEY:
             self.embeddings = None
             self.llm = None
             self.retriever = None
             self.qa_chain = None
             self.langchain_ready = False
             app.logger.critical("LangChain RAG not initialized: GOOGLE_API_KEY is missing.")
             return

        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", # Use the recommended embedding model
                task_type="RETRIEVAL_DOCUMENT" # Default task type for embeddings
            )
            self.embedding_dim = 768 # As per models/text-embedding-004 documentation

            self.llm = ChatGoogleGenerativeAI(model="gemini-pro") # Use gemini-pro for chat/QA

            # --- FAISS Vector Store Setup ---
            # Try loading existing index
            try:
                self.vectorstore = FAISS.load_local(
                    folder_path=FAISS_INDEX_PATH,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True # Required for loading arbitrary pickle/json
                )
                app.logger.info(f"✅ Loaded existing FAISS index from {FAISS_INDEX_PATH}")
            except Exception as e:
                app.logger.warning(f"FAISS index not found or failed to load from {FAISS_INDEX_PATH}: {e}. Creating a new one.")
                # If loading fails, create a new empty index
                # Need an initial dummy document to create the index structure
                # This is a common FAISS-LangChain quirk for empty indexes
                # A better way might be to create an empty FAISS index manually if load fails
                # Let's create an index only when the first document is added
                self.vectorstore = None # Will be initialized on first add_document


            # --- RetrievalQA Chain Setup ---
            # Define the prompt template
            template = """
            You are an AI assistant designed to answer questions based *only* on the following context.
            If the answer is not found in the context, please state that you cannot find the answer in the provided information.
            Do not make up information.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

            if self.vectorstore:
                 self.retriever = self.vectorstore.as_retriever()
                 self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff", # Uses all context in one go
                    retriever=self.retriever,
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                    return_source_documents=True # To get retrieved docs back
                )
            else:
                self.retriever = None
                self.qa_chain = None
                app.logger.warning("FAISS index not loaded or created yet. RAG chain will be initialized on first document add.")


            self.langchain_ready = True
            app.logger.info("✅ LangChain RAG components initialized.")

        except Exception as e:
            app.logger.critical(f"❌ Failed to initialize LangChain RAG components: {e}", exc_info=True)
            self.embeddings = None
            self.llm = None
            self.retriever = None
            self.qa_chain = None
            self.langchain_ready = False


    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        if not self.mongodb_connected:
            app.logger.error(f"Cannot add document {doc_id}: MongoDB is not connected.")
            raise ConnectionError("MongoDB is not connected.") # Indicate failure
            
        if not self.langchain_ready:
            app.logger.error(f"Cannot add document {doc_id}: LangChain RAG components not ready (likely Gemini API issue).")
            raise RuntimeError("LangChain RAG components not ready.") # Indicate failure

        try:
            # 1. Store original document in MongoDB (Upsert)
            doc_data = {
                "doc_id": doc_id,
                "text": text,
                "metadata": metadata or {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            # Delete existing doc and its chunks first if updating
            self.delete_document(doc_id, skip_mongo_delete=True) # Delete FAISS chunks, keep mongo for now

            self.documents_collection.replace_one(
                {"doc_id": doc_id},
                doc_data,
                upsert=True
            )
            app.logger.info(f"Added/Updated original document {doc_id} in MongoDB.")

            # 2. Process for LangChain/FAISS
            lc_document = Document(page_content=text, metadata={**(metadata or {}), "doc_id": doc_id})

            # Split into chunks
            chunks = self.text_splitter.split_documents([lc_document])
            app.logger.info(f"Split document {doc_id} into {len(chunks)} chunks.")

            if not chunks:
                app.logger.warning(f"No chunks generated for document {doc_id}. Document stored, but not searchable.")
                return

            # 3. Add chunks to FAISS
            if self.vectorstore is None:
                 # Initialize FAISS index if it doesn't exist
                 app.logger.info(f"Initializing new FAISS index with first {len(chunks)} chunks.")
                 self.vectorstore = FAISS.from_documents(
                    chunks,
                    self.embeddings
                 )
                 self.retriever = self.vectorstore.as_retriever()
                 # Re-initialize the QA chain with the new vectorstore
                 self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.retriever,
                    chain_type_kwargs={"prompt": PromptTemplate.from_template("""
                        You are an AI assistant designed to answer questions based *only* on the following context.
                        If the answer is not found in the context, please state that you cannot find the answer in the provided information.
                        Do not make up information.

                        Context:
                        {context}

                        Question:
                        {question}

                        Answer:
                        """)},
                    return_source_documents=True
                )
            else:
                # Add documents to existing index
                self.vectorstore.add_documents(chunks)
                app.logger.info(f"Added {len(chunks)} chunks to existing FAISS index for document {doc_id}.")

            # 4. Save FAISS index
            self.vectorstore.save_local(FAISS_INDEX_PATH)
            app.logger.info(f"Saved FAISS index to {FAISS_INDEX_PATH}.")

            app.logger.info(f"Successfully processed and indexed document: {doc_id}")

        except PyMongoError as e:
            app.logger.error(f"MongoDB error adding document {doc_id}: {e}")
            raise # Re-raise to indicate failure
        except Exception as e:
            app.logger.error(f"Error adding document {doc_id} to LangChain/FAISS: {e}", exc_info=True)
            raise # Re-raise

    def search_and_answer(self, query: str) -> Dict[str, Any]:
        """Uses the LangChain QA chain to find relevant docs and generate an answer."""
        if not self.langchain_ready or self.qa_chain is None:
             app.logger.error("LangChain RAG chain is not ready.")
             return {
                 "answer": "Sorry, my knowledge base is not ready to answer questions right now.",
                 "source_documents": [],
                 "query": query
             }
        
        try:
            # The qa_chain performs both retrieval (from vectorstore) and generation (with LLM)
            result = self.qa_chain({"query": query})
            
            # result will contain 'answer' and 'source_documents' if return_source_documents=True
            app.logger.info(f"LangChain QA chain executed for query: '{query[:50]}...'")

            # Extract relevant info from source_documents
            # Source documents from FAISS will have 'metadata' which includes our 'doc_id'
            retrieved_docs_info = []
            for doc in result.get("source_documents", []):
                 retrieved_docs_info.append({
                      "page_content": doc.page_content,
                      "metadata": doc.metadata,
                      # FAISS search results implicitly have a score, but RetrievalQA chain doesn't easily expose per-doc scores directly here
                      # If scores are needed, we would need to run retriever.get_relevant_documents(query, return_scores=True) separately
                      # For simplicity with RetrievalQA chain, we won't include scores here by default
                 })

            return {
                "answer": result.get("answer", "I couldn't find a direct answer based on the provided information."),
                "retrieved_documents": retrieved_docs_info,
                "query": query
            }

        except Exception as e:
            app.logger.error(f"Error during LangChain QA chain execution for query '{query[:50]}...': {e}", exc_info=True)
            return {
                "answer": f"An error occurred while generating the answer. {e}",
                "retrieved_documents": [],
                "query": query
            }


    def get_all_documents(self):
        """Get summary of original documents from MongoDB."""
        if not self.mongodb_connected:
            app.logger.error("Cannot get all documents: MongoDB is not connected.")
            return {"ids": [], "documents": [], "metadatas": [], "total_documents": 0}

        try:
            documents = list(self.documents_collection.find({}, {"doc_id": 1, "text": 1, "metadata": 1}))
            return {
                "ids": [doc["doc_id"] for doc in documents],
                "documents": [doc["text"] for doc in documents], # Full text stored in MongoDB
                "metadatas": [doc["metadata"] for doc in documents],
                "total_documents": len(documents)
            }
        except PyMongoError as e:
            app.logger.error(f"MongoDB error getting all documents: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "total_documents": 0}
        except Exception as e:
            app.logger.error(f"Error getting all documents from MongoDB: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "total_documents": 0}

    def delete_document(self, doc_id: str, skip_mongo_delete: bool = False) -> bool:
        """Delete an original document from MongoDB and its corresponding chunks from FAISS."""
        
        # First, attempt to delete from FAISS
        faiss_deleted = False
        if self.vectorstore and self.langchain_ready:
            try:
                # Find chunk IDs associated with this doc_id in FAISS metadata
                # FAISS in LangChain currently doesn't have a direct delete_by_metadata method.
                # A common workaround is to search for related chunks, get their internal IDs, and delete by ID.
                # Or, if you know how metadata is stored, interact with the underlying index.
                # A simpler approach might involve creating a new index without the deleted documents,
                # but that's inefficient for single deletions.
                # LangChain's FAISS wrapper has a `delete` method that takes a list of ids (which correspond to the order added)
                # or a list of internal document indices. This is tricky without direct ID lookup.

                # Let's try a simplified approach: Get all docs from vectorstore and filter
                # This is NOT efficient for large indexes, but works for a simple example.
                # A proper solution might involve a custom FAISS subclass or tracking LC doc IDs.

                # Alternative: Rebuild index (inefficient but works) or use a DB that supports filtered deletion (e.g., Milvus, Pinecone)
                # Given the constraint to use FAISS, finding exact chunks to delete is hard.
                # Let's log a warning about the limitation and proceed with MongoDB deletion.
                # For a real app, you'd need to manage chunk IDs when adding to FAISS and store them in MongoDB
                # or use a vector store with better deletion by metadata support.

                # *** SIMPLIFIED DELETION ***
                # We will only delete from MongoDB and rebuild the FAISS index from scratch
                # with remaining documents. This is resource-intensive but correct for this example's limitations.

                # Rebuild FAISS index from remaining documents after MongoDB deletion
                app.logger.warning(f"Attempting to delete document {doc_id} from FAISS by rebuilding the index (inefficient).")
                faiss_deleted = True # Assume rebuild covers FAISS deletion conceptually


            except Exception as e:
                 app.logger.error(f"Error deleting document {doc_id} chunks from FAISS (rebuild method failed?): {e}")
                 # Continue to delete from MongoDB even if FAISS fails, to avoid inconsistency

        # Second, attempt to delete from MongoDB
        mongo_deleted = False
        if not skip_mongo_delete and self.mongodb_connected:
            try:
                doc_result = self.documents_collection.delete_one({"doc_id": doc_id})
                if doc_result.deleted_count > 0:
                    app.logger.info(f"Deleted document {doc_id} from MongoDB.")
                    mongo_deleted = True
                else:
                    app.logger.warning(f"Document {doc_id} not found in MongoDB for deletion.")
            except PyMongoError as e:
                app.logger.error(f"MongoDB error deleting document {doc_id}: {e}")
                raise # Re-raise failure
            except Exception as e:
                 app.logger.error(f"Error deleting document {doc_id} from MongoDB: {e}")
                 raise # Re-raise failure

        # If MongoDB deletion was successful (or skipped) and we are rebuilding FAISS
        if (mongo_deleted or skip_mongo_delete) and self.langchain_ready:
            try:
                # Rebuild FAISS index from remaining documents in MongoDB
                remaining_docs = list(self.documents_collection.find({}))
                if remaining_docs:
                    app.logger.info(f"Rebuilding FAISS index with {len(remaining_docs)} remaining documents.")
                    all_text = "\n---\n".join([doc["text"] for doc in remaining_docs])
                    # Need to re-process all remaining docs individually to get metadata linked
                    lc_docs_to_rebuild = [
                        Document(page_content=doc["text"], metadata={**(doc.get("metadata", {})), "doc_id": doc["doc_id"]})
                        for doc in remaining_docs
                    ]
                    all_chunks = self.text_splitter.split_documents(lc_docs_to_rebuild)

                    if all_chunks:
                         self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
                         self.vectorstore.save_local(FAISS_INDEX_PATH)
                         self.retriever = self.vectorstore.as_retriever()
                         # Re-initialize the QA chain if the vectorstore was just created/replaced
                         self.qa_chain = RetrievalQA.from_chain_type(
                            llm=self.llm,
                            chain_type="stuff",
                            retriever=self.retriever,
                            chain_type_kwargs={"prompt": PromptTemplate.from_template("""
                                You are an AI assistant designed to answer questions based *only* on the following context.
                                If the answer is not found in the context, please state that you cannot find the answer in the provided information.
                                Do not make up information.

                                Context:
                                {context}

                                Question:
                                {question}

                                Answer:
                                """)},
                            return_source_documents=True
                        )
                         app.logger.info(f"Rebuilt FAISS index with {self.vectorstore.index.ntotal} chunks from {len(remaining_docs)} documents.")
                    else:
                         # If no documents remain, delete the FAISS index files and reset vectorstore
                         app.logger.info("No documents remaining after deletion. Deleting FAISS index files.")
                         self.vectorstore = None # Reset vectorstore
                         self.retriever = None
                         self.qa_chain = None
                         if os.path.exists(FAISS_INDEX_PATH):
                             import shutil
                             shutil.rmtree(FAISS_INDEX_PATH)
                             app.logger.info(f"Deleted FAISS index directory: {FAISS_INDEX_PATH}")


            except Exception as e:
                 app.logger.error(f"Error rebuilding FAISS index after deleting {doc_id}: {e}", exc_info=True)
                 # Index might be in an inconsistent state. Log and move on.

        # Return True if the MongoDB deletion happened, as that's the primary record
        return mongo_deleted


    def get_database_stats(self):
        """Get statistics about MongoDB and FAISS."""
        mongo_stats = {"total_documents": 0, "database_name": MONGODB_DB_NAME, "mongodb_connected": self.mongodb_connected}
        if self.mongodb_connected:
            try:
                mongo_stats["total_documents"] = self.documents_collection.count_documents({})
            except Exception as e:
                app.logger.error(f"Error getting MongoDB document count: {e}")
                mongo_stats["error"] = "Failed to get document count from MongoDB."

        faiss_stats = {"total_chunks_in_faiss": 0, "faiss_index_path": FAISS_INDEX_PATH, "embedding_dimension": getattr(self, 'embedding_dim', 'N/A')}
        if self.vectorstore:
             try:
                 faiss_stats["total_chunks_in_faiss"] = self.vectorstore.index.ntotal if hasattr(self.vectorstore, 'index') else 'N/A'
             except Exception as e:
                 app.logger.error(f"Error getting FAISS index total: {e}")
                 faiss_stats["error"] = "Failed to get FAISS index stats."


        return {**mongo_stats, **faiss_stats} # Combine stats

# --- Initialize KB globally ---
# This uses the LangChain-based implementation
kb = LangchainRAGKnowledgeBase()
app.logger.info("✅ LangChain RAG Knowledge Base initialization attempted.")


# --- PDF Processing Utility Function ---
def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from a file-like object assumed to be a PDF."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        # Check if PDF is encrypted and if we can decrypt (requires password)
        if reader.is_encrypted:
             app.logger.warning("PDF is encrypted. Attempting decryption is not supported in this basic example.")
             raise Exception("Encrypted PDFs are not supported.")

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() or ""
    except Exception as e:
        app.logger.error(f"Error extracting text from PDF: {e}")
        raise
    return text

# --- Greeting and other conversational functions (Keep mostly as is) ---
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
        "what can you do",
        "how can you help",
        "how are you",
        "how's it going",
        "can you help",
        "need help",
        "assist me"
    ]
    for greeting_phrase in greetings:
        if greeting_phrase in text_lower:
            return True
    return False

def generate_greeting_response():
    time_greeting = get_time_based_greeting()
    return f"{time_greeting} How may I assist you? 😊"

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

def enhanced_generate_response(
    question: str,
    rag_result: Dict[str, Any],
) -> str:
    question_lower = question.lower().strip()

    if detect_greeting(question_lower):
        return generate_greeting_response()
    if detect_thanks(question_lower):
        return generate_thanks_response()

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

    # Check the answer from the LangChain RAG chain
    answer = rag_result.get("answer", "").strip()
    retrieved_docs = rag_result.get("retrieved_documents", [])

    # If the RAG chain produced a standard "not found" response or no relevant docs
    # We check both the answer text and if any documents were actually retrieved
    # LangChain's default prompt/chain might return "I couldn't find..." based on context
    # Or it might return a generic answer if no *relevant* context was found by the retriever
    if not retrieved_docs or answer.lower() in ["i couldn't find a direct answer based on the provided information.", ""]:
         return "I couldn't find relevant information for your question. 🤔 Could you rephrase or ask about something else?"

    # If we got a valid answer from the RAG chain based on retrieved docs
    response_text = answer

    # Optional: Append source information from retrieved docs
    source_info = ""
    if retrieved_docs:
         unique_sources = set()
         for doc in retrieved_docs:
             # Assuming metadata contains original doc_id or source info
             source_id = doc.get("metadata", {}).get("doc_id", "Unknown Source")
             unique_sources.add(source_id)
         if unique_sources:
              source_info = "\n\nSources: " + ", ".join(unique_sources)

    # LangChain RetrievalQA doesn't easily provide a single confidence score for the answer
    # We can only rely on the fact that it used the top retrieved docs.
    # Let's use a static emoji or one based on number of retrieved docs
    confidence_emoji = "📚" if retrieved_docs else "💡"


    return (
        f"{confidence_emoji} {response_text}{source_info}\n\n"
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
        if detect_greeting(text) or detect_thanks(text) or any(word in text.lower().strip() for word in ["bye", "goodbye", "good night"]):
            # For conversational simple replies, no RAG result is needed for the enhanced_generate_response function
            rag_result = {"answer": "", "retrieved_documents": []} # Pass empty result
            response = enhanced_generate_response(text, rag_result)
        else:
            # Check if RAG is ready
            if not kb.langchain_ready:
                 response = "Sorry, my knowledge base is not operational right now. Please check the server status and configuration (Gemini API, MongoDB, FAISS)."
            else:
                # Use the LangChain RAG chain via kb.search_and_answer
                rag_result = kb.search_and_answer(text)
                response = enhanced_generate_response(text, rag_result)

        if SLACK_BOT_TOKEN:
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
                    text="Sorry, an internal error occurred while processing your request. 😅",
                    thread_ts=event.get("ts"),
                )
            except Exception as slack_err:
                app.logger.error(f"Failed to send error message to Slack: {slack_err}")


# --- Flask Routes ---
@app.route("/slack/events", methods=["POST"])
def slack_events_route():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data["challenge"]})
    if data.get("type") == "event_callback":
        event = data.get("event")
        if not event:
            return jsonify({"status": "ok", "message": "No event payload"}), 200
        if event.get("subtype") == "bot_message" or event.get("bot_id"):
            app.logger.info("Ignoring bot message.")
            return jsonify({"status": "ok", "message": "Ignored bot message"})

        if event.get("type") == "app_mention" or (
            event.get("type") == "message" and event.get("channel_type") == "im"
        ):
            # Run the event handling in a non-blocking way if possible in production
            # For this example, sync is okay
            handle_message(event)
    return jsonify({"status": "ok"})


@app.route("/delete_knowledge/<doc_id>", methods=["DELETE"])
def delete_knowledge(doc_id):
    try:
        # Check if KB is ready enough for deletion
        if not kb.mongodb_connected: # MongoDB is primary source of original docs
             return jsonify({"error": "Knowledge base not ready: MongoDB not connected."}), 503

        success = kb.delete_document(doc_id)
        if success:
            return jsonify({"status": "success", "message": f"Document '{doc_id}' deleted from MongoDB and chunks removed from FAISS (via rebuild)."})
        else:
            return jsonify({"error": "Document not found or deletion failed in MongoDB."}), 404
    except Exception as e:
        app.logger.error(f"Error deleting knowledge: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/db_stats", methods=["GET"])
def database_stats_route():
    try:
        stats = kb.get_database_stats()
        # Check for critical errors in stats retrieval
        if not stats.get("mongodb_connected"):
            return jsonify({"status": "error", "message": stats.get("error", "MongoDB not connected.")}), 503
        if stats.get("error") and stats.get("mongodb_connected"): # Other errors but MongoDB connected
            return jsonify({"status": "warning", "message": stats.get("error", "Could not retrieve all stats.")}), 500 # Still return some stats if possible

        return jsonify({"status": "success", "stats": stats})
    except Exception as e:
        app.logger.error(f"Error getting database stats via route: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/add_knowledge", methods=["POST"])
def add_knowledge():
    try:
        data = request.json
        if not data or not data.get("id") or not data.get("text"):
            return jsonify({"error": "Missing 'id' or 'text' in request body."}), 400

        # Check if KB is ready for adding
        if not kb.mongodb_connected:
             return jsonify({"error": "Knowledge base not ready: MongoDB not connected."}), 503
        if not kb.langchain_ready: # Includes Gemini/FAISS setup check
             return jsonify({"error": "Knowledge base not ready: LangChain components (Gemini, FAISS) not initialized correctly. Check logs for API key or FAISS path issues."}), 503

        kb.add_document(data["id"], data["text"], data.get("metadata", {}))
        return jsonify(
            {
                "status": "success",
                "message": f"Document '{data['id']}' added/updated in MongoDB and indexed in FAISS using Gemini embeddings.",
            }
        )
    except (ConnectionError, RuntimeError) as e:
        # Handle specific errors from kb.add_document indicating readiness issues
        app.logger.error(f"KB readiness error adding knowledge: {e}")
        return jsonify({"error": str(e)}), 503
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
            doc_id = f"pdf_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}"

        metadata_str = request.form.get('metadata', '{}')
        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            metadata = {}
            app.logger.warning(f"Invalid metadata JSON provided for PDF upload: {metadata_str}")

        pdf_content = file.read()
        text = extract_text_from_pdf(io.BytesIO(pdf_content))

        if not text.strip():
            return jsonify({"error": "No text extracted from PDF. PDF might be image-based, encrypted, or empty."}), 400

        # Check if KB is ready for adding
        if not kb.mongodb_connected:
             return jsonify({"error": "Knowledge base not ready: MongoDB not connected."}), 503
        if not kb.langchain_ready: # Includes Gemini/FAISS setup check
             return jsonify({"error": "Knowledge base not ready: LangChain components (Gemini, FAISS) not initialized correctly. Check logs for API key or FAISS path issues."}), 503

        kb.add_document(doc_id, text, metadata)

        return jsonify({
            "status": "success",
            "message": f"PDF '{file.filename}' processed and added as document '{doc_id}' to MongoDB and indexed in FAISS.",
            "doc_id": doc_id,
            "extracted_text_preview": text[:200] + "..." if len(text) > 200 else text
        })

    except PyPDF2.errors.PdfReadError as e:
        app.logger.error(f"PDF Read Error during upload: {e}")
        return jsonify({"error": f"Failed to read PDF: {e}. It might be corrupted or encrypted."}), 400
    except (ConnectionError, RuntimeError) as e:
        app.logger.error(f"KB readiness error uploading PDF: {e}")
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        app.logger.error(f"Error uploading PDF: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/test_search", methods=["POST"])
def test_search():
    try:
        data = request.json
        if not data or not data.get("query"):
            return jsonify({"error": "Missing query"}), 400
        query = data["query"]
        # n_results, search_type are now handled by the RAG chain configuration, not explicit parameters here

        # Check if KB is ready for search
        if not kb.langchain_ready or kb.qa_chain is None:
             return jsonify({"error": "Knowledge base not ready for search. LangChain RAG chain not initialized."}), 503

        rag_result = kb.search_and_answer(query)

        # Format the retrieved documents nicely
        retrieved_docs_formatted = []
        for doc in rag_result.get("retrieved_documents", []):
             retrieved_docs_formatted.append({
                 "page_content": doc.page_content,
                 "metadata": doc.metadata,
                 # Scores are not directly available from RetrievalQA output
                 # If scores are critical for this endpoint, we'd need to modify kb.search_and_answer
                 # to call the retriever separately with score=True
             })

        return jsonify(
            {
                "query": query,
                "answer": rag_result.get("answer"),
                "retrieved_documents": retrieved_docs_formatted,
                # "search_type": "RAG (LangChain RetrievalQA)", # Indicate the type
                # "n_results_retrieved": len(retrieved_docs_formatted) # Can add retrieved count
            }
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
        
        # Get FAISS index size if available
        faiss_chunk_count = kb.vectorstore.index.ntotal if kb.vectorstore and hasattr(kb.vectorstore, 'index') else 0


        return jsonify(
            {
                "total_documents_in_db": docs_summary["total_documents"],
                "total_chunks_in_faiss": faiss_chunk_count,
                "embedding_model": getattr(kb, 'embeddings', None).__class__.__name__ if kb.embeddings else "Not Initialized",
                "embedding_dimension": getattr(kb, 'embedding_dim', 'N/A'),
                "llm_model": getattr(kb, 'llm', None).__class__.__name__ if kb.llm else "Not Initialized",
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
    # Check core dependencies
    is_mongodb_connected = kb.mongodb_connected
    is_gemini_key_configured = bool(GEMINI_API_KEY)
    is_slack_token_configured = bool(SLACK_BOT_TOKEN)
    is_langchain_ready = kb.langchain_ready

    # Check state of LangChain components
    embeddings_loaded = kb.embeddings is not None
    llm_loaded = kb.llm is not None
    vectorstore_loaded = kb.vectorstore is not None
    retriever_loaded = kb.retriever is not None
    qa_chain_loaded = kb.qa_chain is not None


    # Get DB stats (handle potential errors)
    db_stats = kb.get_database_stats()


    overall_status = "healthy"
    status_message = "All core components configured and initialized."

    if not is_gemini_key_configured:
        overall_status = "unhealthy"
        status_message = "Gemini API key not configured."
    elif not is_mongodb_connected:
         overall_status = "unhealthy"
         status_message = "MongoDB not connected."
    elif not is_slack_token_configured:
        overall_status = "warning" # Bot will run but not connect to Slack
        status_message = "Slack bot token not configured."
    elif not is_langchain_ready:
         overall_status = "unhealthy"
         status_message = "LangChain RAG components failed to initialize (check Gemini key or FAISS path)."
    elif not vectorstore_loaded:
         overall_status = "warning"
         status_message = "FAISS vector store not yet initialized/loaded. Add documents to create it."
    elif not qa_chain_loaded:
         overall_status = "warning"
         status_message = "LangChain QA chain not initialized (might be waiting for vector store)."


    return jsonify({
        "status": overall_status,
        "message": status_message,
        "dependencies": {
            "mongodb_connected": is_mongodb_connected,
            "gemini_api_key_configured": is_gemini_key_configured,
            "slack_bot_token_configured": is_slack_token_configured,
            "langchain_initialized": is_langchain_ready,
        },
        "rag_components": {
            "embeddings_loaded": embeddings_loaded,
            "llm_loaded": llm_loaded,
            "vectorstore_loaded": vectorstore_loaded,
            "retriever_loaded": retriever_loaded,
            "qa_chain_loaded": qa_chain_loaded,
        },
        "database_stats": db_stats
    })


@app.route("/")
def home():
    # Get stats for the home page display
    stats = kb.get_database_stats()
    
    mongo_connected_display = stats.get("mongodb_connected", False)
    gemini_api_display = bool(GEMINI_API_KEY)
    slack_token_display = bool(SLACK_BOT_TOKEN)
    langchain_ready_display = kb.langchain_ready
    vectorstore_loaded_display = kb.vectorstore is not None

    overall_status_display = "✅ Running"
    if not gemini_api_display or not mongo_connected_display or not langchain_ready_display:
        overall_status_display = "❌ Critical Configuration Issue!"
    elif not slack_token_display:
        overall_status_display = "⚠️ Slack Integration Issue!"
    elif not vectorstore_loaded_display:
         overall_status_display = "⚠️ FAISS Index Empty/Not Loaded!"


    return render_template_string("""
    <!doctype html>
    <html>
    <head><title>RAG Slack AI Bot Status</title></head>
    <body>
        <h1>🚀 RAG Slack AI Bot (LangChain + Gemini + FAISS)</h1>
        <p>Status: {{ overall_status_display }}</p>
        <h2>🔧 RAG Features:</h2>
        <ul>
            <li><strong>Framework:</strong> LangChain</li>
            <li><strong>Embedding Service:</strong> Google Gemini ({{ stats.get('embedding_model', 'N/A') }})</li>
            <li><strong>Vector DB:</strong> FAISS (Persistent at {{ stats.get('faiss_index_path', 'N/A') }})</li>
            <li><strong>Text Generation/QA:</strong> Google Gemini ({{ getattr(kb, 'llm', None).model_name if getattr(kb, 'llm', None) else 'Not Initialized' }})</li>
        </ul>
        <h2>📊 Stats:</h2>
        <ul>
            <li><strong>Embedding Dimension:</strong> {{ stats.get('embedding_dimension', 'N/A') }}</li>
            <li><strong>MongoDB Connection:</strong> {{ '✅ Connected' if stats.get('mongodb_connected', False) else '❌ NOT CONNECTED!' }}</li>
            <li><strong>Database Name:</strong> {{ stats.get('database_name', 'N/A') }}</li>
            <li><strong>Total Original Docs in DB:</strong> {{ stats.get('total_documents', 'N/A') }}</li>
            <li><strong>FAISS Index Status:</strong> {{ '✅ Loaded' if vectorstore_loaded_display else '❌ Not Loaded/Empty' }}</li>
            <li><strong>Total Chunks in FAISS Index:</strong> {{ stats.get('total_chunks_in_faiss', 'N/A') }}</li>
            <li><strong>Gemini API Key:</strong> {{ '✅ Configured' if gemini_api_display else '❌ NOT CONFIGURED!' }}</li>
            <li><strong>Slack Token:</strong> {{ '✅ Configured' if slack_token_display else '❌ NOT CONFIGURED!' }}</li>
             <li><strong>LangChain Ready:</strong> {{ '✅ Yes' if langchain_ready_display else '❌ No' }}</li>
        </ul>
        <p><em>See code/logs for API details. Ensure GOOGLE_API_KEY, SLACK_BOT_TOKEN, and MONGODB_URI are set in your .env file. FAISS index persists to the specified path.</em></p>
         <h3>Endpoints:</h3>
        <ul>
            <li><code>POST /add_knowledge</code> (json: {{"{ id: '...', text: '...', metadata: {{...}} }"}})</li>
            <li><code>POST /upload_pdf</code> (form-data: file=@file.pdf, optional doc_id, metadata)</li>
            <li><code>GET /list_knowledge</code></li>
            <li><code>DELETE /delete_knowledge/<doc_id></code></li>
            <li><code>POST /test_search</code> (json: {{"{ query: '...' }"}})</li>
            <li><code>GET /db_stats</code></li>
            <li><code>GET /health</code></li>
        </ul>
    </body>
    </html>
    """, overall_status_display=overall_status_display, stats=stats, kb=kb, vectorstore_loaded_display=vectorstore_loaded_display)


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

    # Only attempt to initialize sample data if core dependencies are met
    if not kb.mongodb_connected:
        app.logger.warning("Skipping sample data initialization: MongoDB not connected.")
        return
    if not kb.langchain_ready:
         app.logger.warning("Skipping sample data initialization: LangChain RAG components not ready (Gemini API issue?).")
         return

    try:
        existing_doc_count = kb.documents_collection.count_documents({})

        # Check if the FAISS index directory exists and is not empty
        faiss_exists = os.path.exists(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH)
        
        # Check if MongoDB has documents AND FAISS is not loaded/empty
        if existing_doc_count > 0 and (kb.vectorstore is None or kb.vectorstore.index.ntotal == 0):
             # This indicates a potential mismatch - docs in DB but not in index.
             # In a real app, you might trigger a rebuild here.
             # For this example, we'll warn and proceed.
             app.logger.warning(f"MongoDB has {existing_doc_count} documents, but FAISS index is empty or not loaded. Index might be out of sync.")

        # Add sample data only if MongoDB is empty AND FAISS is empty/not loaded
        # This prevents adding duplicates or overwriting existing indexes unless empty.
        if existing_doc_count == 0 and (kb.vectorstore is None or kb.vectorstore.index.ntotal == 0):
            app.logger.info(
                "🔄 Initializing RAG with sample data (using LangChain/Gemini/FAISS)..."
            )
            # Create a dummy document list for LangChain to process
            lc_sample_docs = [
                 Document(page_content=doc["text"], metadata={**(doc.get("metadata", {})), "doc_id": doc["id"]})
                 for doc in sample_docs
            ]
            
            # Use add_document method to store in MongoDB and add to FAISS
            for doc in sample_docs:
                 try:
                      kb.add_document(doc["id"], doc["text"], doc.get("metadata", {}))
                      app.logger.info(f"Successfully added sample document: {doc['id']}")
                 except Exception as e:
                      app.logger.error(f"Failed to add sample document {doc['id']}: {e}")


            app.logger.info(
                f"✅ Attempted to add {len(sample_docs)} sample docs. Current FAISS chunks: {kb.vectorstore.index.ntotal if kb.vectorstore and hasattr(kb.vectorstore, 'index') else 'N/A'}"
            )
        elif existing_doc_count > 0:
            app.logger.info(
                f"📚 MongoDB already contains {existing_doc_count} documents. Skipping sample data initialization."
            )
            # If DB has docs but FAISS wasn't loaded, kb.__init__ should have warned.
            # The FAISS load happens in __init__. If it failed, kb.vectorstore might be None.
            # We rely on __init__ to handle the initial load. If it failed, adding docs later will rebuild.

        elif not faiss_exists and existing_doc_count == 0:
             app.logger.info("MongoDB is empty and no existing FAISS index found. Sample data will be added if core KB components are ready.")
             # Sample data will be added by the loop above if langhcain_ready was True

        elif faiss_exists and kb.vectorstore is None:
             app.logger.warning(f"FAISS index directory exists at {FAISS_INDEX_PATH}, but failed to load into LangChain. A new index might overwrite it on first add.")

        else:
             app.logger.info("FAISS index exists and/or MongoDB has documents. Skipping sample data load.")


    except Exception as e:
        app.logger.error(f"Error during sample data initialization: {e}", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    app.logger.info("🚀 Starting RAG-Powered Slack AI Bot with LangChain (Gemini + FAISS)")

    # Check critical configs again for startup message clarity
    gemini_ok_startup = bool(GEMINI_API_KEY)
    slack_ok_startup = bool(SLACK_BOT_TOKEN)
    mongodb_ok_startup = kb.mongodb_connected
    langchain_ok_startup = kb.langchain_ready

    if not gemini_ok_startup:
        app.logger.critical(
            "FATAL: GOOGLE_API_KEY is not configured. Embedding and generation will fail. LangChain RAG is likely not initialized."
        )
    if not slack_ok_startup:
        app.logger.critical(
            "FATAL: SLACK_BOT_TOKEN is not configured. Bot cannot connect to Slack."
        )
    if not mongodb_ok_startup:
        app.logger.critical(
            "FATAL: MongoDB is not connected. Original documents cannot be stored/retrieved."
        )
    if not langchain_ok_startup:
         app.logger.critical(
             "FATAL: LangChain RAG components failed to initialize. Check logs for specific errors (Gemini API key, FAISS path)."
         )


    # Initialize sample data only if core components *needed for adding* are okay
    # MongoDB is needed for original doc storage, LangChain/Gemini/FAISS is needed for indexing
    if mongodb_ok_startup and langchain_ok_startup:
        initialize_sample_data()
    else:
        app.logger.warning(
            "Skipping sample data initialization due to critical configuration issues."
        )


    app.logger.info("\n" + "=" * 80)
    app.logger.info("🎯 LangChain RAG Bot is ready (or attempting to be)!")

    # Access components safely for logging
    embedding_model_name = getattr(getattr(kb, 'embeddings', None), 'model_name', 'Not Initialized')
    llm_model_name = getattr(getattr(kb, 'llm', None), 'model_name', 'Not Initialized')
    vectorstore_status = "✅ Loaded" if kb.vectorstore is not None else "❌ Not Loaded/Empty"


    app.logger.info(
        f"   • Embeddings by: Google Gemini ('{embedding_model_name}') {'✅' if gemini_ok_startup else '❌ (KEY ISSUE!)'}"
    )
    app.logger.info(
        f"   • Text Generation (QA): Google Gemini ('{llm_model_name}') {'✅' if getattr(kb, 'llm', None) else '❌ (ISSUE!)'}"
    )
    app.logger.info(
        f"   • Vector Store (FAISS): {vectorstore_status}"
    )
    app.logger.info(
        f"   • MongoDB Connection: {'✅' if mongodb_ok_startup else '❌ (CONNECTION ISSUE!)'}"
    )
    app.logger.info(
        f"   • Slack Integration: {'✅' if slack_ok_startup else '❌ (TOKEN ISSUE!)'}"
    )
    app.logger.info(
        f"🌐 Web interface: http://localhost:{os.environ.get('PORT', 3000)}"
    )
    app.logger.info("=" * 80 + "\n")

    if not gemini_ok_startup or not slack_ok_startup or not mongodb_ok_startup or not langchain_ok_startup:
        app.logger.warning(
            "CRITICAL CONFIGURATION ISSUES DETECTED. BOT MAY NOT FUNCTION CORRECTLY. Check logs above."
        )

    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)