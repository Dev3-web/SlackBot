import os
import json
import random
from datetime import datetime
import io
import threading # Used for optional background processing to ensure quick Slack response

# --- Core Imports ---
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging
from dotenv import load_dotenv

# --- Vector Store Imports (ChromaDB is now the sole DB) ---
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# --- Utility Imports ---
import PyPDF2 # Corrected import casing

# Load environment variables from .env file
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# --- Configuration ---
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db_store")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_BOT_USER_ID = os.environ.get("SLACK_BOT_USER_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# --- Initialize Core LangChain Components ---
llm = None
embeddings = None
openai_available = False

if not OPENAI_API_KEY:
    app.logger.critical("CRITICAL: OPENAI_API_KEY is not set. RAG features will be disabled.")
else:
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=OPENAI_API_KEY)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
        app.logger.info("OpenAI LLM and Embeddings models initialized.")
        openai_available = True
    except Exception as e:
        app.logger.critical(f"Failed to initialize OpenAI components: {e}. RAG features disabled.")

# --- Initialize Slack Client ---
if not SLACK_BOT_TOKEN:
    app.logger.critical("CRITICAL: SLACK_BOT_TOKEN not found. Slack integration will fail.")
slack_client = WebClient(token=SLACK_BOT_TOKEN or "dummy-token")

# --- RAG Knowledge Base Class (ChromaDB Only) ---
class RAGKnowledgeBase:
    def __init__(self, embeddings_model, persist_directory):
        self.persist_directory = persist_directory
        self.embeddings = embeddings_model
        
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        self._init_vectorstore()

    def _init_vectorstore(self):
        if not self.embeddings:
            app.logger.error("❌ Cannot initialize ChromaDB: Embeddings model is not available.")
            return
        
        try:
            # Chroma will create the directory if it doesn't exist and load it if it does.
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            app.logger.info(f"✅ ChromaDB vector store initialized from/to: {self.persist_directory}")
        except Exception as e:
            app.logger.critical(f"❌ CRITICAL: Failed to initialize ChromaDB vector store: {e}")

    def add_document(self, doc_id: str, text: str, metadata: dict = None):
        if not self.vectorstore:
            raise Exception("ChromaDB vector store not initialized.")

        # First, delete existing document chunks if it's an update
        self.delete_document(doc_id, silent=True) # silent=True avoids logging "not found" on new adds

        # 1. Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        if not chunks:
            app.logger.warning(f"No chunks were generated for document {doc_id}. Nothing to add to vector store.")
            return

        # 2. Create LangChain Document objects for each chunk, embedding all necessary metadata
        chunk_docs = []
        chunk_ids = []
        base_metadata = metadata or {}
        creation_time = datetime.utcnow().isoformat() + "Z"
        
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                "doc_id": doc_id,
                "chunk_index": i,
                "created_at": creation_time,
                **base_metadata
            }
            # Create a unique ID for each chunk for precise control
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_docs.append(Document(page_content=chunk_text, metadata=chunk_metadata))
            chunk_ids.append(chunk_id)

        # 3. Add chunks to ChromaDB. It handles embedding and indexing automatically.
        self.vectorstore.add_documents(documents=chunk_docs, ids=chunk_ids)
        app.logger.info(f"Added {len(chunk_docs)} chunks for document '{doc_id}' to ChromaDB.")

    def delete_document(self, doc_id: str, silent: bool = False):
        if not self.vectorstore:
            raise Exception("ChromaDB vector store not initialized.")
        
        existing_chunks = self.vectorstore._collection.get(where={"doc_id": doc_id})
        num_to_delete = len(existing_chunks.get("ids", []))

        if num_to_delete == 0:
            if not silent:
                app.logger.warning(f"Document '{doc_id}' not found in ChromaDB; no chunks to delete.")
            return False
        
        self.vectorstore._collection.delete(where={"doc_id": doc_id})
        
        if not silent:
            app.logger.info(f"Deleted {num_to_delete} chunks for document '{doc_id}' from ChromaDB.")
        return True

    def search(self, query: str, n_results: int = 5) -> list:
        if not self.vectorstore:
            app.logger.warning("Vector store not initialized. Cannot search.")
            return []
        
        results_with_scores = self.vectorstore.similarity_search_with_relevance_scores(query, k=n_results)
        
        formatted_results = []
        for doc, score in results_with_scores:
            formatted_results.append({
                "chunk_text": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            })
        return formatted_results

    def get_retriever(self, search_kwargs={"k": 5}):
        if not self.vectorstore:
            return None
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def get_all_documents(self):
        """
        Retrieves a summary of all unique documents by fetching all chunk
        metadata from ChromaDB and de-duplicating by 'doc_id'.
        """
        if not self.vectorstore:
            return []

        all_records = self.vectorstore._collection.get(include=["metadatas"])
        all_metadata = all_records.get("metadatas", [])
        
        unique_docs = {}
        for meta in all_metadata:
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in unique_docs:
                unique_docs[doc_id] = {
                    "doc_id": doc_id,
                    "created_at": meta.get("created_at"),
                    "metadata": {k: v for k, v in meta.items() if k not in ['doc_id', 'chunk_index', 'created_at']}
                }
        
        return list(unique_docs.values())

    def get_database_stats(self):
        stats = {
            "vector_store_type": "ChromaDB",
            "vector_store_initialized": self.vectorstore is not None,
            "total_unique_documents": 0,
            "total_chunks_in_vector_store": 0,
        }
        if self.vectorstore:
            stats["total_chunks_in_vector_store"] = self.vectorstore._collection.count()
            stats["total_unique_documents"] = len(self.get_all_documents())
        return stats

# --- Global Initialization ---
kb = None
rag_chain = None

if openai_available:
    kb = RAGKnowledgeBase(embeddings, CHROMA_PERSIST_DIR)
    
    if kb.vectorstore and llm:
        retriever = kb.get_retriever()
        if retriever:
            rag_prompt_template = """You are a helpful AI assistant. Answer the question based ONLY on the following context. If you don't know the answer from the context, say you couldn't find the information in the knowledge base. Do not make up information.

Context:
{context}

Question: {question}

Answer:"""
            RAG_PROMPT = PromptTemplate(template=rag_prompt_template, input_variables=["context", "question"])
            
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": RAG_PROMPT},
                return_source_documents=True
            )
            app.logger.info("✅ LangChain RAG chain initialized.")
        else:
            app.logger.error("❌ Failed to get retriever from ChromaDB. RAG chain not initialized.")
    else:
        app.logger.error("❌ RAG chain not initialized due to missing LLM or Vector Store.")
else:
    app.logger.critical("❌ Knowledge Base and RAG chain skipped due to OpenAI unavailability.")


# --- PDF Processing Utility ---
def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file) # Corrected usage
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    except Exception as e:
        app.logger.error(f"Error extracting text from PDF: {e}")
        raise
    return text.strip()


# --- Enhanced Conversational Helpers ---

GREETINGS = [
    "Hello there! 👋 I'm your friendly knowledge assistant. How can I help you ?",
    "Hi! 🌟 Great to see you! What would you like to discover?",
    "Hey there! 😊 I'm here and ready to help you find answers. What's on your mind?",
    "Greetings! 🚀 I'm your AI knowledge companion. Ask me anything about our documents and data!",
    "Hello! ✨ Welcome! I'm excited to helping you . What can I find for you?",
    "Hi there! 🎯 I'm your personal knowledge assistant. Ready to dive into some questions?",
    "Hey! 💡 Good to see you! I'm here to help you.",
    "Hello and welcome! 🔍 I'm your dedicated search assistant. What information are you looking for today?"
]

FAREWELLS = [
    "Goodbye! 👋 It was wonderful helping you today. Feel free to come back anytime with more questions!",
    "Farewell! 🌟 Hope I was able to help. Don't hesitate to reach out whenever you need assistance!",
    "See you later! 😊 Thanks for the great conversation. I'll be here whenever you need me!",
    "Bye for now! 🚀 Keep exploring and learning. I'm always here when you need answers!",
    "Take care! ✨ It's been a pleasure assisting you. Come back anytime with new questions!",
    "Goodbye and have a fantastic day! 🎯 Remember, I'm just a message away when you need help!",
    "Until next time! 💡 Hope you found what you were looking for. Stay curious!",
    "Farewell! 🔍 Thanks for letting me help. I'm always ready for your next knowledge quest!"
]

THANKS_RESPONSES = [
    "You're absolutely welcome! 😊 Happy to help anytime!",
    "My pleasure! 🌟 That's what I'm here for!",
    "No problem at all! 🚀 Glad I could assist you!",
    "Anytime! 💡 I love helping people find answers!",
    "You're so welcome! ✨ It makes me happy to be helpful!",
    "Happy to help! 🎯 Feel free to ask more questions anytime!",
    "Of course! 👋 Helping you succeed is my favorite thing!",
    "Always a pleasure! 🔍 I'm here whenever you need assistance!"
]

def get_random_response(response_list):
    return random.choice(response_list)

def detect_greeting(text):
    text = text.lower().strip()
    greeting_patterns = [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "greetings", "hiya", "howdy", "what's up", "whats up", "sup",
        "good day", "hei", "hola", "bonjour", "namaste"
    ]
    # Check if text starts with greeting or is just a greeting
    return any(text.startswith(pattern) or text == pattern for pattern in greeting_patterns)

def detect_farewell(text):
    text = text.lower().strip()
    farewell_patterns = [
        "bye", "goodbye", "farewell", "see ya", "see you", "cya", "talk later",
        "catch you later", "until next time", "take care", "so long",
        "have a good day", "have a great day", "good night", "goodnight",
        "peace", "later", "bye bye", "see you later", "ttyl", "gotta go"
    ]
    return any(pattern in text for pattern in farewell_patterns)

def detect_thanks(text):
    text = text.lower().strip()
    thanks_patterns = [
        "thanks", "thank you", "thx", "appreciate it", "appreciate",
        "grateful", "cheers", "much appreciated", "thanks a lot",
        "thank you so much", "ty", "tysm", "appreciated"
    ]
    return any(pattern in text for pattern in thanks_patterns)


# --- Slack Event Handler ---
def handle_message(event):
    channel = event["channel"]
    user_id = event.get("user")
    
    # Skip if this is the bot's own message
    if user_id == SLACK_BOT_USER_ID:
        return
    
    # Remove bot mention from text, if present, and strip whitespace
    text = event.get("text", "").replace(f"<@{SLACK_BOT_USER_ID}>", "").strip()
    thread_ts = event.get("ts") # Get thread_ts to reply in thread if it's a threaded message

    # Ignore empty messages
    if not text:
        return

    response_text = ""

    # --- Enhanced Conversational Logic ---
    if detect_greeting(text):
        response_text = get_random_response(GREETINGS)
    elif detect_thanks(text):
        response_text = get_random_response(THANKS_RESPONSES)
    elif detect_farewell(text):
        response_text = get_random_response(FAREWELLS)
    else:
        # --- RAG Logic ---
        if not rag_chain:
            response_text = "I'm sorry, but my knowledge system is currently unavailable. Please check with the administrator or try again later. 🔧"
            app.logger.error("RAG chain not available for Slack request.")
        else:
            try:
                app.logger.info(f"Invoking RAG chain for query: '{text}'")
                result = rag_chain.invoke({"query": text})
                
                if result.get("result"): # Ensure there's a result from the LLM
                    response_text = result["result"]
                    # Optional: Add source document info for transparency
                    # if result.get("source_documents"):
                    #     sources = [doc.metadata.get('doc_id', 'Unknown') for doc in result['source_documents']]
                    #     response_text += f"\n\n_Source(s): {', '.join(sorted(list(set(sources))))}_"
                else:
                    response_text = "I couldn't find any relevant information in the knowledge base to answer your question. Try rephrasing or asking about something else! 🔍"

            except Exception as e:
                app.logger.error(f"Error during RAG chain invocation: {e}", exc_info=True)
                response_text = "Sorry, I encountered an error while searching my knowledge base. Please try again! ⚠️"

    # --- Send Response to Slack ---
    try:
        slack_client.chat_postMessage(
            channel=channel,
            text=response_text,
            thread_ts=thread_ts # Reply in thread if message was in a thread
        )
        app.logger.info(f"Sent response to channel {channel}: {response_text[:100]}...")
    except SlackApiError as e:
        app.logger.error(f"Error sending Slack message: {e.response['error']}")

# --- Flask Routes ---

@app.route("/slack/events", methods=["POST"])
def slack_events_route():
    data = request.json
    
    # Crucial: Immediately acknowledge the Slack request to prevent retries.
    # The actual processing will happen in a separate thread/background task.
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data["challenge"]})
    
    # Process event in a non-blocking way
    if data.get("type") == "event_callback":
        event = data.get("event", {})
        
        # FIXED: Only process app_mention events OR direct messages to avoid duplicate responses
        # Skip bot messages and only handle one event type per message
        if not event.get("bot_id") and event.get("user") != SLACK_BOT_USER_ID:
            # Handle app mentions (when bot is @mentioned)
            if event.get("type") == "app_mention":
                thread = threading.Thread(target=handle_message, args=(event,))
                thread.start()
            # Handle direct messages (only if it's NOT already an app_mention)
            elif event.get("type") == "message" and event.get("channel_type") == "im":
                thread = threading.Thread(target=handle_message, args=(event,))
                thread.start()
            
    return jsonify({"status": "ok"}) # Acknowledge receipt immediately


@app.route("/add_knowledge", methods=["POST"])
def add_knowledge_route():
    if not kb: return jsonify({"error": "Knowledge Base not initialized."}), 503
    data = request.json
    if not data or not data.get("id") or not data.get("text"):
        return jsonify({"error": "Missing 'id' or 'text'."}), 400
    try:
        kb.add_document(data["id"], data["text"], data.get("metadata", {}))
        return jsonify({"status": "success", "message": f"Document '{data['id']}' added/updated."})
    except Exception as e:
        app.logger.error(f"Error in /add_knowledge: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf_route():
    if not kb: return jsonify({"error": "Knowledge Base not initialized."}), 503
    if 'file' not in request.files: return jsonify({"error": "No file part."}), 400
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Invalid or no selected file."}), 400

    doc_id = request.form.get('doc_id') or f"pdf_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    metadata_str = request.form.get('metadata', '{}')
    try:
        metadata = json.loads(metadata_str)
        metadata['original_filename'] = file.filename
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid metadata JSON."}), 400

    try:
        pdf_content = file.read()
        text = extract_text_from_pdf(io.BytesIO(pdf_content))
        if not text: return jsonify({"error": "No text extracted from PDF."}), 400
        
        kb.add_document(doc_id, text, metadata)
        
        return jsonify({
            "status": "success",
            "message": f"PDF '{file.filename}' processed and added as document '{doc_id}'.",
            "doc_id": doc_id,
            "extracted_text_preview": text[:200] + "..."
        })
    except Exception as e:
        app.logger.error(f"Error in /upload_pdf: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/delete_knowledge/<doc_id>", methods=["DELETE"])
def delete_knowledge_route(doc_id):
    if not kb: return jsonify({"error": "Knowledge Base not initialized."}), 503
    try:
        if kb.delete_document(doc_id):
            return jsonify({"status": "success", "message": f"Document '{doc_id}' deleted."})
        else:
            return jsonify({"error": "Document not found."}), 404
    except Exception as e:
        app.logger.error(f"Error in /delete_knowledge: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/list_knowledge", methods=["GET"])
def list_knowledge_route():
    if not kb: return jsonify({"error": "Knowledge Base not initialized."}), 503
    try:
        documents = kb.get_all_documents()
        return jsonify({"documents": documents, "total_unique_documents": len(documents)})
    except Exception as e:
        app.logger.error(f"Error in /list_knowledge: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
        
@app.route("/test_search", methods=["POST"])
def test_search_route():
    if not kb: return jsonify({"error": "Knowledge Base not initialized."}), 503
    data = request.json
    if not data or not data.get("query"): return jsonify({"error": "Missing query"}), 400
    try:
        results = kb.search(data["query"], data.get("n_results", 3))
        return jsonify({"query": data["query"], "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test_rag_gen", methods=["POST"])
def test_rag_gen_route():
    if not rag_chain: return jsonify({"error": "RAG chain not initialized."}), 503
    data = request.json
    if not data or not data.get("query"): return jsonify({"error": "Missing query"}), 400
    try:
        result = rag_chain.invoke({"query": data["query"]})
        source_docs = [{"text": doc.page_content, "metadata": doc.metadata} for doc in result.get('source_documents', [])]
        return jsonify({
            "query": data["query"],
            "generated_answer": result.get("result"),
            "source_documents": source_docs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check_route():
    db_stats = kb.get_database_stats() if kb else {}
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "openai_api": {"status": "ok" if openai_available else "error"},
            "llm_model": {"status": "ok" if llm else "error", "model_name": getattr(llm, 'model_name', 'N/A')},
            "embedding_model": {"status": "ok" if embeddings else "error", "model_name": getattr(embeddings, 'model_name', 'N/A')},
            "vector_store": {"status": "ok" if kb and kb.vectorstore else "error", **db_stats},
            "rag_chain": {"status": "ok" if rag_chain else "error"},
            "slack_token": {"status": "ok" if SLACK_BOT_TOKEN else "error"},
        }
    }
    is_healthy = all(comp["status"] == "ok" for comp in status["components"].values())
    status["status"] = "healthy" if is_healthy else "unhealthy"
    
    return jsonify(status), 200 if is_healthy else 503

@app.route("/")
def home_route():
    return "<h1>RAG Slack Bot with ChromaDB is running.</h1><p>Check the <a href='/health'>/health</a> endpoint for detailed status.</p>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.logger.info(f"🚀 Starting RAG Bot on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)