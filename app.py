import os
import json
import random
from datetime import datetime
import io
import threading 
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import PyPDF2



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
            app.logger.error("Cannot initialize ChromaDB: Embeddings model is not available.")
            return
        
        try:
            # Chroma will create the directory if it doesn't exist and load it if it does.
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            app.logger.info(f"ChromaDB vector store initialized from/to: {self.persist_directory}")
        except Exception as e:
            app.logger.critical(f"CRITICAL: Failed to initialize ChromaDB vector store: {e}")

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

    ### MODIFICATION START: Define the intent classification chain ###
    intent_prompt_template = """You are an expert at classifying user intent. Analyze the user's message and classify it into one of the following categories:
- GREETING: For hellos, good mornings, or general greetings.
- FAREWELL: For goodbyes, see you later, etc.
- THANKS: For expressions of gratitude.
- KNOWLEDGE_BASE_QUERY: For any question that seeks specific information, asks for an explanation, or requests data that would likely be found in documents. This is the default for any real question.

User message: "{user_input}"

You must respond with ONLY the category name and nothing else.
Category:"""
    INTENT_PROMPT = PromptTemplate(template=intent_prompt_template, input_variables=["user_input"])
    
    intent_classifier_chain = INTENT_PROMPT | llm 
    app.logger.info("Intent Classification chain initialized.")
    ### MODIFICATION END ###

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
            app.logger.info("LangChain RAG chain initialized.")
        else:
            app.logger.error("Failed to get retriever from ChromaDB. RAG chain not initialized.")
    else:
        app.logger.error("RAG chain not initialized due to missing LLM or Vector Store.")
else:
    app.logger.critical("Knowledge Base, RAG chain, and Intent Classifier skipped due to OpenAI unavailability.")


# --- PDF Processing Utility ---
def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
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
    "Hello there! I'm your friendly knowledge assistant. How can I help you explore our knowledge base today?",
    "Hi! Great to see you! What would you like to discover in our knowledge base?",
    "Hey there! I'm here and ready to help you find answers. What's on your mind?",
]
FAREWELLS = [
    "Goodbye! It was wonderful helping you today. Feel free to come back anytime!",
    "Farewell! Hope I was able to help. Don't hesitate to reach out again!",
    "See you later! Thanks for the great conversation. I'll be here if you need me!",
]
THANKS_RESPONSES = [
    "You're absolutely welcome! Happy to help anytime!",
    "My pleasure! That's what I'm here for!",
    "No problem at all! Glad I could assist you!",
]


def get_random_response(response_list):
    return random.choice(response_list)

# --- Slack Event Handler ---
def handle_message(event):
    channel = event["channel"]
    user_id = event.get("user")
    
    # Skip if this is the bot's own message
    if user_id == SLACK_BOT_USER_ID:
        return
    
    # Get original text and remove bot mention
    original_text = event.get("text", "")
    text = original_text.replace(f"<@{SLACK_BOT_USER_ID}>", "").strip()
    thread_ts = event.get("ts") # Get thread_ts to reply in thread if it's a threaded message

    # Ignore empty messages
    if not text:
        return

    response_text = ""

    # --- Enhanced Conversational Logic ---
    if not intent_classifier_chain:
        response_text = "I'm sorry, but my core intelligence system is currently offline. Please notify an administrator."
        app.logger.error("Intent classifier chain is not available for Slack request.")
    else:
        try:
            # Step 1: Classify the user's intent using cleaned text
            intent_result = intent_classifier_chain.invoke({"user_input": text})
            
            # Extract intent from result (handle both string and dict responses)
            if isinstance(intent_result, dict):
                intent = intent_result.get("intent", intent_result.get("output", ""))
            else:
                intent = str(intent_result)
            
            app.logger.info(f"Classified intent as '{intent}' for query: '{text}'")

            # Step 2: Route based on intent (case-insensitive matching)
            intent_upper = intent.upper()
            
            if "GREETING" in intent_upper:
                response_text = get_random_response(GREETINGS)
            elif "THANKS" in intent_upper or "THANK" in intent_upper:
                response_text = get_random_response(THANKS_RESPONSES)
            elif "FAREWELL" in intent_upper or "GOODBYE" in intent_upper or "BYE" in intent_upper:
                response_text = get_random_response(FAREWELLS)
            elif "KNOWLEDGE_BASE_QUERY" in intent_upper or "QUESTION" in intent_upper:
                # This is a real question, so we use the RAG chain
                if not rag_chain:
                    response_text = "I'm sorry, but my knowledge base is currently unavailable. Please try again later."
                    app.logger.error("RAG chain not available for a knowledge base query.")
                else:
                    app.logger.info(f"Invoking RAG chain for query: '{text}'")
                    result = rag_chain.invoke({"query": text})
                    
                    if result.get("result"):
                        response_text = result["result"]
                    else:
                        response_text = "I couldn't find any relevant information in the knowledge base to answer that. Could you try rephrasing your question?"
            else:
                # Fallback - try to detect common greeting patterns manually
                text_lower = text.lower()
                greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
                farewell_words = ['bye', 'goodbye', 'see you', 'farewell', 'take care']
                thanks_words = ['thanks', 'thank you', 'thx', 'appreciated']
                
                if any(word in text_lower for word in greeting_words):
                    response_text = get_random_response(GREETINGS)
                elif any(word in text_lower for word in farewell_words):
                    response_text = get_random_response(FAREWELLS)
                elif any(word in text_lower for word in thanks_words):
                    response_text = get_random_response(THANKS_RESPONSES)
                else:
                    # Default to treating as a question if no clear intent
                    if not rag_chain:
                        response_text = "I'm sorry, but my knowledge base is currently unavailable. Please try again later."
                        app.logger.error("RAG chain not available for fallback query.")
                    else:
                        app.logger.info(f"Treating as knowledge query (fallback): '{text}'")
                        result = rag_chain.invoke({"query": text})
                        
                        if result.get("result"):
                            response_text = result["result"]
                        else:
                            response_text = "I'm not quite sure how to handle that. I can answer questions about our documents or have a simple chat. How can I help?"

        except Exception as e:
            app.logger.error(f"Error during intent classification or RAG chain invocation: {e}", exc_info=True)
            response_text = "Sorry, I encountered an error while trying to understand your request. Please try again!"

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

@app.route("/slack/events", methods=["POST"])
def slack_events_route():
    data = request.json
    
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data["challenge"]})
    if data.get("type") == "event_callback":
        event = data.get("event", {})
        if not event.get("bot_id") and event.get("user") != SLACK_BOT_USER_ID:
            # Handle app mentions (when bot is @mentioned)
            if event.get("type") == "app_mention":
                thread = threading.Thread(target=handle_message, args=(event,))
                thread.start()
            # Handle direct messages (only if it's NOT already an app_mention)
            elif event.get("type") == "message" and event.get("channel_type") == "im":
                thread = threading.Thread(target=handle_message, args=(event,))
                thread.start()
            
    return jsonify({"status": "ok"}) 

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

@app.route("/question-gen", methods=["POST"])
def question_gen_route():
    if not rag_chain or not hasattr(rag_chain, 'llm') or not hasattr(rag_chain, 'retriever'):
        app.logger.error("RAG chain, LLM, or retriever not initialized for /question-gen.")
        return jsonify({"error": "RAG chain or its components not properly initialized."}), 503

    data = request.json
    if not data or not data.get("text"):
        app.logger.warning("/question-gen called with missing 'text' in request body.")
        return jsonify({"error": "Missing 'text' in request body."}), 400

    input_text = data["text"]
    app.logger.info(f"/question-gen received input text: '{input_text[:100]}...'")

    try:
        app.logger.info(f"Retrieving relevant documents for: '{input_text[:100]}...'")
        retrieved_docs = rag_chain.retriever.get_relevant_documents(input_text)
        app.logger.info(f"Retrieved {len(retrieved_docs)} documents.")

        retrieved_context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt_for_llm = f"""You are an expert assistant designed to generate insightful questions.
            Based on the Original Input Text and the Retrieved Context from a knowledge base below, generate exactly 3 distinct questions.
            These questions should help explore the topics further or clarify ambiguities found in the provided texts.
            Output ONLY the questions, each on a new line. Do not include any preamble, numbering, or extraneous text.

            Original Input Text:

            {input_text}


            Retrieved Context from Knowledge Base:


            Generated Questions (each on a new line):
        """
        app.logger.info("Invoking LLM to generate questions.")
        llm_response = rag_chain.llm.invoke(prompt_for_llm)
        
        generated_questions_raw = ""
        if hasattr(llm_response, 'content'): 
            generated_questions_raw = llm_response.content
        elif isinstance(llm_response, str): 
            generated_questions_raw = llm_response
        else:
            app.logger.warning(f"Unexpected LLM response type: {type(llm_response)}")
        
        app.logger.info(f"Raw LLM output for questions: '{generated_questions_raw}'")

        
        questions_list = [q.strip() for q in generated_questions_raw.split('\n') if q.strip()]
        
        
        if not questions_list and generated_questions_raw.strip():
            questions_list = [generated_questions_raw.strip()]
            app.logger.info("Parsed questions list was empty, using raw output as a single question.")
        
        app.logger.info(f"Successfully generated {len(questions_list)} questions.")

        full_combined_context_for_debug = f"Input Text:\n{input_text}\n\nRetrieved Context:\n{retrieved_context}"


        return jsonify({
            "input_text": input_text,
            "retrieved_context_preview": (retrieved_context[:300] + "...") if retrieved_context else "N/A",
            "combined_context_preview_for_prompting_structure": (full_combined_context_for_debug[:500] + "...") if full_combined_context_for_debug else "N/A",
            "generated_questions": questions_list,
            "source_documents": [{"text": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs]
        })

    except Exception as e:
        app.logger.error(f"Error in /question-gen for input '{input_text[:50]}...': {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred while generating questions: {str(e)}"}), 500


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
            "intent_classifier_chain": {"status": "ok" if intent_classifier_chain else "error"},
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
    app.logger.info(f"Starting RAG Bot on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)