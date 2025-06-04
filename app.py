import os
import json
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging
from dotenv import load_dotenv
import sqlite3
import re

# from collections import Counter # Unused
# import math # Unused
import random
from datetime import datetime
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Optional  # Tuple unused
import openai
from transformers import pipeline
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Configure Gemini API Key
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    # Fallback for the hardcoded one if GOOGLE_API_KEY is not set, but strongly advise against it.
    GEMINI_API_KEY = "AIzaSyCnpefG1lhxQwvkDOoCdbRJUbFu1Icp08w"  # Replace with your actual key if testing without .env

if (
    not GEMINI_API_KEY
    or GEMINI_API_KEY == "YOUR_GOOGLE_API_KEY_HERE"
    or GEMINI_API_KEY == "AIzaSyCnpefG1lhxQwvkDOoCdbRJUbFu1Icp08w"
):  # Added check for placeholder
    app.logger.error(
        "CRITICAL: Gemini API Key is not configured or is a placeholder. The application will likely fail to get embeddings."
    )
    # You might want to exit here if the API key is essential and not set
    # sys.exit("Gemini API Key not configured.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    app.logger.info("Gemini API Key configured.")


# Initialize Flask app


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
    def __init__(
        self,
        db_path="knowledge_base.db",
    ):
        self.db_path = db_path
        self.gemini_embedding_model_id = (
            "models/text-embedding-004"  # Current recommended model
        )
        self.model_name = self.gemini_embedding_model_id
        self.embedding_dim = 768  # Dimension for models/text-embedding-004

        app.logger.info(
            f"Initializing RAG with Gemini model: {self.gemini_embedding_model_id} (Dimension: {self.embedding_dim})"
        )
        app.logger.warning(
            "If you have an existing 'knowledge_base.db' created with a different embedding model, "
            "it will likely be incompatible. Old embeddings might be skipped during loading. "
            "Consider deleting 'knowledge_base.db' to re-initialize and re-populate with Gemini embeddings."
        )

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.doc_ids = []
        self.doc_texts = []
        self.doc_metadatas = []

        try:
            self.qa_pipeline = pipeline(
                "question-answering", model="distilbert-base-cased-distilled-squad"
            )
            app.logger.info(
                "QA pipeline 'distilbert-base-cased-distilled-squad' loaded successfully."
            )
        except Exception as e:
            self.qa_pipeline = None
            app.logger.warning(
                f"QA pipeline not loaded (model 'distilbert-base-cased-distilled-squad'). Error: {e}"
            )
            app.logger.warning(
                "Using simple context retrieval instead of QA model for answers."
            )

        self.init_database()
        self.load_existing_embeddings()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata TEXT,
                embedding BLOB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id TEXT PRIMARY KEY, doc_id TEXT, chunk_text TEXT NOT NULL,
                chunk_embedding BLOB, chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents (id)
            )"""
        )
        conn.commit()
        conn.close()

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[str]:
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
            last_period = current_chunk_text.rfind(".")
            last_newline = current_chunk_text.rfind("\n")
            break_point_in_chunk = max(last_period, last_newline)
            if break_point_in_chunk > chunk_size // 2:
                end = start + break_point_in_chunk + 1
            chunks.append(text[start:end])
            next_start = end - overlap
            if next_start <= start:
                start = end
            else:
                start = next_start
        return [c for c in chunks if c.strip()]

    def get_embedding(
        self, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> np.ndarray:
        """Generate embedding for text using Gemini."""
        if (
            not GEMINI_API_KEY
            or GEMINI_API_KEY == "YOUR_GOOGLE_API_KEY_HERE"
            or GEMINI_API_KEY == "AIzaSyCnpefG1lhxQwvkDOoCdbRJUbFu1Icp08w"
        ):  # Check again before use
            app.logger.error(
                "Gemini API key not configured. Cannot generate embeddings."
            )
            return np.zeros(self.embedding_dim, dtype=np.float32)

        if not text or not text.strip():
            app.logger.warning(
                "Attempted to get embedding for empty/whitespace text. Returning zero vector."
            )
            return np.zeros(self.embedding_dim, dtype=np.float32)
        try:
            # Uses the globally configured API key
            result = genai.embed_content(
                model=self.gemini_embedding_model_id,
                content=text,
                task_type=task_type,
            )
            embedding = np.array(result["embedding"], dtype=np.float32)
            return embedding
        except Exception as e:
            app.logger.error(
                f"Error generating Gemini embedding for text (first 50 chars: '{text[:50]}...'): {e}"
            )
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None):
        conn = None  # Initialize conn to None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            metadata_json = json.dumps(metadata or {})
            doc_embedding = self.get_embedding(text, task_type="RETRIEVAL_DOCUMENT")
            doc_embedding_blob = pickle.dumps(doc_embedding)
            cursor.execute(
                "INSERT OR REPLACE INTO documents (id, text, metadata, embedding) VALUES (?, ?, ?, ?)",
                (doc_id, text, metadata_json, doc_embedding_blob),
            )
            chunks = self.chunk_text(text)
            if not chunks:
                app.logger.warning(f"No chunks generated for document {doc_id}.")

            chunk_embeddings_data = []
            for i, chunk_text_content in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_embedding = self.get_embedding(
                    chunk_text_content, task_type="RETRIEVAL_DOCUMENT"
                )
                chunk_embedding_blob = pickle.dumps(chunk_embedding)
                chunk_embeddings_data.append(
                    (chunk_id, doc_id, chunk_text_content, chunk_embedding_blob, i)
                )

            if chunk_embeddings_data:
                cursor.executemany(
                    "INSERT OR REPLACE INTO document_chunks (chunk_id, doc_id, chunk_text, chunk_embedding, chunk_index) VALUES (?, ?, ?, ?, ?)",
                    chunk_embeddings_data,
                )
            conn.commit()
            app.logger.info(
                f"Added/Updated document: {doc_id} with {len(chunks)} chunks using Gemini embeddings."
            )
        except Exception as e:
            app.logger.error(f"Error adding document {doc_id}: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()
            self.load_existing_embeddings()

    def load_existing_embeddings(self):
        app.logger.info("Loading existing embeddings into FAISS index...")
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """SELECT dc.chunk_id, dc.doc_id, dc.chunk_text, dc.chunk_embedding, d.metadata
                   FROM document_chunks dc JOIN documents d ON dc.doc_id = d.id
                   ORDER BY dc.doc_id, dc.chunk_index"""
            )
            all_chunks_data = cursor.fetchall()

            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.doc_ids, self.doc_texts, self.doc_metadatas = [], [], []
            embeddings_to_add_to_faiss = []

            if not all_chunks_data:
                app.logger.info("No chunks found in database to load into FAISS index.")
                return

            for (
                chunk_id,
                doc_id,
                chunk_text,
                embedding_blob,
                metadata_json,
            ) in all_chunks_data:
                try:
                    if embedding_blob is None:
                        app.logger.warning(
                            f"Skipping chunk {chunk_id}: embedding blob is NULL."
                        )
                        continue
                    embedding = pickle.loads(embedding_blob)
                    if not isinstance(embedding, np.ndarray) or embedding.ndim != 1:
                        app.logger.warning(
                            f"Skipping chunk {chunk_id}: invalid embedding format."
                        )
                        continue
                    if embedding.shape[0] != self.embedding_dim:
                        app.logger.warning(
                            f"Skipping chunk {chunk_id}: dim mismatch (expected {self.embedding_dim}, got {embedding.shape[0]})."
                        )
                        continue
                    embeddings_to_add_to_faiss.append(embedding)
                    self.doc_ids.append(chunk_id)
                    self.doc_texts.append(chunk_text)
                    self.doc_metadatas.append(json.loads(metadata_json or "{}"))
                except Exception as inner_e:
                    app.logger.warning(
                        f"Error processing chunk {chunk_id} during load: {inner_e}. Skipping."
                    )

            if embeddings_to_add_to_faiss:
                embeddings_array = np.array(
                    embeddings_to_add_to_faiss, dtype=np.float32
                )
                self.index.add(embeddings_array)
                app.logger.info(
                    f"Successfully loaded {self.index.ntotal} compatible chunk embeddings into FAISS."
                )
            else:
                app.logger.info(
                    "No compatible embeddings found/loaded into FAISS index."
                )
        except Exception as e:
            app.logger.error(f"Error loading embeddings into FAISS: {e}", exc_info=True)
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Reset on error
            self.doc_ids, self.doc_texts, self.doc_metadatas = [], [], []
        finally:
            if conn:
                conn.close()

    def semantic_search(
        self, query: str, n_results: int = 5, score_threshold: float = 0.5
    ) -> Dict:  # Gemini scores are higher
        try:
            if self.index.ntotal == 0:
                app.logger.warning("FAISS index empty. Cannot search.")
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}
            query_embedding = self.get_embedding(query, task_type="RETRIEVAL_QUERY")
            if np.all(query_embedding == 0):
                app.logger.warning("Query embedding failed. Cannot search.")
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            scores, indices = self.index.search(
                query_embedding, k=min(n_results, self.index.ntotal)
            )

            results_data = {"documents": [], "scores": [], "metadatas": [], "ids": []}
            for score, idx in zip(scores[0], indices[0]):
                if (
                    score >= score_threshold
                    and idx != -1
                    and 0 <= idx < len(self.doc_texts)
                ):
                    results_data["documents"].append(self.doc_texts[idx])
                    results_data["scores"].append(float(score))
                    results_data["metadatas"].append(self.doc_metadatas[idx])
                    results_data["ids"].append(self.doc_ids[idx])
            return results_data
        except Exception as e:
            app.logger.error(f"Error in semantic search: {e}", exc_info=True)
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}

    def hybrid_search(self, query: str, n_results: int = 5) -> Dict:
        # Threshold for Gemini (cosine similarity, higher is better)
        semantic_results = self.semantic_search(
            query, n_results * 2, score_threshold=0.4
        )
        combined_results = {}
        query_words_kw = set(re.findall(r"\w+", query.lower()))

        for doc_chunk, score, metadata, chunk_id in zip(
            semantic_results["documents"],
            semantic_results["scores"],
            semantic_results["metadatas"],
            semantic_results["ids"],
        ):
            semantic_component = score * 0.7  # Weight for semantic score
            keyword_score_for_chunk = 0
            if query_words_kw:
                chunk_text_words = set(re.findall(r"\w+", doc_chunk.lower()))
                common_words = query_words_kw.intersection(chunk_text_words)
                if common_words:
                    keyword_score_for_chunk = len(common_words) / len(
                        query_words_kw.union(chunk_text_words)
                    )

            keyword_component = (
                keyword_score_for_chunk * 0.3
            )  # Weight for keyword score
            combined_results[chunk_id] = {
                "document": doc_chunk,
                "semantic_score": score,
                "keyword_score": keyword_score_for_chunk,
                "metadata": metadata,
                "id": chunk_id,
                "final_score": semantic_component + keyword_component,
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
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, text, metadata FROM documents")
            all_docs = cursor.fetchall()
            if not all_docs:
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}
            query_words = set(re.findall(r"\w+", query.lower()))
            if not query_words:
                return {"documents": [], "scores": [], "metadatas": [], "ids": []}
            scored_docs = []
            for doc_id, text, metadata_json in all_docs:
                text_words = set(re.findall(r"\w+", text.lower()))
                common_words = query_words.intersection(text_words)
                if common_words:
                    score = len(common_words) / len(query_words.union(text_words))
                    scored_docs.append(
                        (score, text, json.loads(metadata_json or "{}"), doc_id)
                    )
            scored_docs.sort(reverse=True, key=lambda x: x[0])
            top_docs = scored_docs[:n_results]
            return {
                "documents": [d[1] for d in top_docs],
                "scores": [d[0] for d in top_docs],
                "metadatas": [d[2] for d in top_docs],
                "ids": [d[3] for d in top_docs],
            }
        except Exception as e:
            app.logger.error(f"Error in keyword search: {e}", exc_info=True)
            return {"documents": [], "scores": [], "metadatas": [], "ids": []}
        finally:
            if conn:
                conn.close()

    def generate_answer_with_qa_model(self, question: str, context: str) -> str:
        if not self.qa_pipeline:
            return context[:500] + "..." if len(context) > 500 else context
        try:
            result = self.qa_pipeline(question=question, context=context)
            answer, confidence = result["answer"], result["score"]
            if (
                confidence < 0.15 or len(answer) < 15
            ):  # Slightly higher confidence threshold
                app.logger.info(
                    f"QA model confidence ({confidence:.2f}) or answer length low. Fallback."
                )
                return context[:500] + "..." if len(context) > 500 else context
            return answer
        except Exception as e:
            app.logger.error(f"Error in QA generation: {e}", exc_info=True)
            return context[:500] + "..." if len(context) > 500 else context

    def search(self, query: str, n_results: int = 3, use_hybrid: bool = True) -> Dict:
        if use_hybrid:
            return self.hybrid_search(query, n_results)
        results = self.semantic_search(query, n_results)
        results["semantic_scores"] = results.get("scores", [])
        results["keyword_scores"] = [0.0] * len(results.get("documents", []))
        return results

    def get_all_documents(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, text, metadata FROM documents")
            docs_data = cursor.fetchall()
            return {
                "ids": [d[0] for d in docs_data],
                "documents": [d[1] for d in docs_data],
                "metadatas": [json.loads(d[2] or "{}") for d in docs_data],
            }
        except Exception as e:
            app.logger.error(f"Error getting all documents: {e}", exc_info=True)
            return {"ids": [], "documents": [], "metadatas": []}
        finally:
            if conn:
                conn.close()


# --- Initialize KB globally ---
kb = EnhancedRAGKnowledgeBase()  # Initialization now happens after Gemini key config
app.logger.info("✅ RAG Knowledge Base initialized with Gemini embeddings.")


# --- Greeting and other conversational functions (unchanged) ---
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
        f"{confidence_emoji} Based on my knowledge, here's some information:\n\n{response_text}\n\n"
        f"Is this helpful? (Relevance: {confidence_score:.2f})"
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
            search_results = kb.search(text, n_results=3, use_hybrid=True)
            context_docs = search_results.get("documents", [])
            scores = search_results.get("scores")
            doc_ids = search_results.get("ids")  # Pass along if needed by response fn
            if context_docs:
                app.logger.info(
                    f"Found {len(context_docs)} relevant chunks. Top score: {scores[0] if scores else 'N/A'}"
                )
            else:
                app.logger.info(f"No relevant documents found for query: '{text}'")
            response = enhanced_generate_response(text, context_docs, scores, doc_ids)

        if SLACK_BOT_TOKEN:  # Ensure token is available before trying to post
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
@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data["challenge"]})
    if data.get("type") == "event_callback":
        event = data.get("event")
        if not event:
            return jsonify({"status": "ok", "message": "No event payload"}), 200
        if event.get("subtype") or event.get("bot_id"):
            return jsonify({"status": "ok", "message": "Ignored bot message/subtype"})
        if event.get("type") == "app_mention" or (
            event.get("type") == "message" and event.get("channel_type") == "im"
        ):
            handle_message(event)
    return jsonify({"status": "ok"})


@app.route("/add_knowledge", methods=["POST"])
def add_knowledge():
    try:
        data = request.json
        if not data or not data.get("id") or not data.get("text"):
            return jsonify({"error": "Missing id or text"}), 400
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
                        "text_preview": docs_summary["documents"][i][:200] + "...",
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
    return jsonify(
        {
            "status": "healthy",
            "rag_enabled": True,
            "embedding_model": kb.model_name,
            "vector_dimension": kb.embedding_dim,
            "total_documents_in_db": len(kb.get_all_documents()["ids"]),
            "total_chunks_in_faiss": kb.index.ntotal if kb.index else 0,
            "qa_model_loaded": kb.qa_pipeline is not None,
            "gemini_api_configured": bool(
                GEMINI_API_KEY
                and GEMINI_API_KEY
                not in [
                    "YOUR_GOOGLE_API_KEY_HERE",
                    "AIzaSyCnpefG1lhxQwvkDOoCdbRJUbFu1Icp08w",
                ]
            ),
            "slack_bot_token_configured": bool(SLACK_BOT_TOKEN),
        }
    )


@app.route("/")
def home():
    total_docs = len(kb.get_all_documents()["ids"])
    total_chunks = kb.index.ntotal if kb.index else "N/A"
    qa_status = "✅ Loaded (DistilBERT)" if kb.qa_pipeline else "❌ Not loaded"
    gemini_ok = bool(
        GEMINI_API_KEY
        and GEMINI_API_KEY
        not in ["YOUR_GOOGLE_API_KEY_HERE", "AIzaSyCnpefG1lhxQwvkDOoCdbRJUbFu1Icp08w"]
    )

    return f"""
    <h1>🚀 RAG Slack AI Bot (Gemini Embeddings)</h1>
    <p>Status: {'✅ Running' if gemini_ok else '⚠️ Check Gemini API Key Configuration!'}</p>
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
        <li><strong>Gemini API Key:</strong> {'✅ Configured' if gemini_ok else '❌ NOT CONFIGURED OR PLACEHOLDER!'}</li>
        <li><strong>Slack Token:</strong> {'✅ Configured' if SLACK_BOT_TOKEN else '❌ NOT CONFIGURED!'}</li>
    </ul>
    <p><em>See code/logs for API details. Remember to set GOOGLE_API_KEY and SLACK_BOT_TOKEN in your .env file.</em></p>
    """


# --- Sample Data Initialization ---
def initialize_sample_data():
    sample_docs = [
        {
            "id": "support_gemini",
            "text": "Our support team, powered by Gemini insights, is available M-F, 9-5 EST. Email support@example.com.",
            "metadata": {"category": "support"},
        },
        {
            "id": "password_gemini",
            "text": "To reset your password with Gemini security, go to login, click 'Forgot Password', and follow email steps.",
            "metadata": {"category": "account"},
        },
        {
            "id": "billing_gemini",
            "text": "For Gemini-enhanced billing, contact billing@example.com. Invoices are monthly.",
            "metadata": {"category": "billing"},
        },
    ]
    conn = sqlite3.connect(kb.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    conn.close()

    if doc_count == 0 and gemini_ok:  # Only add if DB empty AND Gemini key is okay
        app.logger.info(
            "🔄 Initializing RAG with sample data (using Gemini embeddings)..."
        )
        for doc in sample_docs:
            kb.add_document(
                doc["id"], doc["text"], doc.get("metadata", {})
            )  # This calls load_existing_embeddings internally
        app.logger.info(
            f"✅ Added {len(sample_docs)} sample docs. FAISS chunks: {kb.index.ntotal if kb.index else 'N/A'}"
        )
    elif doc_count > 0:
        app.logger.info(
            f"📚 DB not empty ({doc_count} docs). Skipping sample data. FAISS chunks: {kb.index.ntotal if kb.index else 'N/A'}"
        )
        if (kb.index is None or kb.index.ntotal == 0) and gemini_ok:
            app.logger.info(
                "Attempting to load existing embeddings as FAISS seems empty..."
            )
            kb.load_existing_embeddings()
    elif not gemini_ok:
        app.logger.warning(
            "Skipping sample data initialization because Gemini API key is not properly configured."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    app.logger.info("🚀 Starting RAG-Powered Slack AI Bot with Gemini Embeddings")

    gemini_ok_startup = bool(
        GEMINI_API_KEY
        and GEMINI_API_KEY
        not in ["YOUR_GOOGLE_API_KEY_HERE", "AIzaSyCnpefG1lhxQwvkDOoCdbRJUbFu1Icp08w"]
    )
    slack_ok_startup = bool(SLACK_BOT_TOKEN)

    if not gemini_ok_startup:
        app.logger.critical(
            "FATAL: Gemini API Key is not configured correctly. Embedding generation will fail. Please set GOOGLE_API_KEY."
        )
    if not slack_ok_startup:
        app.logger.critical(
            "FATAL: SLACK_BOT_TOKEN is not configured. Bot cannot connect to Slack. Please set SLACK_BOT_TOKEN."
        )

    # Initialize sample data only if critical configs are okay
    if gemini_ok_startup:  # Sample data depends on embeddings
        initialize_sample_data()
    else:
        app.logger.warning(
            "Skipping sample data initialization due to missing Gemini API key."
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
        f"   • QA Model: {'DistilBERT (loaded)' if kb.qa_pipeline else 'Not loaded'}"
    )
    app.logger.info(
        f"🌐 Web interface: http://localhost:{os.environ.get('PORT', 4000)}"
    )
    app.logger.info("=" * 80 + "\n")

    if not gemini_ok_startup or not slack_ok_startup:
        app.logger.warning(
            "CRITICAL CONFIGURATION ISSUES DETECTED. BOT MAY NOT FUNCTION CORRECTLY."
        )

    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
