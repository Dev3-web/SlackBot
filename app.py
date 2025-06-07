import os
import json
import re
import random
import io
import pickle
import logging
import threading # Added for async processing
from datetime import datetime
from typing import List, Dict, Optional

# Flask and Slack
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Data & AI
import numpy as np
import faiss
import openai
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2

# Database
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import gridfs
from bson import ObjectId

# YouTube and Google Drive Integrations
from youtube_transcript_api import YouTubeTranscriptApi
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "rag_summarizer_kb")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_BOT_USER_ID = os.environ.get("SLACK_BOT_USER_ID")
GOOGLE_API_SCOPES = os.environ.get("GOOGLE_API_SCOPES", "https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/documents.readonly").split()
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

# --- Initialization of Clients ---
if not GEMINI_API_KEY:
    app.logger.error("CRITICAL: GOOGLE_API_KEY is not set.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

if not SLACK_BOT_TOKEN:
    app.logger.error("CRITICAL: SLACK_BOT_TOKEN not found.")
slack_client = WebClient(token=SLACK_BOT_TOKEN)

# --- Google Drive Authentication ---
# (Keep the get_google_creds function exactly as it was)
def get_google_creds():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, GOOGLE_API_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                app.logger.error(f"Failed to refresh Google token: {e}. Re-authenticating.")
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, GOOGLE_API_SCOPES)
                creds = flow.run_local_server(port=0)
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                app.logger.critical(f"FATAL: {CREDENTIALS_FILE} not found.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, GOOGLE_API_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return creds

# --- Utility and Processor Functions ---
# (Keep extract_text_from_pdf, process_youtube_url, and the smart process_google_drive_url functions as they were)
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        app.logger.error(f"Error extracting text from PDF: {e}")
        raise
    return text

def process_youtube_url(video_url: str):
    try:
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
        if not video_id_match: return None
        video_id = video_id_match.group(1)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([item['text'] for item in transcript_list])
        if not transcript.strip(): return None
        return {"doc_id": f"yt_{video_id}", "text": transcript, "metadata": {"source": "youtube", "url": f"https://www.youtube.com/watch?v={video_id}"}}
    except Exception as e:
        app.logger.error(f"Failed to get YouTube transcript for {video_url}: {e}")
        return None

def process_google_drive_url(drive_url: str):
    try:
        file_id_match = re.search(r"/d/([a-zA-Z0-9_-]{25,})", drive_url)
        if not file_id_match: return None
        file_id = file_id_match.group(1)
        drive_service, docs_service, file_metadata = None, None, None
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                public_drive_service = build('drive', 'v3', developerKey=api_key)
                file_metadata = public_drive_service.files().get(fileId=file_id, fields='name, mimeType').execute()
                drive_service = public_drive_service
                if file_metadata.get('mimeType') == 'application/vnd.google-apps.document':
                    docs_service = build('docs', 'v1', developerKey=api_key)
            except HttpError as e:
                if e.resp.status in [401, 403]: file_metadata = None
                else: raise e
        if file_metadata is None:
            creds = get_google_creds()
            if not creds: return None
            drive_service = build('drive', 'v3', credentials=creds)
            file_metadata = drive_service.files().get(fileId=file_id, fields='name, mimeType').execute()
            if file_metadata.get('mimeType') == 'application/vnd.google-apps.document':
                docs_service = build('docs', 'v1', credentials=creds)
        file_name, mime_type, text_content = file_metadata.get('name'), file_metadata.get('mimeType'), ""
        if mime_type == 'application/vnd.google-apps.document':
            doc = docs_service.documents().get(documentId=file_id).execute()
            text_content = "".join(elem.get('paragraph').get('elements')[0].get('textRun').get('content') for elem in doc.get('body').get('content') if 'paragraph' in elem and 'elements' in elem.get('paragraph') and elem.get('paragraph').get('elements')[0].get('textRun'))
        elif mime_type == 'application/pdf':
            request_obj = drive_service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request_obj)
            done = False
            while not done: _, done = downloader.next_chunk()
            fh.seek(0)
            text_content = extract_text_from_pdf(fh)
        else:
            return None
        if not text_content.strip(): return None
        return {"doc_id": f"gd_{file_id}", "text": text_content, "metadata": {"source": "google_drive", "url": drive_url, "filename": file_name}}
    except Exception as e:
        app.logger.error(f"Error processing Google Drive URL {drive_url}: {e}", exc_info=True)
        return None

# --- EnhancedRAGKnowledgeBase Class ---
# (Keep the entire EnhancedRAGKnowledgeBase class exactly as it was)
class EnhancedRAGKnowledgeBase:
    def __init__(self):
        self.mongodb_connected = False
        self.mongo_client = None
        self.db = None
        self.documents_collection = None
        self.chunks_collection = None
        self.gemini_embedding_model_id = "models/text-embedding-004"
        self.gemini_generation_model_id = "gemini-pro"
        self.embedding_dim = 768
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.doc_ids, self.doc_texts, self.doc_metadatas = [], [], []
        self.gemini_generation_available = False
        try:
            self.mongo_client = MongoClient(MONGODB_URI)
            self.db = self.mongo_client[MONGODB_DB_NAME]
            self.documents_collection = self.db.documents
            self.chunks_collection = self.db.document_chunks
            self.documents_collection.create_index("doc_id", unique=True)
            self.chunks_collection.create_index("doc_id")
            self.mongo_client.admin.command('ping')
            self.mongodb_connected = True
            app.logger.info("✅ Connected to MongoDB")
        except Exception as e:
            app.logger.error(f"❌ Failed to connect to MongoDB: {e}")
        if GEMINI_API_KEY:
            try:
                genai.GenerativeModel(self.gemini_generation_model_id).generate_content("test").resolve()
                self.gemini_generation_available = True
            except Exception:
                app.logger.warning("Gemini generation model not available.")
        if self.mongodb_connected and GEMINI_API_KEY:
            self.load_existing_embeddings()
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50):
        if not text: return []
        chunks, start = [], 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap if end - overlap > start else end
        return [c for c in chunks if c.strip()]
    def get_embedding(self, text: str, task_type: str):
        if not text or not GEMINI_API_KEY: return np.zeros(self.embedding_dim)
        try:
            return np.array(genai.embed_content(model=self.gemini_embedding_model_id, content=text, task_type=task_type)["embedding"])
        except Exception: return np.zeros(self.embedding_dim)
    def add_document(self, doc_id: str, text: str, metadata: dict):
        if not self.mongodb_connected: return
        self.documents_collection.replace_one({"doc_id": doc_id}, {"doc_id": doc_id, "text": text, "metadata": metadata}, upsert=True)
        self.chunks_collection.delete_many({"doc_id": doc_id})
        chunks = self.chunk_text(text)
        chunk_docs = [{"chunk_id": f"{doc_id}_chunk_{i}", "doc_id": doc_id, "chunk_text": chunk, "chunk_embedding": self.get_embedding(chunk, "RETRIEVAL_DOCUMENT").tolist()} for i, chunk in enumerate(chunks)]
        if chunk_docs: self.chunks_collection.insert_many(chunk_docs)
        self.load_existing_embeddings()
    def load_existing_embeddings(self):
        if not self.mongodb_connected: return
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        all_chunks = list(self.chunks_collection.find({}))
        if not all_chunks: return
        embeddings = np.array([d['chunk_embedding'] for d in all_chunks]).astype('float32')
        self.index.add(embeddings)
        self.doc_ids = [d['chunk_id'] for d in all_chunks]
        self.doc_texts = [d['chunk_text'] for d in all_chunks]
        app.logger.info(f"Loaded {self.index.ntotal} embeddings.")
    def semantic_search(self, query: str, n_results: int, score_threshold: float):
        if self.index.ntotal == 0: return {"documents": [], "scores": [], "ids": []}
        query_embedding = self.get_embedding(query, "RETRIEVAL_QUERY").reshape(1, -1)
        scores, indices = self.index.search(query_embedding, n_results)
        results = {"documents": [], "scores": [], "ids": []}
        for score, idx in zip(scores[0], indices[0]):
            if score >= score_threshold:
                results["documents"].append(self.doc_texts[idx])
                results["scores"].append(float(score))
                results["ids"].append(self.doc_ids[idx])
        return results
    def generate_response_with_llm(self, question: str, context: str, task: str):
        if not self.gemini_generation_available: return "AI generation unavailable."
        model = genai.GenerativeModel(self.gemini_generation_model_id)
        prompt = f"Summarize this content with bullet points:\n\n{context}" if task == "summarize" else f"Answer based on context:\n\nContext: {context}\n\nQuestion: {question}"
        return model.generate_content(prompt).text
    def get_database_stats(self):
        if not self.mongodb_connected: return {}
        return {"total_documents": self.documents_collection.count_documents({}), "total_chunks": self.chunks_collection.count_documents({}), "faiss_index_size": self.index.ntotal}


# --- Initialize KB and Thread Context DB ---
kb = EnhancedRAGKnowledgeBase() 
thread_context_db = MongoClient(MONGODB_URI)[MONGODB_DB_NAME].thread_context
thread_context_db.create_index("thread_ts", unique=True)

# --- Slack Event Handling (New Asynchronous Structure) ---
def process_event_async(event):
    channel = event["channel"]
    text = event.get("text", "")
    thread_ts = event.get("thread_ts") or event.get("ts")

    if SLACK_BOT_USER_ID:
        text = text.replace(f"<@{SLACK_BOT_USER_ID}>", "").strip()
    if not text: return

    yt_regex = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([\w-]+)"
    gdrive_regex = r"(https?://)?(www\.)?drive\.google\.com/file/d/([\w-]+)"
    yt_match = re.search(yt_regex, text)
    gdrive_match = re.search(gdrive_regex, text)

    try:
        if yt_match or gdrive_match:
            url = yt_match.group(0) if yt_match else gdrive_match.group(0)
            slack_client.chat_postMessage(channel=channel, text=f"Got it! Processing your link:\n`{url}`...", thread_ts=thread_ts)
            
            source_type = "youtube" if yt_match else "google_drive"
            content_data = process_youtube_url(url) if yt_match else process_google_drive_url(url)
            
            if not content_data or not content_data.get("text"):
                error_message = "Sorry, I couldn't get a transcript for that YouTube video. This usually means the video has captions disabled." if source_type == "youtube" else "Sorry, I couldn't access that Google Drive file. It might be private or an unsupported type."
                slack_client.chat_postMessage(channel=channel, text=error_message, thread_ts=thread_ts)
                return

            kb.add_document(content_data["doc_id"], content_data["text"], content_data["metadata"])
            thread_context_db.replace_one({"thread_ts": thread_ts}, {"thread_ts": thread_ts, "doc_id": content_data["doc_id"]}, upsert=True)
            summary = kb.generate_response_with_llm(question="", context=content_data["text"], task="summarize")
            response = f"Here is a summary for `{content_data['metadata'].get('url')}`:\n\n{summary}\n\nFeel free to ask specific questions about this document in this thread!"
            slack_client.chat_postMessage(channel=channel, text=response, thread_ts=thread_ts)
        
        elif thread_ts:
            context_info = thread_context_db.find_one({"thread_ts": thread_ts})
            if context_info:
                doc_id = context_info["doc_id"]
                search_results = kb.semantic_search(text, n_results=3, score_threshold=0.5)
                relevant_chunks = [doc for i, doc in enumerate(search_results['documents']) if search_results['ids'][i].startswith(doc_id)]
                if not relevant_chunks:
                    response = "I couldn't find a specific answer in the document for that."
                else:
                    response = kb.generate_response_with_llm(question=text, context="\n\n---\n\n".join(relevant_chunks), task="qa")
                slack_client.chat_postMessage(channel=channel, text=response, thread_ts=thread_ts)
        else:
            response = "Hello! I can summarize YouTube videos and Google Docs. Please paste a link to get started. 😊"
            slack_client.chat_postMessage(channel=channel, text=response, thread_ts=thread_ts)
    except Exception as e:
        app.logger.error(f"Error in background processing thread: {e}", exc_info=True)
        try:
            slack_client.chat_postMessage(channel=channel, text="Oops! Something went wrong on my end.", thread_ts=thread_ts)
        except Exception as slack_err:
            app.logger.error(f"Failed to send error message to slack: {slack_err}")

# --- Flask Routes ---
@app.route("/slack/events", methods=["POST"]) 
def slack_events_route():
    data = request.json
    if data.get("type") == "url_verification": return jsonify({"challenge": data["challenge"]})
    if data.get("type") == "event_callback":
        event = data.get("event", {})
        if not (event.get("subtype") == "bot_message" or event.get("bot_id")):
            thread = threading.Thread(target=process_event_async, args=(event,))
            thread.start()
    return jsonify({"status": "ok"})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "mongodb_connected": kb.mongodb_connected,
        "gemini_generation_available": kb.gemini_generation_available,
        "google_credentials_found": os.path.exists(CREDENTIALS_FILE),
        **kb.get_database_stats()
    })

@app.route("/")
def home():
    stats = kb.get_database_stats()
    return f"""<h1>🚀 RAG Summarizer Bot</h1>
    <h2>🔧 Status:</h2>
    <ul>
        <li>MongoDB: {'✅ Connected' if kb.mongodb_connected else '❌ NOT CONNECTED'}</li>
        <li>Gemini Generation: {'✅ Available' if kb.gemini_generation_available else '❌ UNAVAILABLE'}</li>
        <li>Google Credentials (`credentials.json`): {'✅ Found' if os.path.exists(CREDENTIALS_FILE) else '❌ NOT FOUND'}</li>
    </ul>
    <h2>📊 DB Stats:</h2>
    <ul>
        <li>Total Documents: {stats.get('total_documents', 'N/A')}</li>
        <li>Total Chunks: {stats.get('total_chunks', 'N/A')}</li>
        <li>Chunks in FAISS: {stats.get('faiss_index_size', 'N/A')}</li>
    </ul>"""

# --- Main Application ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    app.logger.info("🚀 Starting RAG-Powered Summarizer Bot")
    if os.path.exists(CREDENTIALS_FILE):
        get_google_creds()
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)