import os
import json
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import openai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import logging

# Initialize Flask app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Initialize Slack client
slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

# Initialize OpenAI (or use any other LLM)
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize embedding model and vector database
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./knowledge_base")

# Get or create collection for knowledge base
try:
    collection = chroma_client.get_collection("knowledge_base")
except:
    collection = chroma_client.create_collection(
        name="knowledge_base",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        ),
    )


class KnowledgeBase:
    def __init__(self, collection):
        self.collection = collection

    def add_document(self, doc_id, text, metadata=None):
        """Add a document to the knowledge base"""
        self.collection.add(documents=[text], ids=[doc_id], metadatas=[metadata or {}])

    def search(self, query, n_results=3):
        """Search for relevant documents"""
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results


kb = KnowledgeBase(collection)


def generate_response(question, context_docs):
    """Generate AI response using context from knowledge base"""
    context = "\n\n".join([doc for doc in context_docs])

    prompt = f"""You are a helpful AI assistant. Answer the following question based on the provided context. 
If the context doesn't contain enough information to answer the question, say so politely.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that answers questions based on provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        app.logger.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error while generating a response. Please try again."


def handle_message(event):
    """Process incoming message and generate response"""
    try:
        # Extract message details
        channel = event["channel"]
        user = event["user"]
        text = event["text"]

        # Remove bot mention from text
        text = text.replace(f"<@{os.environ.get('SLACK_BOT_USER_ID')}>", "").strip()

        if not text:
            return

        # Search knowledge base
        search_results = kb.search(text, n_results=3)

        if search_results["documents"] and search_results["documents"][0]:
            context_docs = search_results["documents"][0]
            response = generate_response(text, context_docs)
        else:
            response = "I couldn't find relevant information in my knowledge base to answer your question. Could you please rephrase or ask about something else?"

        # Send response to Slack
        slack_client.chat_postMessage(
            channel=channel,
            text=response,
            thread_ts=event.get("ts"),  # Reply in thread if mentioned in a thread
        )

    except Exception as e:
        app.logger.error(f"Error handling message: {e}")
        # Send error message to user
        try:
            slack_client.chat_postMessage(
                channel=event["channel"],
                text="Sorry, I encountered an error processing your request. Please try again.",
                thread_ts=event.get("ts"),
            )
        except:
            pass


@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle Slack events"""
    data = request.json

    # Handle URL verification challenge
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data["challenge"]})

    # Handle events
    if data.get("type") == "event_callback":
        event = data["event"]

        # Ignore bot messages and message changes
        if event.get("subtype") or event.get("bot_id"):
            return jsonify({"status": "ok"})

        # Handle mentions and DMs
        if event["type"] in ["app_mention", "message"]:
            handle_message(event)

    return jsonify({"status": "ok"})


@app.route("/add_knowledge", methods=["POST"])
def add_knowledge():
    """Endpoint to add documents to knowledge base"""
    try:
        data = request.json
        doc_id = data.get("id")
        text = data.get("text")
        metadata = data.get("metadata", {})

        if not doc_id or not text:
            return jsonify({"error": "Missing id or text"}), 400

        kb.add_document(doc_id, text, metadata)
        return jsonify(
            {"status": "success", "message": "Document added to knowledge base"}
        )

    except Exception as e:
        app.logger.error(f"Error adding knowledge: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    # Load some sample knowledge base data
    sample_docs = [
        {
            "id": "doc1",
            "text": "Our company offers technical support Monday through Friday, 9 AM to 5 PM EST. You can reach support by emailing support@company.com or calling 1-800-123-4567.",
            "metadata": {"category": "support"},
        },
        {
            "id": "doc2",
            "text": 'To reset your password, go to the login page and click "Forgot Password". Enter your email address and follow the instructions sent to your email.',
            "metadata": {"category": "account"},
        },
    ]

    # Add sample documents to knowledge base
    for doc in sample_docs:
        kb.add_document(doc["id"], doc["text"], doc["metadata"])

    app.run(host="0.0.0.0", port=3000, debug=True)
