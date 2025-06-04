import os
import json
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import logging
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Initialize Slack client
slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))

# Initialize embedding model (this will download the model locally)
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")

# Initialize ChromaDB (local persistent storage)
chroma_client = chromadb.PersistentClient(path="./local_knowledge_base")

# Get or create collection for knowledge base
try:
    collection = chroma_client.get_collection("knowledge_base")
    print("Loaded existing knowledge base")
except:
    collection = chroma_client.create_collection(
        name="knowledge_base",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        ),
    )
    print("Created new knowledge base")


class LocalKnowledgeBase:
    def __init__(self, collection):
        self.collection = collection

    def add_document(self, doc_id, text, metadata=None):
        """Add a document to the knowledge base"""
        try:
            self.collection.add(
                documents=[text], ids=[doc_id], metadatas=[metadata or {}]
            )
            print(f"Added document: {doc_id}")
        except Exception as e:
            print(f"Error adding document {doc_id}: {e}")

    def search(self, query, n_results=3):
        """Search for relevant documents"""
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}

    def get_all_documents(self):
        """Get all documents in the knowledge base"""
        try:
            return self.collection.get()
        except Exception as e:
            print(f"Error getting documents: {e}")
            return {"documents": [], "ids": [], "metadatas": []}


kb = LocalKnowledgeBase(collection)


def simple_ai_response(question, context_docs):
    """Generate response using simple template-based approach (no external API needed)"""
    if not context_docs or not any(context_docs):
        return "I couldn't find relevant information in my knowledge base to answer your question. Could you please rephrase or ask about something else?"

    # Simple template-based response
    context = "\n\n".join([doc for doc in context_docs if doc])

    # Basic keyword matching and response generation
    question_lower = question.lower()

    if any(word in question_lower for word in ["hello", "hi", "hey"]):
        return "Hello! How can I help you today? I can answer questions based on my knowledge base."

    if any(word in question_lower for word in ["thanks", "thank you", "thx"]):
        return "You're welcome! Is there anything else I can help you with?"

    # Return the most relevant context with a simple introduction
    best_match = context_docs[0] if context_docs else ""
    if len(best_match) > 500:
        best_match = best_match[:500] + "..."

    return f"Based on my knowledge base:\n\n{best_match}\n\nIs this helpful? Let me know if you need more specific information!"


def handle_message(event):
    """Process incoming message and generate response"""
    try:
        # Extract message details
        channel = event["channel"]
        user = event["user"]
        text = event["text"]

        # Remove bot mention from text
        bot_user_id = os.environ.get("SLACK_BOT_USER_ID")
        if bot_user_id:
            text = text.replace(f"<@{bot_user_id}>", "").strip()

        if not text:
            return

        print(f"Processing question: {text}")

        # Search knowledge base
        search_results = kb.search(text, n_results=3)

        if search_results["documents"] and search_results["documents"][0]:
            context_docs = search_results["documents"][0]
            print(f"Found {len(context_docs)} relevant documents")
            response = simple_ai_response(text, context_docs)
        else:
            response = "I couldn't find relevant information in my knowledge base to answer your question. Could you please rephrase or ask about something else?"

        # Send response to Slack
        slack_client.chat_postMessage(
            channel=channel,
            text=response,
            thread_ts=event.get("ts"),  # Reply in thread if mentioned in a thread
        )

        print(f"Sent response: {response[:100]}...")

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


@app.route("/list_knowledge", methods=["GET"])
def list_knowledge():
    """List all documents in knowledge base"""
    try:
        docs = kb.get_all_documents()
        return jsonify(
            {
                "total_documents": len(docs["ids"]),
                "documents": [
                    {
                        "id": docs["ids"][i],
                        "text": (
                            docs["documents"][i][:200] + "..."
                            if len(docs["documents"][i]) > 200
                            else docs["documents"][i]
                        ),
                        "metadata": (
                            docs["metadatas"][i] if i < len(docs["metadatas"]) else {}
                        ),
                    }
                    for i in range(len(docs["ids"]))
                ],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/test_search", methods=["POST"])
def test_search():
    """Test search functionality"""
    try:
        data = request.json
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Missing query"}), 400

        results = kb.search(query, n_results=3)

        return jsonify(
            {
                "query": query,
                "results": [
                    {
                        "document": (
                            results["documents"][0][i]
                            if results["documents"][0]
                            else ""
                        ),
                        "distance": (
                            results["distances"][0][i] if results["distances"][0] else 0
                        ),
                        "metadata": (
                            results["metadatas"][0][i]
                            if results["metadatas"][0]
                            else {}
                        ),
                    }
                    for i in range(
                        len(results["documents"][0]) if results["documents"][0] else 0
                    )
                ],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {"status": "healthy", "knowledge_base_docs": len(kb.get_all_documents()["ids"])}
    )


@app.route("/")
def home():
    """Simple home page for testing"""
    return """
    <h1>Slack AI Bot - Local Development</h1>
    <p>Bot is running! Here are the available endpoints:</p>
    <ul>
        <li><strong>POST /slack/events</strong> - Slack events webhook</li>
        <li><strong>POST /add_knowledge</strong> - Add documents to knowledge base</li>
        <li><strong>GET /list_knowledge</strong> - View all knowledge base documents</li>
        <li><strong>POST /test_search</strong> - Test search functionality</li>
        <li><strong>GET /health</strong> - Health check</li>
    </ul>
    <p>Total documents in knowledge base: <strong>{}</strong></p>
    """.format(
        len(kb.get_all_documents()["ids"])
    )


def initialize_sample_data():
    """Initialize with sample knowledge base data"""
    sample_docs = [
        {
            "id": "support_hours",
            "text": "Our technical support team is available Monday through Friday, 9 AM to 5 PM EST. You can reach us by emailing support@company.com or calling 1-800-123-4567. For urgent issues outside business hours, please email emergency@company.com.",
            "metadata": {"category": "support", "type": "hours"},
        },
        {
            "id": "password_reset",
            "text": 'To reset your password: 1) Go to the login page 2) Click "Forgot Password" 3) Enter your email address 4) Check your email for reset instructions 5) Follow the link in the email to create a new password. The reset link expires after 24 hours.',
            "metadata": {"category": "account", "type": "password"},
        },
        {
            "id": "billing_info",
            "text": "Billing questions can be resolved by contacting our billing department at billing@company.com or calling 1-800-123-4568. Invoices are sent monthly, and payments are due within 30 days. We accept credit cards, bank transfers, and checks.",
            "metadata": {"category": "billing", "type": "general"},
        },
        {
            "id": "product_features",
            "text": "Our software includes advanced analytics, real-time reporting, team collaboration tools, API access, custom integrations, and 24/7 monitoring. Premium plans also include priority support and dedicated account management.",
            "metadata": {"category": "product", "type": "features"},
        },
    ]

    # Check if knowledge base is empty
    existing_docs = kb.get_all_documents()
    if len(existing_docs["ids"]) == 0:
        print("Initializing knowledge base with sample data...")
        for doc in sample_docs:
            kb.add_document(doc["id"], doc["text"], doc["metadata"])
        print(f"Added {len(sample_docs)} sample documents to knowledge base")
    else:
        print(f"Knowledge base already contains {len(existing_docs['ids'])} documents")


if __name__ == "__main__":
    print("Starting Slack AI Bot (Local Development)")

    # Initialize sample data
    initialize_sample_data()

    print("\n" + "=" * 50)
    print("🚀 Bot is ready!")
    print("📝 Add documents via: POST /add_knowledge")
    print("🔍 Test search via: POST /test_search")
    print("📋 View docs via: GET /list_knowledge")
    print("🌐 Web interface: http://localhost:3000")
    print("=" * 50 + "\n")

    # Run Flask app
    app.run(host="0.0.0.0", port=3000, debug=True)
