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

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Initialize Slack client
slack_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))


class SimpleKnowledgeBase:
    def __init__(self, db_path="knowledge_base.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    def add_document(self, doc_id, text, metadata=None):
        """Add a document to the knowledge base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            metadata_json = json.dumps(metadata or {})

            cursor.execute(
                """
                INSERT OR REPLACE INTO documents (id, text, metadata)
                VALUES (?, ?, ?)
            """,
                (doc_id, text, metadata_json),
            )

            conn.commit()
            conn.close()
            print(f"Added document: {doc_id}")

        except Exception as e:
            print(f"Error adding document {doc_id}: {e}")

    def search(self, query, n_results=3):
        """Search for relevant documents using simple text matching"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all documents
            cursor.execute("SELECT id, text, metadata FROM documents")
            all_docs = cursor.fetchall()
            conn.close()

            if not all_docs:
                return {"documents": [[]], "scores": [[]], "metadatas": [[]]}

            # Simple scoring based on keyword matching
            query_words = set(re.findall(r"\w+", query.lower()))
            scored_docs = []

            for doc_id, text, metadata in all_docs:
                text_words = set(re.findall(r"\w+", text.lower()))

                # Calculate simple similarity score
                common_words = query_words.intersection(text_words)
                if common_words:
                    score = len(common_words) / len(query_words.union(text_words))
                    scored_docs.append((score, text, json.loads(metadata)))

            # Sort by score and return top results
            scored_docs.sort(reverse=True, key=lambda x: x[0])
            top_docs = scored_docs[:n_results]

            documents = [doc[1] for doc in top_docs]
            scores = [doc[0] for doc in top_docs]
            metadatas = [doc[2] for doc in top_docs]

            return {
                "documents": [documents],
                "scores": [scores],
                "metadatas": [metadatas],
            }

        except Exception as e:
            print(f"Error searching: {e}")
            return {"documents": [[]], "scores": [[]], "metadatas": [[]]}

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
                "metadatas": [json.loads(doc[2]) for doc in docs],
            }

        except Exception as e:
            print(f"Error getting documents: {e}")
            return {"ids": [], "documents": [], "metadatas": []}


# Initialize knowledge base
kb = SimpleKnowledgeBase()


def generate_response(question, context_docs):
    """Generate response using simple template-based approach"""
    if not context_docs or not any(context_docs):
        return "I couldn't find relevant information in my knowledge base to answer your question. Could you please rephrase or ask about something else?"

    question_lower = question.lower()

    # Handle greetings
    if any(
        word in question_lower
        for word in ["hello", "hi", "hey", "good morning", "good afternoon"]
    ):
        return "Hello! How can I help you today? I can answer questions based on my knowledge base."

    # Handle thanks
    if any(word in question_lower for word in ["thanks", "thank you", "thx"]):
        return "You're welcome! Is there anything else I can help you with?"

    # Handle help requests
    if any(word in question_lower for word in ["help", "what can you do"]):
        return "I can help answer questions based on my knowledge base. Try asking about support, billing, passwords, or product features!"

    # Return the most relevant context
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
            response = generate_response(text, context_docs)
        else:
            response = "I couldn't find relevant information in my knowledge base to answer your question. Could you please rephrase or ask about something else?"

        # Send response to Slack
        slack_client.chat_postMessage(
            channel=channel, text=response, thread_ts=event.get("ts")
        )

        print(f"Sent response: {response[:100]}...")

    except Exception as e:
        app.logger.error(f"Error handling message: {e}")
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
                        "score": results["scores"][0][i] if results["scores"][0] else 0,
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
    <h1>Slack AI Bot - Simple Local Version</h1>
    <p>Bot is running! Here are the available endpoints:</p>
    <ul>
        <li><strong>POST /slack/events</strong> - Slack events webhook</li>
        <li><strong>POST /add_knowledge</strong> - Add documents to knowledge base</li>
        <li><strong>GET /list_knowledge</strong> - View all knowledge base documents</li>
        <li><strong>POST /test_search</strong> - Test search functionality</li>
        <li><strong>GET /health</strong> - Health check</li>
    </ul>
    <p>Total documents in knowledge base: <strong>{}</strong></p>
    <p><em>This version uses simple text matching instead of embeddings</em></p>
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
    print("Starting Simple Slack AI Bot (Local Development)")
    print("Using SQLite database and simple text matching")

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
