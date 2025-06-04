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

    # Common greetings
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
        "evening",
        "good night",
        "goodnight",
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

    # Check for exact matches or if text starts with greeting
    for greeting in greetings:
        if (
            text_lower == greeting
            or text_lower.startswith(greeting + " ")
            or text_lower.startswith(greeting + ",")
        ):
            return True

    # Check for greeting patterns with punctuation
    greeting_patterns = [
        r"^(hi|hello|hey)[\s!,.]*$",
        r"^(good\s*(morning|afternoon|evening|night))[\s!,.]*$",
        r"^(how\s*(are\s*you|\'s\s*it\s*going))[\s!,.?]*$",
        r"^\w*(morning|afternoon|evening)\w*[\s!,.]*$",
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

    # Sometimes add a friendly emoji or extra touch
    extras = [
        "😊",
        "👋",
        "🤖",
        "✨",
        "",
        "",
        "",  # More empty strings to make extras less frequent
    ]
    extra = random.choice(extras)

    if extra:
        return f"{time_greeting} {follow_up} {extra}"
    else:
        return f"{time_greeting} {follow_up}"


def detect_thanks(text):
    """Enhanced thanks detection"""
    text_lower = text.lower().strip()

    thanks_patterns = [
        "thank you",
        "thanks",
        "thx",
        "ty",
        "thank u",
        "thankyou",
        "much appreciated",
        "appreciate it",
        "appreciate that",
        "grateful",
        "cheers",
        "awesome",
        "perfect",
        "great",
        "that helps",
        "that's helpful",
        "very helpful",
        "exactly what i needed",
        "that's perfect",
    ]

    for pattern in thanks_patterns:
        if pattern in text_lower:
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
    ]

    return random.choice(responses)


def detect_help_request(text):
    """Enhanced help request detection"""
    text_lower = text.lower().strip()

    help_patterns = [
        "help",
        "what can you do",
        "what do you do",
        "how can you help",
        "what are you",
        "who are you",
        "what's your purpose",
        "what can i ask",
        "how does this work",
        "what are your capabilities",
        "what kind of questions",
        "what do you know",
        "tell me about yourself",
    ]

    for pattern in help_patterns:
        if pattern in text_lower:
            return True

    return False


def generate_help_response():
    """Generate helpful response about bot capabilities"""
    responses = [
        "I'm your friendly AI assistant! 🤖 I can help answer questions based on my knowledge base. Try asking about support, billing, passwords, or product features!",
        "I'm here to help! ✨ I have knowledge about:\n• Technical support and contact info\n• Password reset procedures\n• Billing and payment questions\n• Product features and capabilities\n\nJust ask me anything!",
        "Hi! I'm an AI bot that can answer questions using my knowledge base. 👋 I know about support hours, account issues, billing, and product features. What would you like to know?",
        "I can help you find information quickly! 🔍 I have access to knowledge about support, billing, passwords, and product features. Feel free to ask me anything - I'm here to make your life easier!",
    ]

    return random.choice(responses)


def generate_response(question, context_docs):
    """Enhanced response generation with better greeting handling"""
    question_lower = question.lower().strip()

    # Handle greetings first
    if detect_greeting(question):
        return generate_greeting_response()

    # Handle thanks
    if detect_thanks(question):
        return generate_thanks_response()

    # Handle help requests
    if detect_help_request(question):
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
    ]
    if any(word in question_lower for word in farewell_words):
        farewell_responses = [
            "Goodbye! Have a great day! 👋",
            "See you later! Feel free to come back anytime! 😊",
            "Take care! I'll be here whenever you need help! ✨",
            "Bye! Hope I was helpful! 🤖",
            "Farewell! Don't hesitate to reach out if you need anything!",
        ]
        return random.choice(farewell_responses)

    # Regular knowledge base search
    if not context_docs or not any(context_docs):
        return "I couldn't find relevant information in my knowledge base to answer your question. Could you please rephrase or ask about something else? 🤔"

    # Return the most relevant context
    best_match = context_docs[0] if context_docs else ""
    if len(best_match) > 500:
        best_match = best_match[:500] + "..."

    return f"Based on my knowledge base:\n\n{best_match}\n\nIs this helpful? Let me know if you need more specific information! 😊"


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

        # Check if it's a greeting, thanks, or help request first
        if detect_greeting(text) or detect_thanks(text) or detect_help_request(text):
            response = generate_response(text, [])
        else:
            # Search knowledge base for other queries
            search_results = kb.search(text, n_results=3)

            if search_results["documents"] and search_results["documents"][0]:
                context_docs = search_results["documents"][0]
                print(f"Found {len(context_docs)} relevant documents")
                response = generate_response(text, context_docs)
            else:
                response = "I couldn't find relevant information in my knowledge base to answer your question. Could you please rephrase or ask about something else? 🤔"

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
                text="Sorry, I encountered an error processing your request. Please try again. 😅",
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


@app.route("/test_greeting", methods=["POST"])
def test_greeting():
    """Test greeting functionality"""
    try:
        data = request.json
        message = data.get("message", "")

        if not message:
            return jsonify({"error": "Missing message"}), 400

        is_greeting = detect_greeting(message)
        is_thanks = detect_thanks(message)
        is_help = detect_help_request(message)

        if is_greeting:
            response = generate_greeting_response()
            response_type = "greeting"
        elif is_thanks:
            response = generate_thanks_response()
            response_type = "thanks"
        elif is_help:
            response = generate_help_response()
            response_type = "help"
        else:
            response = "This doesn't appear to be a greeting, thanks, or help request."
            response_type = "other"

        return jsonify(
            {
                "message": message,
                "detected_as": response_type,
                "is_greeting": is_greeting,
                "is_thanks": is_thanks,
                "is_help": is_help,
                "response": response,
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
    <h1>Enhanced Slack AI Bot - Local Version</h1>
    <p>Bot is running with enhanced greeting capabilities! 🤖✨</p>
    
    <h2>Available Endpoints:</h2>
    <ul>
        <li><strong>POST /slack/events</strong> - Slack events webhook</li>
        <li><strong>POST /add_knowledge</strong> - Add documents to knowledge base</li>
        <li><strong>GET /list_knowledge</strong> - View all knowledge base documents</li>
        <li><strong>POST /test_search</strong> - Test search functionality</li>
        <li><strong>POST /test_greeting</strong> - Test greeting detection (NEW!)</li>
        <li><strong>GET /health</strong> - Health check</li>
    </ul>
    
    <h2>New Features:</h2>
    <ul>
        <li>🌅 Time-aware greetings (morning, afternoon, evening, night)</li>
        <li>👋 Enhanced greeting detection (supports multiple languages)</li>
        <li>😊 Varied and friendly responses</li>
        <li>🙏 Better thanks/appreciation handling</li>
        <li>❓ Improved help request responses</li>
        <li>👋 Farewell message handling</li>
    </ul>
    
    <p>Total documents in knowledge base: <strong>{}</strong></p>
    <p><em>Try greeting the bot with: "Hello", "Good morning", "Hey there", "How are you?", etc.</em></p>
    """.format(
        len(kb.get_all_documents()["ids"])
    )


def initialize_sample_data():
    """Initialize with sample knowledge base data"""
    sample_docs = [
        {
            "id": "support_hours",
            "text": "Our technical support team is available Monday through Friday, 9 AM to 5 PM EST. You can reach us by emailing enquiery@nervesparks.in or calling 88000-10366. For urgent issues outside business hours, please email varun@nervesparks.in",
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
    print("Starting Enhanced Slack AI Bot (Local Development)")
    print("🤖 Now with improved greeting capabilities!")
    print("Using SQLite database and simple text matching")

    # Initialize sample data
    initialize_sample_data()

    print("\n" + "=" * 60)
    print("🚀 Enhanced Bot is ready!")
    print("👋 Try greeting: 'Hello', 'Good morning', 'Hey there'")
    print("📝 Add documents via: POST /add_knowledge")
    print("🔍 Test search via: POST /test_search")
    print("🎯 Test greetings via: POST /test_greeting")
    print("📋 View docs via: GET /list_knowledge")
    print("🌐 Web interface: http://localhost:3000")
    print("=" * 60 + "\n")

    # Run Flask app
    app.run(host="0.0.0.0", port=3000, debug=True)
