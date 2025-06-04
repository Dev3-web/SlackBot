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
            "id": "data_display_errors",
            "text": "Data not displaying correctly? Check filters and date ranges, refresh the page, verify data permissions, and ensure data source is connected. Sometimes cached data needs manual refresh.",
            "metadata": {"category": "troubleshooting", "type": "display"},
        },
        {
            "id": "automation_failures",
            "text": "Automation not running? Verify trigger conditions are met, check automation is enabled, review error logs in automation history, and ensure all connected services are functioning properly.",
            "metadata": {"category": "troubleshooting", "type": "automation"},
        },
        # GETTING STARTED (10 docs)
        {
            "id": "quick_start_guide",
            "text": "Quick start: 1) Complete profile setup 2) Import your data or start with templates 3) Customize dashboard 4) Invite team members 5) Set up integrations. Full onboarding takes 15-30 minutes.",
            "metadata": {"category": "getting_started", "type": "quickstart"},
        },
        {
            "id": "first_login_steps",
            "text": "After first login: 1) Verify email address 2) Complete profile information 3) Choose your timezone 4) Set notification preferences 5) Take the product tour. Skip tour anytime from Help menu.",
            "metadata": {"category": "getting_started", "type": "first_login"},
        },
        {
            "id": "data_migration",
            "text": "Migrating from another platform? Use our migration wizard in Settings > Data Import. Supports CSV, Excel, and direct connections to popular tools. Professional migration services available.",
            "metadata": {"category": "getting_started", "type": "migration"},
        },
        {
            "id": "team_setup",
            "text": "Setting up your team: Invite members from Settings > Team, assign roles and permissions, create shared workspaces, and establish collaboration guidelines. Bulk invite via CSV supported.",
            "metadata": {"category": "getting_started", "type": "team"},
        },
        {
            "id": "workspace_organization",
            "text": "Organize workspaces by project, department, or client. Use consistent naming conventions, set up folder structures, apply tags for easy filtering, and establish workspace templates for new projects.",
            "metadata": {"category": "getting_started", "type": "organization"},
        },
        {
            "id": "initial_configuration",
            "text": "Initial setup checklist: 1) Configure company settings 2) Set up user roles 3) Import existing data 4) Configure integrations 5) Set up backups 6) Train team members. Configuration wizard available.",
            "metadata": {"category": "getting_started", "type": "configuration"},
        },
        {
            "id": "trial_limitations",
            "text": "Free trial includes full features for 14 days with some limitations: 5 team members max, 1GB storage, basic support. No credit card required. Upgrade anytime to continue with full access.",
            "metadata": {"category": "getting_started", "type": "trial"},
        },
        {
            "id": "best_practices",
            "text": "Best practices: Use descriptive naming conventions, regularly backup important data, set up proper user permissions, maintain clean folder structures, and provide team training on key features.",
            "metadata": {"category": "getting_started", "type": "best_practices"},
        },
        {
            "id": "common_mistakes",
            "text": "Avoid common mistakes: Don't skip initial setup steps, don't ignore security settings, don't create too many duplicate folders, don't forget to set up regular backups, don't overlook user training.",
            "metadata": {"category": "getting_started", "type": "mistakes"},
        },
        {
            "id": "success_metrics",
            "text": "Track success with built-in analytics: user adoption rates, feature usage, productivity metrics, and goal completion rates. Custom metrics available for Enterprise plans. Monthly reports provided.",
            "metadata": {"category": "getting_started", "type": "metrics"},
        },
        # ADVANCED FEATURES (10 docs)
        {
            "id": "advanced_analytics",
            "text": "Advanced analytics include predictive modeling, cohort analysis, funnel visualization, statistical calculations, and custom formulas. Machine learning algorithms provide automated insights and recommendations.",
            "metadata": {"category": "advanced", "type": "analytics"},
        },
        {
            "id": "custom_scripting",
            "text": "Custom scripting support with JavaScript, Python, and SQL. Secure sandbox environment prevents security issues. Script library with community contributions. Professional scripting services available.",
            "metadata": {"category": "advanced", "type": "scripting"},
        },
        {
            "id": "enterprise_sso",
            "text": "Enterprise SSO supports SAML 2.0, OAuth 2.0, OpenID Connect, and Active Directory. Automatic user provisioning and de-provisioning. Multi-domain support and custom attribute mapping available.",
            "metadata": {"category": "advanced", "type": "sso"},
        },
        {
            "id": "white_label_options",
            "text": "White-label your instance with custom branding, domain names, logos, and color schemes. Remove all references to our brand. Custom login pages and email templates. Enterprise feature only.",
            "metadata": {"category": "advanced", "type": "whitelabel"},
        },
        {
            "id": "advanced_permissions",
            "text": "Granular permissions system with field-level access control, conditional permissions based on data values, time-based access, and IP restrictions. Audit trail for all permission changes.",
            "metadata": {"category": "advanced", "type": "permissions"},
        },
        {
            "id": "workflow_engine",
            "text": "Advanced workflow engine with parallel processing, conditional branching, loops, error handling, and external service calls. Visual workflow designer with debugging capabilities.",
            "metadata": {"category": "advanced", "type": "workflows"},
        },
        {
            "id": "data_transformation",
            "text": "Built-in data transformation tools with ETL capabilities, data cleansing, format conversion, and validation rules. Schedule automated transformations and monitor data quality metrics.",
            "metadata": {"category": "advanced", "type": "transformation"},
        },
        {
            "id": "custom_fields",
            "text": "Create unlimited custom fields with various data types: text, number, date, dropdown, checkbox, file upload, and calculated fields. Field dependencies and validation rules supported.",
            "metadata": {"category": "advanced", "type": "custom_fields"},
        },
        {
            "id": "advanced_search",
            "text": "Advanced search with full-text indexing, faceted search, saved queries, search within results, and relevance scoring. Search across all data types with real-time suggestions.",
            "metadata": {"category": "advanced", "type": "search"},
        },
        {
            "id": "performance_optimization",
            "text": "Performance optimization features: data caching, lazy loading, query optimization, CDN delivery, and database indexing. Performance monitoring dashboard shows response times and bottlenecks.",
            "metadata": {"category": "advanced", "type": "performance"},
        },
        # BUSINESS & ENTERPRISE (5 docs)
        {
            "id": "enterprise_features",
            "text": "Enterprise features include unlimited users, advanced security, dedicated support, SLA guarantees, custom integrations, professional services, and priority feature requests. Volume discounts available.",
            "metadata": {"category": "enterprise", "type": "features"},
        },
        {
            "id": "dedicated_support",
            "text": "Enterprise customers get dedicated customer success manager, priority phone support, quarterly business reviews, and custom training sessions. Direct escalation path to engineering team.",
            "metadata": {"category": "enterprise", "type": "support"},
        },
        {
            "id": "professional_services",
            "text": "Professional services include custom development, data migration, integration setup, training programs, and consulting. Certified implementation partners available in major regions.",
            "metadata": {"category": "enterprise", "type": "services"},
        },
        {
            "id": "sla_guarantees",
            "text": "Enterprise SLA: 99.99% uptime, 1-hour response time for critical issues, 4-hour resolution target, and monthly performance reports. Financial penalties for SLA breaches. Maintenance windows scheduled.",
            "metadata": {"category": "enterprise", "type": "sla"},
        },
        {
            "id": "volume_licensing",
            "text": "Volume licensing available for 100+ users with significant discounts. Site licenses, concurrent user models, and department-based pricing options. Multi-year agreements offer additional savings.",
            "metadata": {"category": "enterprise", "type": "licensing"},
        },
        {
            "id": "account_locked",
            "text": "If your account is locked, it's usually due to multiple failed login attempts. Wait 15 minutes and try again, or click 'Forgot Password' to reset. For immediate assistance, contact support at enquiery@nervesparks.in",
            "metadata": {"category": "account", "type": "locked"},
        },
        {
            "id": "username_recovery",
            "text": "Forgot your username? Go to the login page and click 'Forgot Username'. Enter your email address and we'll send your username to your registered email within 5 minutes.",
            "metadata": {"category": "account", "type": "username"},
        },
        {
            "id": "email_change",
            "text": "To change your email address: 1) Log into your account 2) Go to Settings > Profile 3) Click 'Change Email' 4) Enter new email and confirm 5) Check both old and new email for verification links",
            "metadata": {"category": "account", "type": "email"},
        },
        {
            "id": "two_factor_auth",
            "text": "Enable two-factor authentication for extra security: 1) Go to Settings > Security 2) Click 'Enable 2FA' 3) Scan QR code with Google Authenticator or Authy 4) Enter verification code 5) Save backup codes safely",
            "metadata": {"category": "account", "type": "security"},
        },
        {
            "id": "profile_update",
            "text": "Update your profile information by going to Settings > Profile. You can change your name, phone number, company details, and preferences. Changes are saved automatically.",
            "metadata": {"category": "account", "type": "profile"},
        },
        {
            "id": "account_deletion",
            "text": "To delete your account: 1) Go to Settings > Account 2) Click 'Delete Account' 3) Enter your password 4) Confirm deletion. Note: This action is permanent and cannot be undone. All data will be removed after 30 days.",
            "metadata": {"category": "account", "type": "deletion"},
        },
        {
            "id": "login_issues",
            "text": "Common login issues: 1) Clear browser cache and cookies 2) Try incognito/private mode 3) Check if Caps Lock is on 4) Ensure you're using the correct login URL 5) Disable browser extensions temporarily",
            "metadata": {"category": "account", "type": "troubleshooting"},
        },
        {
            "id": "session_timeout",
            "text": "Sessions timeout after 30 minutes of inactivity for security. Your work is auto-saved every 2 minutes. To extend session, upgrade to Premium for 8-hour sessions.",
            "metadata": {"category": "account", "type": "session"},
        },
        {
            "id": "multiple_devices",
            "text": "You can access your account from multiple devices. Premium users can have up to 10 active sessions simultaneously. Free users are limited to 3 active sessions.",
            "metadata": {"category": "account", "type": "devices"},
        },
        {
            "id": "password_requirements",
            "text": "Password must be at least 8 characters long, contain uppercase and lowercase letters, at least one number, and one special character (!@#$%^&*). Avoid common passwords and personal information.",
            "metadata": {"category": "account", "type": "password"},
        },
        {
            "id": "account_verification",
            "text": "New accounts require email verification. Check your inbox for a verification email and click the link. If you don't receive it within 10 minutes, check spam folder or request a new verification email.",
            "metadata": {"category": "account", "type": "verification"},
        },
        {
            "id": "guest_access",
            "text": "Guest access allows limited functionality for 7 days. To continue using all features, create a free account. Guest data is automatically deleted after 7 days of inactivity.",
            "metadata": {"category": "account", "type": "guest"},
        },
        {
            "id": "account_security",
            "text": "Keep your account secure: 1) Use strong, unique passwords 2) Enable 2FA 3) Log out from public computers 4) Monitor login activity in Settings > Security 5) Report suspicious activity immediately",
            "metadata": {"category": "account", "type": "security"},
        },
        {
            "id": "privacy_settings",
            "text": "Manage privacy settings in Settings > Privacy. Control who can see your profile, contact you, and access your shared content. You can also manage data sharing preferences and cookie settings.",
            "metadata": {"category": "account", "type": "privacy"},
        },
        # BILLING & PAYMENTS (20 docs)
        {
            "id": "billing_info",
            "text": "Billing questions can be resolved by contacting our billing department at billing@nervesparks.in or calling 88000-10366. Invoices are sent monthly, and payments are due within 30 days. We accept credit cards, bank transfers, and PayPal.",
            "metadata": {"category": "billing", "type": "general"},
        },
        {
            "id": "payment_methods",
            "text": "Accepted payment methods: Visa, MasterCard, American Express, PayPal, bank transfers (ACH), and wire transfers. Cryptocurrency payments accepted for annual plans. Payment processing is secure and PCI compliant.",
            "metadata": {"category": "billing", "type": "payment_methods"},
        },
        {
            "id": "subscription_plans",
            "text": "Available plans: Free (basic features), Professional ($29/month), Business ($99/month), Enterprise (custom pricing). All paid plans include priority support, advanced features, and no usage limits.",
            "metadata": {"category": "billing", "type": "plans"},
        },
        {
            "id": "invoice_questions",
            "text": "Invoices are emailed on the 1st of each month to your billing email. Need a copy? Log into your account > Billing > Invoice History. For custom invoices or purchase orders, contact billing@nervesparks.in",
            "metadata": {"category": "billing", "type": "invoices"},
        },
        {
            "id": "refund_policy",
            "text": "We offer 30-day money-back guarantee for new subscriptions. Annual plans can be refunded within 60 days. Contact billing@nervesparks.in with your refund request and reason. Processing takes 3-5 business days.",
            "metadata": {"category": "billing", "type": "refunds"},
        },
        {
            "id": "upgrade_plan",
            "text": "Upgrade anytime from your account dashboard > Billing > Change Plan. Upgrades are prorated and take effect immediately. You'll be charged the difference for the current billing period.",
            "metadata": {"category": "billing", "type": "upgrade"},
        },
        {
            "id": "downgrade_plan",
            "text": "Downgrades take effect at the end of your current billing cycle. Your data and settings are preserved, but some premium features will be disabled. No refunds for partial periods.",
            "metadata": {"category": "billing", "type": "downgrade"},
        },
        {
            "id": "payment_failed",
            "text": "Payment failed? Check your card details, expiration date, and available balance. Update payment method in Settings > Billing. If issues persist, contact your bank or try a different payment method.",
            "metadata": {"category": "billing", "type": "payment_issues"},
        },
        {
            "id": "billing_address",
            "text": "Update billing address in Settings > Billing > Billing Information. Required for tax calculations and invoice generation. Changes take effect on your next billing cycle.",
            "metadata": {"category": "billing", "type": "address"},
        },
        {
            "id": "tax_information",
            "text": "Tax rates are calculated based on your billing address. VAT/GST is applied for applicable regions. Tax-exempt organizations can submit exemption certificates to billing@nervesparks.in",
            "metadata": {"category": "billing", "type": "tax"},
        },
        {
            "id": "cancel_subscription",
            "text": "Cancel subscription anytime from Settings > Billing > Cancel Subscription. Access continues until end of billing period. Your data is retained for 90 days after cancellation for potential reactivation.",
            "metadata": {"category": "billing", "type": "cancellation"},
        },
        {
            "id": "billing_cycle",
            "text": "Billing cycles: Monthly (charged every 30 days), Annual (charged yearly with 20% discount). Pro-rated charges apply for mid-cycle changes. Billing date is based on your initial subscription date.",
            "metadata": {"category": "billing", "type": "cycle"},
        },
        {
            "id": "corporate_billing",
            "text": "Corporate accounts can pay by invoice with NET 30 terms. Minimum annual commitment required. Contact sales@nervesparks.in for enterprise billing options and volume discounts.",
            "metadata": {"category": "billing", "type": "corporate"},
        },
        {
            "id": "usage_overage",
            "text": "Usage overages are charged at $0.10 per additional unit. You'll receive notifications at 80% and 100% of your plan limits. Upgrade to a higher plan to avoid overage charges.",
            "metadata": {"category": "billing", "type": "overage"},
        },
        {
            "id": "receipt_download",
            "text": "Download receipts from Settings > Billing > Transaction History. Receipts are available immediately after payment and include all necessary details for expense reporting and accounting.",
            "metadata": {"category": "billing", "type": "receipts"},
        },
        {
            "id": "auto_renewal",
            "text": "Subscriptions auto-renew by default. Disable auto-renewal in Settings > Billing. You'll receive reminder emails 7 days and 1 day before renewal. Manual renewal available after cancellation.",
            "metadata": {"category": "billing", "type": "renewal"},
        },
        {
            "id": "student_discount",
            "text": "Students get 50% off Professional plans with valid .edu email address. Apply for student discount in Settings > Billing > Student Discount. Verification required annually.",
            "metadata": {"category": "billing", "type": "discounts"},
        },
        {
            "id": "nonprofit_discount",
            "text": "Registered nonprofits qualify for 30% discount on all plans. Submit 501(c)(3) documentation to billing@nervesparks.in for approval. Discount applies to new and existing subscriptions.",
            "metadata": {"category": "billing", "type": "discounts"},
        },
        {
            "id": "payment_security",
            "text": "All payments are processed securely using 256-bit SSL encryption. We never store complete credit card numbers. Payment processing partners are PCI DSS Level 1 certified.",
            "metadata": {"category": "billing", "type": "security"},
        },
        {
            "id": "billing_contacts",
            "text": "Add multiple billing contacts in Settings > Billing > Contacts. Invoices and payment notifications will be sent to all billing contacts. Useful for accounting departments and managers.",
            "metadata": {"category": "billing", "type": "contacts"},
        },
        # TECHNICAL SUPPORT (15 docs)
        {
            "id": "support_hours",
            "text": "Our technical support team is available Monday through Friday, 9 AM to 5 PM EST. You can reach us by emailing enquiery@nervesparks.in or calling 88000-10366. For urgent issues outside business hours, please email varun@nervesparks.in",
            "metadata": {"category": "support", "type": "hours"},
        },
        {
            "id": "ticket_system",
            "text": "Submit support tickets through the Help Center or email enquiery@nervesparks.in. Include detailed description, screenshots, and steps to reproduce the issue. Response time: Free users 48h, Premium users 4h, Enterprise 1h.",
            "metadata": {"category": "support", "type": "tickets"},
        },
        {
            "id": "emergency_support",
            "text": "Emergency support available 24/7 for Enterprise customers. Contact varun@nervesparks.in or call emergency hotline. Emergencies include system outages, security breaches, or critical business impact issues.",
            "metadata": {"category": "support", "type": "emergency"},
        },
        {
            "id": "live_chat",
            "text": "Live chat available for Premium and Enterprise users during business hours (9 AM - 5 PM EST). Access chat from the help icon in your dashboard. Average response time: 2-5 minutes.",
            "metadata": {"category": "support", "type": "chat"},
        },
        {
            "id": "phone_support",
            "text": "Phone support available for Business and Enterprise plans. Call 88000-10366 during business hours. Have your account details ready. Phone support includes screen sharing for complex technical issues.",
            "metadata": {"category": "support", "type": "phone"},
        },
        {
            "id": "remote_assistance",
            "text": "Remote screen sharing available for Enterprise customers. Our technicians can directly assist with configuration, troubleshooting, and training. Sessions are recorded for quality assurance with your permission.",
            "metadata": {"category": "support", "type": "remote"},
        },
        {
            "id": "knowledge_base",
            "text": "Comprehensive knowledge base available 24/7 at help.nervesparks.in. Includes tutorials, FAQs, video guides, and troubleshooting steps. Search functionality helps find answers quickly.",
            "metadata": {"category": "support", "type": "knowledge_base"},
        },
        {
            "id": "video_tutorials",
            "text": "Video tutorial library covers all major features. Access from Help > Video Tutorials in your dashboard. New tutorials added weekly. Closed captions available in multiple languages.",
            "metadata": {"category": "support", "type": "tutorials"},
        },
        {
            "id": "webinar_training",
            "text": "Free monthly webinars covering advanced features and best practices. Register at training.nervesparks.in. Recorded sessions available for Premium users. Q&A session included in each webinar.",
            "metadata": {"category": "support", "type": "training"},
        },
        {
            "id": "onboarding_support",
            "text": "New Enterprise customers receive dedicated onboarding specialist for first 30 days. Includes setup assistance, user training, and best practice consultation. Schedule at success@nervesparks.in",
            "metadata": {"category": "support", "type": "onboarding"},
        },
        {
            "id": "api_support",
            "text": "API support available through developer forum and documentation at developers.nervesparks.in. Technical questions answered within 24 hours. Code examples and SDKs available for popular languages.",
            "metadata": {"category": "support", "type": "api"},
        },
        {
            "id": "status_page",
            "text": "System status and maintenance schedules at status.nervesparks.in. Subscribe to notifications for real-time updates on outages, maintenance, and performance issues. Mobile app available.",
            "metadata": {"category": "support", "type": "status"},
        },
        {
            "id": "community_forum",
            "text": "Community forum at community.nervesparks.in for user discussions, tips, and peer support. Monitored by support team. Active community with quick responses to common questions.",
            "metadata": {"category": "support", "type": "community"},
        },
        {
            "id": "bug_reports",
            "text": "Report bugs at bugs.nervesparks.in or email bugs@nervesparks.in. Include browser/device info, steps to reproduce, and screenshots. Critical bugs prioritized and fixed within 24 hours.",
            "metadata": {"category": "support", "type": "bugs"},
        },
        {
            "id": "feature_requests",
            "text": "Submit feature requests through the feedback portal in your dashboard or email features@nervesparks.in. Popular requests are prioritized for upcoming releases. You'll be notified when implemented.",
            "metadata": {"category": "support", "type": "features"},
        },
        # PRODUCT FEATURES (20 docs)
        {
            "id": "product_overview",
            "text": "Our software includes advanced analytics, real-time reporting, team collaboration tools, API access, custom integrations, and 24/7 monitoring. Premium plans also include priority support and dedicated account management.",
            "metadata": {"category": "product", "type": "overview"},
        },
        {
            "id": "dashboard_features",
            "text": "Customizable dashboard with drag-and-drop widgets, real-time data visualization, multiple chart types, filtering options, and export capabilities. Save multiple dashboard layouts and share with team members.",
            "metadata": {"category": "product", "type": "dashboard"},
        },
        {
            "id": "reporting_tools",
            "text": "Generate detailed reports with 50+ templates. Custom report builder with advanced filtering, grouping, and calculations. Schedule automatic report delivery via email. Export to PDF, Excel, or CSV formats.",
            "metadata": {"category": "product", "type": "reporting"},
        },
        {
            "id": "collaboration_features",
            "text": "Team collaboration includes shared workspaces, real-time commenting, task assignments, @mentions, file sharing, and activity feeds. Video conferencing integration with Zoom and Teams.",
            "metadata": {"category": "product", "type": "collaboration"},
        },
        {
            "id": "mobile_app",
            "text": "Native mobile apps for iOS and Android with offline sync, push notifications, and core functionality. Download from App Store or Google Play. Requires active subscription for full access.",
            "metadata": {"category": "product", "type": "mobile"},
        },
        {
            "id": "api_capabilities",
            "text": "RESTful API with comprehensive documentation, SDKs for popular languages, webhook support, and OAuth authentication. Rate limits: 1000 requests/hour (Free), 10,000/hour (Pro), unlimited (Enterprise).",
            "metadata": {"category": "product", "type": "api"},
        },
        {
            "id": "integrations_list",
            "text": "Pre-built integrations with Slack, Microsoft Teams, Google Workspace, Salesforce, HubSpot, Zapier, and 100+ other tools. Custom integrations available through API or professional services.",
            "metadata": {"category": "product", "type": "integrations"},
        },
        {
            "id": "data_import",
            "text": "Import data from CSV, Excel, JSON, XML, or database connections. Supports MySQL, PostgreSQL, MongoDB, and SQL Server. Data validation and error reporting included. Bulk import up to 1M records.",
            "metadata": {"category": "product", "type": "import"},
        },
        {
            "id": "data_export",
            "text": "Export data in multiple formats: CSV, Excel, PDF, JSON, XML. Scheduled exports available. API endpoints for programmatic data access. Data retention policies apply based on your plan.",
            "metadata": {"category": "product", "type": "export"},
        },
        {
            "id": "automation_features",
            "text": "Workflow automation with triggers, conditions, and actions. Pre-built templates for common workflows. Custom scripting support. Schedule recurring tasks and notifications. Visual workflow builder included.",
            "metadata": {"category": "product", "type": "automation"},
        },
        {
            "id": "search_functionality",
            "text": "Advanced search with filters, boolean operators, fuzzy matching, and saved searches. Global search across all data types. Search suggestions and autocomplete. Index updated in real-time.",
            "metadata": {"category": "product", "type": "search"},
        },
        {
            "id": "customization_options",
            "text": "Extensive customization: custom fields, layouts, themes, branding, and business rules. White-label options for Enterprise. Custom CSS and JavaScript injection for advanced users.",
            "metadata": {"category": "product", "type": "customization"},
        },
        {
            "id": "notification_system",
            "text": "Comprehensive notification system with email, SMS, push notifications, and in-app alerts. Customize frequency and triggers. Digest options available to reduce notification volume.",
            "metadata": {"category": "product", "type": "notifications"},
        },
        {
            "id": "version_control",
            "text": "Built-in version control tracks all changes with timestamps and user attribution. Restore previous versions, compare changes, and maintain audit trails. Branch and merge functionality for complex projects.",
            "metadata": {"category": "product", "type": "versioning"},
        },
        {
            "id": "templates_library",
            "text": "Extensive template library with 200+ pre-built templates for common use cases. Community-contributed templates available. Create and share custom templates with your team or publicly.",
            "metadata": {"category": "product", "type": "templates"},
        },
        {
            "id": "performance_metrics",
            "text": "Real-time performance monitoring with response times, uptime statistics, and usage analytics. Performance alerts and optimization recommendations. Historical performance data retained for 2 years.",
            "metadata": {"category": "product", "type": "performance"},
        },
        {
            "id": "batch_operations",
            "text": "Bulk operations for editing, deleting, and updating multiple records simultaneously. Background processing for large operations. Progress tracking and cancellation options available.",
            "metadata": {"category": "product", "type": "batch"},
        },
        {
            "id": "ai_features",
            "text": "AI-powered features include smart suggestions, auto-categorization, predictive analytics, and natural language queries. Machine learning models trained on your data for personalized insights.",
            "metadata": {"category": "product", "type": "ai"},
        },
        {
            "id": "calendar_integration",
            "text": "Calendar integration with Google Calendar, Outlook, and Apple Calendar. Schedule meetings, set reminders, and sync deadlines. Two-way sync ensures consistency across platforms.",
            "metadata": {"category": "product", "type": "calendar"},
        },
        {
            "id": "file_management",
            "text": "Integrated file storage with 10GB (Free), 100GB (Pro), 1TB (Business), unlimited (Enterprise). Version control for files, sharing permissions, and preview for 50+ file types.",
            "metadata": {"category": "product", "type": "files"},
        },
        # SECURITY & PRIVACY (15 docs)
        {
            "id": "data_security",
            "text": "Data encrypted in transit (TLS 1.3) and at rest (AES-256). SOC 2 Type II certified with annual audits. Zero-knowledge architecture ensures we cannot access your encrypted data without your permission.",
            "metadata": {"category": "security", "type": "encryption"},
        },
        {
            "id": "privacy_policy",
            "text": "We never sell your data. Minimal data collection for functionality only. GDPR, CCPA, and PIPEDA compliant. Full privacy policy at privacy.nervesparks.in. Regular privacy audits by third parties.",
            "metadata": {"category": "security", "type": "privacy"},
        },
        {
            "id": "data_backup",
            "text": "Automated daily backups with 30-day retention. Geographic redundancy across 3 data centers. Backup restoration available within 4 hours. Enterprise customers can request custom backup schedules.",
            "metadata": {"category": "security", "type": "backup"},
        },
        {
            "id": "access_control",
            "text": "Role-based access control with customizable permissions. Single Sign-On (SSO) support via SAML, OAuth, and Active Directory. Session management with automatic timeout and device tracking.",
            "metadata": {"category": "security", "type": "access"},
        },
        {
            "id": "audit_logs",
            "text": "Comprehensive audit logging of all user actions, data changes, and system events. Logs retained for 1 year (Pro), 7 years (Enterprise). Export capabilities for compliance requirements.",
            "metadata": {"category": "security", "type": "audit"},
        },
        {
            "id": "compliance_standards",
            "text": "Compliant with SOC 2, ISO 27001, GDPR, HIPAA, and PCI DSS. Regular third-party security assessments. Compliance documentation available for Enterprise customers upon request.",
            "metadata": {"category": "security", "type": "compliance"},
        },
        {
            "id": "incident_response",
            "text": "24/7 security monitoring with automated threat detection. Incident response team activates within 15 minutes of security alerts. Customers notified of any security incidents within 2 hours.",
            "metadata": {"category": "security", "type": "incidents"},
        },
        {
            "id": "data_residency",
            "text": "Data centers in US, EU, and Asia-Pacific regions. Choose your preferred data residency location. Data never leaves your selected region without explicit consent. Migration services available.",
            "metadata": {"category": "security", "type": "residency"},
        },
        {
            "id": "vulnerability_management",
            "text": "Regular security scanning and penetration testing. Vulnerability disclosure program with responsible reporting. Security patches applied within 24 hours of discovery. Bug bounty program active.",
            "metadata": {"category": "security", "type": "vulnerabilities"},
        },
        {
            "id": "secure_sharing",
            "text": "Secure link sharing with expiration dates, password protection, and access tracking. Granular sharing permissions at file and folder levels. Revoke access anytime instantly.",
            "metadata": {"category": "security", "type": "sharing"},
        },
        {
            "id": "data_retention",
            "text": "Data retention policies: Active accounts retain data indefinitely, deleted accounts purged after 90 days. Backups retained per compliance requirements. Custom retention policies for Enterprise.",
            "metadata": {"category": "security", "type": "retention"},
        },
        {
            "id": "security_training",
            "text": "All employees complete annual security training. Background checks for all staff with data access. Regular security awareness updates and phishing simulations conducted quarterly.",
            "metadata": {"category": "security", "type": "training"},
        },
        {
            "id": "infrastructure_security",
            "text": "Infrastructure hosted on AWS/Azure with DDoS protection, WAF, and intrusion detection. Network segmentation and zero-trust architecture. 99.99% uptime SLA with redundant systems.",
            "metadata": {"category": "security", "type": "infrastructure"},
        },
        {
            "id": "data_portability",
            "text": "Full data export available anytime in standard formats. API access for automated data extraction. No vendor lock-in - migrate your data easily. Data export completes within 24 hours.",
            "metadata": {"category": "security", "type": "portability"},
        },
        {
            "id": "cookie_policy",
            "text": "We use essential cookies for functionality and optional cookies for analytics. Cookie consent banner allows granular control. Full cookie policy at cookies.nervesparks.in. Manage preferences anytime.",
            "metadata": {"category": "security", "type": "cookies"},
        },
        # TROUBLESHOOTING (15 docs)
        {
            "id": "browser_compatibility",
            "text": "Supported browsers: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+. For best performance, keep your browser updated. Disable ad blockers if experiencing loading issues. Clear cache if features aren't working.",
            "metadata": {"category": "troubleshooting", "type": "browser"},
        },
        {
            "id": "slow_performance",
            "text": "Slow performance? Try: 1) Clear browser cache 2) Disable browser extensions 3) Check internet connection 4) Close unnecessary tabs 5) Use latest browser version. Contact support if issues persist.",
            "metadata": {"category": "troubleshooting", "type": "performance"},
        },
        {
            "id": "file_upload_issues",
            "text": "File upload problems: Check file size limits (10MB free, 100MB pro), supported formats, and internet connection. Try different browser or incognito mode. Large files may take several minutes to upload.",
            "metadata": {"category": "troubleshooting", "type": "uploads"},
        },
        {
            "id": "sync_problems",
            "text": "Sync issues? Force sync by refreshing page or clicking sync button. Check internet connection and login status. Data syncs every 30 seconds when online. Offline changes sync when reconnected.",
            "metadata": {"category": "troubleshooting", "type": "sync"},
        },
        {
            "id": "notification_issues",
            "text": "Not receiving notifications? Check: 1) Notification settings in your account 2) Email spam folder 3) Browser notification permissions 4) Do Not Disturb settings. Test notifications in Settings.",
            "metadata": {"category": "troubleshooting", "type": "notifications"},
        },
        {
            "id": "mobile_app_issues",
            "text": "Mobile app problems: Update to latest version, restart app, check device storage, and ensure stable internet. For persistent issues, try logging out and back in, or reinstall the app.",
            "metadata": {"category": "troubleshooting", "type": "mobile"},
        },
        {
            "id": "integration_failures",
            "text": "Integration not working? Verify API keys, check service status of integrated app, review permission settings, and ensure account connections are active. Re-authorize connections if needed.",
            "metadata": {"category": "troubleshooting", "type": "integrations"},
        },
        {
            "id": "search_not_working",
            "text": "Search issues: Wait 5 minutes for new data to be indexed, check spelling and try different keywords, use filters to narrow results, clear search cache in Settings > Advanced.",
            "metadata": {"category": "troubleshooting", "type": "search"},
        },
        {
            "id": "export_failures",
            "text": "Export not working? Check file size limits, verify export format is supported, ensure stable internet connection, and try exporting smaller data sets. Large exports may take up to 30 minutes.",
            "metadata": {"category": "troubleshooting", "type": "export"},
        },
        {
            "id": "permission_errors",
            "text": "Permission denied errors? Contact your admin to verify access rights, check if your role has required permissions, ensure account is active, and confirm you're accessing correct workspace/project.",
            "metadata": {"category": "troubleshooting", "type": "permissions"},
        },
        {
            "id": "timeout_errors",
            "text": "Request timeout errors indicate server overload or slow connection. Try again in a few minutes, check internet speed, reduce request size, or contact support if persistent during off-peak hours.",
            "metadata": {"category": "troubleshooting", "type": "timeout"},
        },
        {
            "id": "formatting_issues",
            "text": "Text formatting problems? Use supported formats (HTML subset), check character limits, avoid special characters in names, and use template formatting for consistency. Rich text editor has formatting help.",
            "metadata": {"category": "troubleshooting", "type": "formatting"},
        },
        {
            "id": "calendar_sync_issues",
            "text": "Calendar sync problems: Re-authorize calendar connection, check time zone settings, verify calendar permissions, and ensure calendar app is updated. Sync typically takes 5-10 minutes.",
            "metadata": {"category": "troubleshooting", "type": "calendar"},
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
