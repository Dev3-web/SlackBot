import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="RAG Slack Bot Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
DEFAULT_API_URL = "http://localhost:3000"

# Initialize session state
if 'api_url' not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL
if 'last_health_check' not in st.session_state:
    st.session_state.last_health_check = None
if 'health_status' not in st.session_state:
    st.session_state.health_status = {}
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# Helper functions
def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> tuple:
    """Make API request and return (success, response_data, error_message)"""
    try:
        url = f"{st.session_state.api_url}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            if files:
                # For file uploads, use data for form fields and files for the file
                response = requests.post(url, data=data, files=files, timeout=60) 
            else:
                response = requests.post(url, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, timeout=10)
        else:
            return False, None, f"Unsupported method: {method}"
        
        if response.status_code == 200:
            return True, response.json(), None
        else:
            return False, None, f"HTTP {response.status_code}: {response.text}"
    
    except requests.exceptions.ConnectionError:
        return False, None, "Could not connect to the API. Is the Flask server running?"
    except requests.exceptions.Timeout:
        return False, None, "Request timed out. The server might be busy."
    except requests.exceptions.RequestException as e:
        return False, None, f"Request error: {str(e)}"
    except json.JSONDecodeError:
        return False, None, f"Invalid JSON response from server: {response.text}"
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"

def check_health() -> Dict:
    """Check API health and return status"""
    success, data, error = make_api_request("/health")
    if success:
        st.session_state.health_status = data
        st.session_state.last_health_check = datetime.now()
        return data
    else:
        st.session_state.health_status = {"status": "error", "error": error}
        return st.session_state.health_status

def format_timestamp(timestamp):
    """Format timestamp for display"""
    if timestamp:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return "Never"

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API URL Configuration
    st.subheader("🔗 API Configuration")
    api_url = st.text_input(
        "Flask API URL",
        value=st.session_state.api_url,
        help="URL of your Flask RAG bot API"
    )
    
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url
        st.rerun()
    
    # Health Check
    st.subheader("🏥 Health Status")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Check Health", use_container_width=True):
            with st.spinner("Checking..."):
                health_data = check_health()
    
    with col2:
        auto_refresh_button_text = "Stop Auto Refresh" if st.session_state.auto_refresh else "Start Auto Refresh"
        if st.button(auto_refresh_button_text, use_container_width=True):
            st.session_state.auto_refresh = not st.session_state.auto_refresh
            if st.session_state.auto_refresh: # Trigger immediate check if starting auto-refresh
                with st.spinner("Checking..."):
                    health_data = check_health()
            
            # st.rerun() # Rerunning immediately causes issues with button state and refresh logic
    
    # Display health status
    if st.session_state.health_status:
        status = st.session_state.health_status.get('status', 'unknown')
        if status == 'healthy':
            st.success("✅ API is healthy")
        elif status == 'unhealthy':
            st.error(f"❌ API Unhealthy: {st.session_state.health_status.get('error', 'Critical components not ready.')}")
        elif status == 'error':
            st.error(f"❌ API Error: {st.session_state.health_status.get('error', 'Unknown error')}")
        else:
            st.warning("⚠️ Unknown status")
        
        st.caption(f"Last checked: {format_timestamp(st.session_state.last_health_check)}")

# Main content
st.markdown('<h1 class="main-header">🤖 RAG Slack Bot Manager</h1>', unsafe_allow_html=True)

# Auto-refresh health check
if st.session_state.get('auto_refresh', False):
    if not st.session_state.last_health_check or \
       (datetime.now() - st.session_state.last_health_check).seconds > 15: # Refresh every 15 seconds
        check_health()
        st.rerun() # Rerun to update the display

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([ # Added tab6 for PDF Upload
    "📊 Dashboard", 
    "📚 Knowledge Base", 
    "⬆️ Upload PDF", # New tab
    "🔍 Search & Test", 
    "💬 Slack Integration",
    "⚙️ Settings"
])

# Tab 1: Dashboard
with tab1:
    st.header("📊 System Overview")
    
    # Get current health status
    if not st.session_state.health_status:
        with st.spinner("Loading system status..."):
            check_health()
    
    health_data = st.session_state.health_status
    
    if health_data.get('status') == 'healthy' or health_data.get('status') == 'unhealthy': # Show data even if unhealthy
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📄 Total Documents",
                value=health_data.get('total_documents_in_db', 0)
            )
        
        with col2:
            st.metric(
                label="🧩 Total Chunks",
                value=health_data.get('total_chunks_in_db', 0)
            )
        
        with col3:
            st.metric(
                label="🔍 FAISS Index Size",
                value=health_data.get('total_chunks_in_faiss', 0)
            )
        
        with col4:
            st.metric(
                label="📐 Vector Dimension",
                value=health_data.get('vector_dimension', 0)
            )
        
        # System status
        st.subheader("🔧 System Components")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 AI Components")
            components = [
                ("Embedding Model", health_data.get('embedding_model', 'Unknown')),
                ("QA Model", "✅ Loaded" if health_data.get('qa_model_loaded') else "❌ Not loaded"),
                ("Gemini API", "✅ Configured" if health_data.get('gemini_api_configured') else "❌ Not configured"),
            ]
            
            for name, status in components:
                st.write(f"**{name}:** {status}")
        
        with col2:
            st.markdown("### 💾 Database")
            db_info = [
                ("Database Type", health_data.get('database_type', 'Unknown')),
                ("Database Name", health_data.get('database_name', 'Unknown')),
                ("MongoDB Connected", "✅ Connected" if health_data.get('mongodb_connected') else "❌ Disconnected"),
            ]
            
            for name, status in db_info:
                st.write(f"**{name}:** {status}")
        
        # Database statistics chart
        if health_data.get('total_documents_in_db', 0) > 0:
            st.subheader("📈 Storage Statistics")
            
            # Create a simple chart
            fig = go.Figure(data=[
                go.Bar(name='Documents', x=['Database'], y=[health_data.get('total_documents_in_db', 0)]),
                go.Bar(name='Chunks (DB)', x=['Database'], y=[health_data.get('total_chunks_in_db', 0)]),
                go.Bar(name='Chunks (FAISS)', x=['Database'], y=[health_data.get('total_chunks_in_faiss', 0)])
            ])
            fig.update_layout(barmode='group', title="Data Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("❌ Unable to connect to the API or unknown status. Please check your configuration.")
        if health_data.get('error'):
            st.error(f"Error: {health_data['error']}")

# Tab 2: Knowledge Base Management
with tab2:
    st.header("📚 Knowledge Base Management")
    
    # Sub-tabs for knowledge base operations
    kb_tab1, kb_tab2, kb_tab3 = st.tabs(["➕ Add Text Document", "📋 View Documents", "🗑️ Delete Documents"])
    
    with kb_tab1:
        st.subheader("➕ Add New Text Document")
        
        with st.form("add_document_form"):
            doc_id = st.text_input(
                "Document ID*",
                help="Unique identifier for the document"
            )
            
            doc_text = st.text_area(
                "Document Text*",
                height=300,
                help="The main content of the document"
            )
            
            # Metadata section
            st.subheader("📝 Metadata (Optional)")
            col1, col2 = st.columns(2)
            
            with col1:
                category = st.text_input("Category", help="Document category (e.g., support, billing)", key="add_category")
                source = st.text_input("Source", help="Source of the document", key="add_source")
            
            with col2:
                author = st.text_input("Author", help="Document author", key="add_author")
                tags = st.text_input("Tags (comma-separated)", help="Comma-separated tags", key="add_tags")
            
            submitted = st.form_submit_button("📤 Add Document")
            
            if submitted:
                if not doc_id or not doc_text:
                    st.error("❌ Document ID and Text are required!")
                else:
                    # Prepare metadata
                    metadata = {}
                    if category: metadata['category'] = category
                    if source: metadata['source'] = source
                    if author: metadata['author'] = author
                    if tags: metadata['tags'] = [tag.strip() for tag in tags.split(',') if tag.strip()]
                    
                    # Add document
                    with st.spinner("Adding document..."):
                        success, response, error = make_api_request(
                            "/add_knowledge",
                            method="POST",
                            data={
                                "id": doc_id,
                                "text": doc_text,
                                "metadata": metadata
                            }
                        )
                    
                    if success:
                        st.success(f"✅ {response.get('message', 'Document added successfully!')}")
                        time.sleep(1)
                        st.rerun() # Rerun to refresh the view documents list
                    else:
                        st.error(f"❌ Error adding document: {error}")
    
    with kb_tab2:
        st.subheader("📋 Current Documents")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Refresh List", use_container_width=True, key="refresh_doc_list"):
                st.rerun()
        
        # Get document list
        with st.spinner("Loading documents..."):
            success, doc_data, error = make_api_request("/list_knowledge")
        
        if success and doc_data:
            st.info(f"📊 **Total Documents:** {doc_data.get('total_documents', 0)} | "
                   f"**FAISS Chunks:** {doc_data.get('total_chunks_in_faiss', 0)} | "
                   f"**Model:** {doc_data.get('embedding_model', 'Unknown')}")
            
            if doc_data.get('documents'):
                for i, doc in enumerate(doc_data['documents']):
                    with st.expander(f"📄 {doc['id']}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write("**Preview:**")
                            st.text(doc['text_preview'])
                            
                            if doc.get('metadata'):
                                st.write("**Metadata:**")
                                st.json(doc['metadata'])
                        
                        with col2:
                            # Use a unique key for each delete button
                            delete_button_key = f"delete_doc_{doc['id']}_{i}" 
                            if st.button(f"🗑️ Delete", key=delete_button_key, use_container_width=True):
                                # Set a session state flag for confirmation
                                st.session_state[f"confirm_delete_{doc['id']}"] = True
                            
                            # Confirmation dialog
                            if st.session_state.get(f"confirm_delete_{doc['id']}", False):
                                st.warning(f"⚠️ Are you sure you want to delete '{doc['id']}'?")
                                col_yes, col_no = st.columns(2)
                                
                                with col_yes:
                                    if st.button("✅ Yes", key=f"yes_delete_{doc['id']}"):
                                        with st.spinner(f"Deleting '{doc['id']}'..."):
                                            del_success, del_response, del_error = make_api_request(
                                                f"/delete_knowledge/{doc['id']}", 
                                                method="DELETE"
                                            )
                                        
                                        if del_success:
                                            st.success(f"✅ Document '{doc['id']}' deleted!")
                                            time.sleep(1)
                                            st.session_state[f"confirm_delete_{doc['id']}"] = False # Reset flag
                                            st.rerun() # Refresh the list
                                        else:
                                            st.error(f"❌ Error deleting document '{doc['id']}': {del_error}")
                                            st.session_state[f"confirm_delete_{doc['id']}"] = False # Reset flag
                                
                                with col_no:
                                    if st.button("❌ No", key=f"no_delete_{doc['id']}"):
                                        st.session_state[f"confirm_delete_{doc['id']}"] = False
                                        st.rerun() # Rerun to remove confirmation prompt
            else:
                st.info("📝 No documents found. Add some documents to get started!")
        
        elif error:
            st.error(f"❌ Error loading documents: {error}")
    
    with kb_tab3:
        st.subheader("🗑️ Bulk Delete Operations")
        
        st.warning("⚠️ **Warning:** This section allows bulk deletion of documents. Use with caution!")
        
        # Get database stats
        with st.spinner("Loading database statistics..."):
            success, stats, error = make_api_request("/db_stats")
        
        if success and stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Documents", stats.get('total_documents', 0))
            with col2:
                st.metric("Chunks", stats.get('total_chunks', 0))
            with col3:
                st.metric("FAISS Index", stats.get('faiss_index_size', 0))
            
            # Simple "Delete All" for demonstration purposes
            if stats.get('total_documents', 0) > 0:
                if st.button("🚨 Delete ALL Documents and Chunks (Irreversible)", type="secondary", use_container_width=True):
                    st.session_state["confirm_delete_all"] = True
                
                if st.session_state.get("confirm_delete_all", False):
                    st.warning("‼️ ABSOLUTELY SURE YOU WANT TO DELETE ALL DATA? This cannot be undone.")
                    col_yes_all, col_no_all = st.columns(2)
                    with col_yes_all:
                        if st.button("✅ YES, DELETE EVERYTHING", key="confirm_yes_delete_all"):
                            with st.spinner("Deleting all data... This may take a moment."):
                                # This is a placeholder. You'd need a backend endpoint for this.
                                # For now, we iterate and delete. More efficient would be a dedicated endpoint.
                                success_all = True
                                errors_all = []
                                _, all_docs, _ = make_api_request("/list_knowledge")
                                if all_docs and all_docs.get("documents"):
                                    for doc in all_docs["documents"]:
                                        del_s, del_r, del_e = make_api_request(f"/delete_knowledge/{doc['id']}", method="DELETE")
                                        if not del_s:
                                            success_all = False
                                            errors_all.append(f"Failed to delete {doc['id']}: {del_e}")
                                
                                if success_all:
                                    st.success("✅ All documents and chunks deleted successfully!")
                                else:
                                    st.error(f"❌ Failed to delete all documents. Errors: {'; '.join(errors_all)}")
                                
                                st.session_state["confirm_delete_all"] = False
                                time.sleep(1)
                                st.rerun()
                    with col_no_all:
                        if st.button("❌ NO, CANCEL", key="confirm_no_delete_all"):
                            st.session_state["confirm_delete_all"] = False
                            st.rerun()
            else:
                st.info("No documents to delete.")
        
        elif error:
            st.error(f"❌ Error loading database stats: {error}")

# Tab 3: PDF Upload
with tab3: # Updated index to 3
    st.header("⬆️ Upload PDF Document")
    
    st.info("Upload a PDF file to add its content to the knowledge base. Text will be extracted, chunked, and embedded.")
    
    with st.form("upload_pdf_form"):
        pdf_file = st.file_uploader(
            "Select a PDF file",
            type="pdf",
            help="Choose a PDF document to upload. Max file size depends on server limits."
        )
        
        pdf_doc_id = st.text_input(
            "Document ID (Optional)",
            help="A unique ID for this PDF. If left blank, one will be generated automatically (e.g., `pdf_YYYYMMDDHHMMSS_XXXX`)."
        )
        
        st.subheader("📝 Metadata (Optional)")
        col1, col2 = st.columns(2)
        
        with col1:
            pdf_category = st.text_input("Category", help="Document category (e.g., manual, report)", key="pdf_category")
            pdf_source = st.text_input("Source", help="Source of the PDF (e.g., website, internal)", key="pdf_source")
        
        with col2:
            pdf_author = st.text_input("Author", help="Author of the PDF", key="pdf_author")
            pdf_tags = st.text_input("Tags (comma-separated)", help="Comma-separated keywords/tags for the PDF", key="pdf_tags")
            
        uploaded = st.form_submit_button("⬆️ Upload and Process PDF")
        
        if uploaded and pdf_file:
            if not pdf_file.name.lower().endswith('.pdf'):
                st.error("❌ Please upload a valid PDF file.")
            else:
                # Prepare metadata for upload
                pdf_metadata = {}
                if pdf_category: pdf_metadata['category'] = pdf_category
                if pdf_source: pdf_metadata['source'] = pdf_source
                if pdf_author: pdf_metadata['author'] = pdf_author
                if pdf_tags: pdf_metadata['tags'] = [tag.strip() for tag in pdf_tags.split(',') if tag.strip()]
                
                # Convert metadata to JSON string for the form data
                pdf_metadata_json = json.dumps(pdf_metadata)

                with st.spinner(f"Uploading and processing '{pdf_file.name}'... This may take a few moments for large files."):
                    files = {'file': (pdf_file.name, pdf_file.getvalue(), 'application/pdf')}
                    data = {'doc_id': pdf_doc_id, 'metadata': pdf_metadata_json}
                    
                    success, response, error = make_api_request(
                        "/upload_pdf",
                        method="POST",
                        data=data,
                        files=files
                    )
                
                if success:
                    st.success(f"✅ {response.get('message', 'PDF uploaded and processed successfully!')}")
                    if response.get('doc_id'):
                        st.info(f"Assigned Document ID: `{response['doc_id']}`")
                    if response.get('extracted_text_preview'):
                        st.text_area("Extracted Text Preview:", value=response['extracted_text_preview'], height=150, disabled=True)
                    time.sleep(1)
                    st.rerun() # Refresh the view to show new docs
                else:
                    st.error(f"❌ Error uploading PDF: {error}")
        elif uploaded and not pdf_file:
            st.warning("⚠️ Please select a PDF file to upload.")

# Tab 4: Search & Test (shifted index to 4)
with tab4:
    st.header("🔍 Search & Test Interface")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Search Query",
            placeholder="Enter your search query here..."
        )
    
    with col2:
        search_type = st.selectbox(
            "Search Type",
            options=["hybrid", "semantic", "keyword"],
            index=0
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_results = st.slider("Number of Results", min_value=1, max_value=10, value=3)
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        search_button = st.button("🔍 Search", use_container_width=True)
    
    with col3:
        st.write("")  # Spacer
        st.write("")  # Spacer
        clear_button = st.button("🧹 Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if search_button and search_query:
        with st.spinner("Searching..."):
            success, results, error = make_api_request(
                "/test_search",
                method="POST",
                data={
                    "query": search_query,
                    "search_type": search_type,
                    "n_results": n_results
                }
            )
        
        if success and results:
            st.success(f"✅ Found {len(results.get('results', []))} results")
            
            # Display search metadata
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Query:** {results['query']}")
            with col2:
                st.info(f"**Search Type:** {results['search_type'].title()}")
            
            # Display results
            if results.get('results'):
                for i, result in enumerate(results['results']):
                    with st.expander(f"🔍 Result {i+1} - {result.get('id', 'Unknown ID')}", expanded=i==0):
                        
                        # Scores
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            score = result.get('score')
                            if score is not None:
                                st.metric("Overall Score", f"{score:.3f}")
                        
                        with col2:
                            sem_score = result.get('semantic_score')
                            if sem_score is not None:
                                st.metric("Semantic Score", f"{sem_score:.3f}")
                        
                        with col3:
                            kw_score = result.get('keyword_score')
                            if kw_score is not None:
                                st.metric("Keyword Score", f"{kw_score:.3f}")
                        
                        # Document content
                        st.write("**Content:**")
                        st.text(result['document'])
                        
                        # Metadata
                        if result.get('metadata'):
                            st.write("**Metadata:**")
                            st.json(result['metadata'])
            else:
                st.info("🔍 No results found for your query.")
        
        elif error:
            st.error(f"❌ Search error: {error}")
    
    elif search_button and not search_query:
        st.warning("⚠️ Please enter a search query.")

# Tab 5: Slack Integration (shifted index to 5)
with tab5:
    st.header("💬 Slack Integration")
    
    # Slack status check
    health_data = st.session_state.health_status
    
    if health_data.get('status') == 'healthy':
        # Slack configuration status
        st.subheader("🔧 Slack Configuration Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Slack Bot Token:** {'✅ Configured' if health_data.get('slack_bot_token_configured') else '❌ Not configured'}")
            st.info("**Bot User ID:** (Set in env for app to use, but not directly checked by API health)")
        
        with col2:
            st.info("**Webhook URL:** `/slack/events`")
            st.info("**Event Subscriptions:** `app_mention`, `message.im`")
        
        # Test message interface
        st.subheader("🧪 Test Bot Response")
        
        test_message = st.text_input(
            "Test Message",
            placeholder="Enter a message to test how the bot would respond..."
        )
        
        if st.button("🤖 Get Bot Response"):
            if test_message:
                # Simulate bot response by doing a search
                with st.spinner("Generating response..."):
                    success, results, error = make_api_request(
                        "/test_search",
                        method="POST",
                        data={
                            "query": test_message,
                            "search_type": "hybrid",
                            "n_results": 3
                        }
                    )
                
                if success and results:
                    st.subheader("🤖 Bot Response Preview:")
                    
                    # Simulate the bot's response logic
                    if results.get('results'):
                        best_result = results['results'][0]
                        confidence = best_result.get('score', 0)
                        
                        emoji = "🎯" if confidence > 0.7 else "📈" if confidence > 0.55 else "💡"
                        
                        response = (
                            f"{emoji} Based on my knowledge, here's some information:\n\n"
                            f"{best_result['document'][:500]}...\n\n"
                            f"Is this helpful? (Relevance: {confidence:.2f})"
                        )
                    else:
                        response = "I couldn't find relevant information for your question. 🤔 Could you rephrase or ask about something else?"
                    
                    st.success(response)
                else:
                    st.error(f"Error generating response: {error}")
            else:
                st.warning("Please enter a test message.")
        
        # Slack setup instructions
        st.subheader("📋 Slack Setup Guide")
        
        with st.expander("🔧 How to set up Slack integration"):
            st.markdown("""
            **1. Create a Slack App:**
            - Go to https://api.slack.com/apps
            - Click "Create New App"
            - Choose "From scratch"
            - Name your app and select your workspace
            
            **2. Configure Bot Token Scopes:**
            - Go to "OAuth & Permissions"
            - Add these scopes:
              - `app_mentions:read`
              - `chat:write`
              - `im:history`
              - `im:read`
              - `im:write`
            
            **3. Install App to Workspace:**
            - Click "Install to Workspace"
            - Copy the "Bot User OAuth Token"
            - Set it as `SLACK_BOT_TOKEN` in your environment (`.env` file)
            
            **4. Configure Event Subscriptions:**
            - Go to "Event Subscriptions"
            - Enable Events
            - Set Request URL to: `your-server.com/slack/events` (This must be a public URL that Slack can reach. Use `ngrok` for local development.)
            - Subscribe to these events:
              - `app_mention`
              - `message.im`
            
            **5. Get Bot User ID:**
            - Go to "App Home"
            - Copy the Bot User ID
            - Set it as `SLACK_BOT_USER_ID` in your environment (`.env` file)
            
            **6. Enable Socket Mode (Alternative to Public URL):**
            - If you can't expose your local server publicly, enable Socket Mode (requires `SLACK_APP_TOKEN` in addition to `SLACK_BOT_TOKEN`).
            - Go to "Basic Information" -> "App-Level Tokens" to generate a token starting with `xapp-`.
            - This method requires a different Slack SDK setup for event listening. The current Flask app uses HTTP events.
            """)
    
    else:
        st.error("❌ Cannot check Slack integration - API connection failed or unhealthy.")

# Tab 6: Settings (shifted index to 6)
with tab6:
    st.header("⚙️ System Settings")
    
    # API Settings
    st.subheader("🔗 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input(
            "Current API URL",
            value=st.session_state.api_url,
            disabled=True
        )
    
    with col2:
        new_api_url = st.text_input(
            "Update API URL",
            placeholder="http://localhost:3000",
            key="update_api_url_input"
        )
        
        if st.button("🔄 Update URL", key="update_api_url_button") and new_api_url:
            st.session_state.api_url = new_api_url
            st.success("✅ API URL updated!")
            time.sleep(1)
            st.rerun()
    
    # System Information
    st.subheader("ℹ️ System Information")
    
    if st.session_state.health_status: # Show available info even if not fully healthy
        health_data = st.session_state.health_status
        
        info_data = {
            "Overall Status": health_data.get('status', 'Unknown'),
            "RAG Enabled": "✅ Yes" if health_data.get('rag_enabled') else "❌ No",
            "Database Type": health_data.get('database_type', 'Unknown'),
            "Database Name": health_data.get('database_name', 'Unknown'),
            "MongoDB Connected": "✅ Connected" if health_data.get('mongodb_connected') else "❌ Disconnected",
            "Embedding Model": health_data.get('embedding_model', 'Unknown'),
            "Vector Dimension": health_data.get('vector_dimension', 'Unknown'),
            "QA Model Loaded": "✅ Yes" if health_data.get('qa_model_loaded') else "❌ No",
            "Gemini API Configured": "✅ Yes" if health_data.get('gemini_api_configured') else "❌ No",
            "Slack Bot Token Configured": "✅ Yes" if health_data.get('slack_bot_token_configured') else "❌ No",
        }
        
        for key, value in info_data.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**{key}:**")
            with col2:
                st.write(value)
    else:
        st.info("No system information available. Check API connection.")
    
    # Advanced Settings
    st.subheader("🔬 Advanced Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Force Refresh Health Status", use_container_width=True, key="force_refresh_health"):
            with st.spinner("Checking health..."):
                check_health()
            st.success("✅ Health status refreshed!")
    
    with col2:
        if st.button("📊 Get Detailed Database Stats", use_container_width=True, key="get_detailed_db_stats"):
            with st.spinner("Loading stats..."):
                success, stats, error = make_api_request("/db_stats")
            
            if success:
                st.json(stats)
            else:
                st.error(f"Error: {error}")
    
    # Environment Variables Guide
    with st.expander("🔧 Environment Variables Guide"):
        st.markdown("""
        **Required Environment Variables:**
        
        ```bash
        # Google Gemini API
        GOOGLE_API_KEY=your_gemini_api_key_here
        
        # Slack Integration
        SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
        SLACK_BOT_USER_ID=U1234567890 # Found in your Slack App Home
        
        # MongoDB (optional - defaults to mongodb://localhost:27017/)
        MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
        MONGODB_DB_NAME=rag_knowledge_base
        
        # OpenAI (optional - for enhanced responses via OpenAI LLM)
        OPENAI_API_KEY=your_openai_api_key_here
        
        # HuggingFace QA Model (optional - defaults to distilbert-base-cased-distilled-squad)
        QA_MODEL=distilbert-base-cased-distilled-squad
        
        # Flask Server Port
        PORT=3000
        ```
        
        **Setup Instructions:**
        1. Create a `.env` file in your project root (`.env` file should be at the same level as `app.py`).
        2. Add the variables above with your actual values.
        3. Restart your Flask server for changes to take effect.
        4. Refresh this Streamlit page to see updated status.
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🤖 RAG Slack Bot Manager | Built with Streamlit | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</div>",
    unsafe_allow_html=True
)