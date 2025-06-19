import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Bot Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for a cleaner look ---
st.markdown("""
<style>
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .st-emotion-cache-1d391kg {
        padding-top: 2rem;
    }
    .st-emotion-cache-1avcm0n {
        background-color: #f0f2f6; /* Light grey for expander headers */
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# --- API Configuration & State Management ---
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://35.209.111.61/api2/"  # Default API URL # health check endpoint

# --- API Helper Function ---
def api_request(endpoint: str, method: str = "GET", data: dict = None, files: dict = None):
    """A centralized function to handle API requests and errors."""
    try:
        url = f"{st.session_state.api_url}{endpoint}"
        
        headers = {'Content-Type': 'application/json'}
        
        if method == "GET":
            response = requests.get(url, timeout=100)
        elif method == "POST":
            # For file uploads, requests handles headers automatically.
            # For JSON, we set it explicitly.
            if files:
                response = requests.post(url, data=data, files=files, timeout=90)
            else:
                response = requests.post(url, json=data, headers=headers, timeout=90)
        elif method == "DELETE":
            response = requests.delete(url, timeout=15)
        else:
            return None, f"Unsupported method: {method}"
        
        response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
        
        # Handle cases where the response might be empty (e.g., DELETE)
        if response.status_code == 204:
            return {}, None
        return response.json(), None

    except requests.exceptions.HTTPError as e:
        error_message = f"API Error {e.response.status_code}"
        try:
            # Try to get the specific error message from the JSON response
            error_details = e.response.json().get('error', e.response.text)
            error_message += f": {error_details}"
        except json.JSONDecodeError:
            error_message += f": {e.response.text}"
        return None, error_message
    except requests.exceptions.ConnectionError:
        return None, "Connection Error: Could not connect to the API. Is the Flask server running?"
    except requests.exceptions.RequestException as e:
        return None, f"Request Error: {e}"


# --- Sidebar: Configuration and Health Status ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.session_state.api_url = st.text_input("API URL", st.session_state.api_url)

    st.divider()
    st.header("🏥 System Health")
    if st.button("Refresh Status", use_container_width=True):
        st.cache_data.clear()  # Clear cache to force a new API call
    
    @st.cache_data(ttl=10)
    def get_health_status():
        """Cached function to fetch health status."""
        return api_request("/health")

    health_data, error = get_health_status()
    
    if error:
        st.error(f"**Health Check Failed:**\n{error}")
    elif health_data:
        components = health_data.get("components", {})
        overall_status = health_data.get("status", "unknown")
        
        if overall_status == "healthy":
            st.success("✅ System is Healthy")
        else:
            st.error("❌ System is Unhealthy")

        for name, comp in components.items():
            status_icon = "✅" if comp.get("status") == "ok" else "❌"
            st.markdown(f"**{name.replace('_', ' ').title()}:** {status_icon}")
        
        st.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")

# --- Main Page Layout ---
st.markdown('<h1 class="main-header">RAG Slack Bot Manager</h1>', unsafe_allow_html=True)

tab1, tab2, tab3= st.tabs(["📊 Dashboard", "📚 Knowledge Base", "🔍 Search"])

# --- Dashboard Tab ---
with tab1:
    st.header("System Overview")
    if health_data and not error:
        components = health_data.get("components", {})
        col1, col2, col3 = st.columns(3)
        
        vector_store_info = components.get("vector_store", {})
        col1.metric("Total Documents (in MongoDB)", vector_store_info.get("total_documents_in_db", "N/A"))
        col2.metric("Total Chunks (in ChromaDB)", vector_store_info.get("total_chunks_in_vector_store", "N/A"))
        
        llm_info = components.get("llm_model", {})
        col3.metric("LLM Model", llm_info.get("model_name", "N/A"))

        st.subheader("Component Status Details")
        st.json(components, expanded=False)
    else:
        st.warning("Could not load dashboard data. Check the API connection in the sidebar.")

# --- Knowledge Base Management Tab ---
with tab2:
    st.header("Manage Knowledge Base")
    
    # Forms for adding new knowledge
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("➕ Add Text Document"):
            with st.form("add_text_form", clear_on_submit=True):
                doc_id = st.text_input("Document ID (unique)*", placeholder="e.g., support-article-01")
                doc_text = st.text_area("Document Text*", height=200)
                meta_str = st.text_input("Metadata (JSON)", value='{"source": "manual"}', help='e.g., {"source": "manual.pdf", "category": "billing"}')
                
                submitted = st.form_submit_button("Add Document")
                if submitted:
                    if not doc_id or not doc_text:
                        st.error("Document ID and Text are required.")
                    else:
                        try:
                            metadata = json.loads(meta_str)
                            with st.spinner("Adding document..."):
                                _, err = api_request("/add_knowledge", "POST", {"id": doc_id, "text": doc_text, "metadata": metadata})
                                if err:
                                    st.error(err)
                                else:
                                    st.success(f"Document '{doc_id}' added!")
                                    st.rerun()
                        except json.JSONDecodeError:
                            st.error("Invalid Metadata JSON format.")
    
    with col2:
        with st.expander("⬆️ Upload PDF Document"):
            with st.form("upload_pdf_form", clear_on_submit=True):
                pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
                pdf_doc_id = st.text_input("Document ID (optional)", placeholder="Auto-generated if blank")
                pdf_meta_str = st.text_input("Metadata (JSON)", value='{}', key="pdf_meta")
                
                submitted = st.form_submit_button("Upload and Process PDF")
                if submitted and pdf_file:
                    with st.spinner(f"Processing '{pdf_file.name}'..."):
                        files = {'file': (pdf_file.name, pdf_file.getvalue(), 'application/pdf')}
                        data = {'doc_id': pdf_doc_id, 'metadata': pdf_meta_str}
                        _, err = api_request("/upload_pdf", "POST", data=data, files=files)
                        if err:
                            st.error(err)
                        else:
                            st.success(f"PDF '{pdf_file.name}' processed!")
                            st.rerun()

    # View and Delete Documents
    st.divider()
    st.subheader("📋 Existing Documents")
    if st.button("Refresh Document List", key="refresh_docs"):
        st.rerun()

    docs_data, docs_error = api_request("/list_knowledge")
if docs_error:
    st.error(f"Could not load documents: {docs_error}")
elif docs_data and docs_data.get("documents"):
    
    # CORRECTED AND MORE ROBUST LOOP
    for index, doc in enumerate(docs_data["documents"]):
        # Use .get() to prevent crashing if a key is missing.
        doc_id = doc.get('doc_id', f'MISSING_ID_{index}') # Provide a fallback ID
        
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**ID:** `{doc_id}`")
                st.json(doc.get("metadata", {}), expanded=False)
            
            with col2:
                # The key for the button must always be unique.
                if st.button("🗑️ Delete", key=f"del_{doc_id}", use_container_width=True):
                    st.session_state[f"confirm_delete_{doc_id}"] = True

            # Confirmation dialog logic also uses the safe doc_id
            if st.session_state.get(f"confirm_delete_{doc_id}", False):
                st.warning(f"Are you sure you want to delete `{doc_id}`?")
                c1, c2, c3 = st.columns([1, 1, 4])
                if c1.button("YES, Delete", key=f"yes_del_{doc_id}", type="primary"):
                    with st.spinner(f"Deleting '{doc_id}'..."):
                        _, err = api_request(f"/delete_knowledge/{doc_id}", "DELETE")
                        if err:
                            st.error(err)
                        else:
                            st.success(f"Deleted '{doc_id}'.")
                        # Clean up session state and rerun
                        if f"confirm_delete_{doc_id}" in st.session_state:
                             del st.session_state[f"confirm_delete_{doc_id}"]
                        st.rerun()

                if c2.button("NO, Cancel", key=f"no_del_{doc_id}"):
                     if f"confirm_delete_{doc_id}" in st.session_state:
                        del st.session_state[f"confirm_delete_{doc_id}"]
                     st.rerun()
else:
    st.info("No documents found in the knowledge base. Add some to get started!")

# --- Search & Debug Tab ---
with tab3:
    st.header("Debug the RAG Pipeline")
    st.info("Test each stage of the RAG pipeline to diagnose issues with retrieval or generation.")
    
    query = st.text_input("Enter a query to test", placeholder="e.g., What are the new billing policies?")

    if query:
        st.divider()
        # --- Stage 1: Retrieval Test ---
        st.subheader("Step 1: Document Retrieval")
        st.markdown("This step tests the vector search. It shows which document chunks are being found for your query.")
        
        with st.spinner("Searching for relevant chunks..."):
            retrieval_data, retrieval_error = api_request("/test_search", "POST", {"query": query, "n_results": 4})
        
        if retrieval_error:
            st.error(f"Retrieval failed: {retrieval_error}")
        elif retrieval_data and retrieval_data.get("results"):
            st.success(f"Found {len(retrieval_data['results'])} relevant chunks.")
            for i, res in enumerate(retrieval_data["results"]):
                with st.expander(f"Chunk {i+1} (Relevance Score: {res['score']:.4f})", expanded= i < 2):
                    st.markdown(res["chunk_text"])
                    st.caption(f"Source Document ID: `{res['metadata'].get('doc_id')}` | Chunk Index: {res['metadata'].get('chunk_index')}")
        else:
            st.warning("No relevant chunks found for this query. The bot will likely say it doesn't know.")
        
        st.divider()

        # --- Stage 2: Generation Test ---
        st.subheader("Step 2: Answer Generation (Full RAG)")
        st.markdown("This step tests the full pipeline: retrieval + passing context to the LLM for a final answer.")

        with st.spinner("Generating final answer..."):
            rag_data, rag_error = api_request("/test_rag_gen", "POST", {"query": query})

        if rag_error:
            st.error(f"Generation failed: {rag_error}")
        elif rag_data:
            st.success("Answer generated successfully.")
            st.markdown("##### Generated Answer:")
            st.info(rag_data.get("generated_answer"))
            
            with st.expander("View Source Documents Used for this Answer"):
                st.json(rag_data.get("source_documents", []))

