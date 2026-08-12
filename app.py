import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore

# Load Environment Variables
load_dotenv()

# Initialize LLM Model 
# Using gemini-3.5-flash as it is fully supported by your API key!
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# RAG Pipeline: Document Loading, Splitting, Embedding, and Vector DB Storage
def documents_process(uploaded_files):
    import tempfile
    import os
    
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        # Save file to a temporary location
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        try:
            # 1. Load the Document
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            
            # Update the source metadata to use the original filename
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            
            # 2. Split Document into Chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
        finally:
            # Clean up the temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                    
    if not all_chunks:
        return None
        
    # 3. Create Embeddings Object
    # Using models/gemini-embedding-001 as it is supported by your API key!
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 4. Store Chunks in In-Memory Vector Store
    vector_db = InMemoryVectorStore.from_documents(documents=all_chunks, embedding=embeddings)
    
    return vector_db

def extract_text_content(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    return str(content)

# ----------------- STREAMLIT UI -----------------

st.set_page_config(
    page_title="Enterprise Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling Injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Main app background and styling */
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a 60%);
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Sidebar background styling */
    [data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }

    /* Sidebar Title and Text */
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 20px;
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .sidebar-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        backdrop-filter: blur(5px);
    }

    /* Header Text styling */
    .app-title-container {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 30px;
    }

    .app-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 5px;
    }

    .app-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 400;
    }

    /* Landing Card styles */
    .landing-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3), 0 0 40px rgba(99, 102, 241, 0.05);
        backdrop-filter: blur(12px);
        margin: 20px auto;
        max-width: 800px;
    }

    .feature-tag {
        background: rgba(99, 102, 241, 0.1);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    /* Upload box customization */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.4);
        border: 2px dashed rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.1);
    }

    /* Button style override */
    div.stButton > button {
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
        color: white !important;
        border: none !important;
        padding: 8px 20px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
        width: 100%;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4) !important;
    }

    /* Chat message overrides */
    [data-testid="stChatMessage"] {
        background-color: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 12px !important;
        backdrop-filter: blur(8px);
    }

    /* Status indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 197, 94, 0.1);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .info-label {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-bottom: 4px;
    }

    .info-value {
        color: #ffffff;
        font-size: 0.95rem;
        font-weight: 600;
        word-break: break-all;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define Session States to preserve variables across reruns
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "original_filenames" not in st.session_state:
    st.session_state.original_filenames = []

# Sidebar panel setup
st.sidebar.markdown('<div class="sidebar-title">⚙️ Control Panel</div>', unsafe_allow_html=True)

if st.session_state.document_uploaded:
    # Build list of active files
    files_html = "".join([f'<div class="info-value" style="margin-bottom: 6px; font-size: 0.85rem;">📄 {name}</div>' for name in st.session_state.original_filenames])
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <div class="status-badge">🟢 Engine Active</div>
            <div style="margin-top: 15px;">
                <div class="info-label">Active Documents ({len(st.session_state.original_filenames)})</div>
                <div style="max-height: 200px; overflow-y: auto; margin-top: 8px;">
                    {files_html}
                </div>
            </div>
            <div style="margin-top: 15px;">
                <div class="info-label">AI Engine</div>
                <div class="info-value">Gemini 3.5 Flash</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Reset Application Button
    if st.sidebar.button("🗑️ Clear & Reset Engine"):
        st.session_state.document_uploaded = False
        st.session_state.vector_db = None
        st.session_state.messages = []
        st.session_state.original_filenames = []
        st.rerun()
else:
    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <div class="info-label">System Status</div>
            <div style="color: #ef4444; font-weight: 600; font-size: 0.95rem;">🔴 Engine Inactive</div>
            <div style="margin-top: 10px; font-size: 0.8rem; color: #94a3b8;">
                Please upload PDF documents in the main panel and click "Process" to initialize the vector store.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown(
    """
    <div style="position: fixed; bottom: 20px; font-size: 0.75rem; color: #64748b;">
        Enterprise Document Intelligence v1.2
    </div>
    """,
    unsafe_allow_html=True
)

# Section 1: PDF Upload Window (Landing Page)
if not st.session_state.document_uploaded:
    st.markdown(
        """
        <div class="app-title-container">
            <div class="app-title">Enterprise Document Intelligence</div>
            <div class="app-subtitle">A high-performance Retrieval-Augmented Generation Chatbot</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown(
            """
            <div class="landing-card">
                <h3 style="margin-top: 0; color: #ffffff; font-size: 1.3rem; font-weight: 600; margin-bottom: 12px;">Get Started</h3>
                <p style="color: #94a3b8; line-height: 1.6; margin-bottom: 20px; font-size: 0.95rem;">
                    Upload one or more PDF documents. The intelligence suite will partition, vectorize, and index the content 
                    using Gemini embeddings and store it in an in-memory vector space. Once processed, you can ask natural language questions with source citations.
                </p>
                <div style="margin-bottom: 15px;">
                    <span class="feature-tag">⚡ Multi-File Ingestion</span>
                    <span class="feature-tag">🔑 Secure Sandbox</span>
                    <span class="feature-tag">📑 Source Referencing</span>
                    <span class="feature-tag">🚀 Gemini Powered</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        uploaded_files = st.file_uploader(label="Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_files:
            # Display uploaded files list
            files_list_html = "".join([f'<div style="font-size: 0.9rem; color: #f1f5f9; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">📄 {f.name} <span style="color: #64748b; font-size: 0.75rem;">({f.size / 1024:.1f} KB)</span></div>' for f in uploaded_files])
            st.markdown(
                f"""
                <div style="margin-top: 15px; margin-bottom: 20px; padding: 16px; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);">
                    <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; margin-bottom: 8px;">Selected Files ({len(uploaded_files)}):</div>
                    {files_list_html}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if st.button("🚀 Process & Start Chatting"):
                # Show a spinner while the document is processing
                with st.spinner("Partitioning and Ingesting Documents into Vector DB..."):
                    st.session_state.original_filenames = [f.name for f in uploaded_files]
                    st.session_state.vector_db = documents_process(uploaded_files)
                    
                if st.session_state.vector_db:
                    st.success("Analysis Complete! Starting session...")
                    time.sleep(1.5)  # Delay to show the success message
                    st.session_state.document_uploaded = True
                    st.rerun()     # Refresh UI to show Chat Window
                else:
                    st.error("Could not process the documents. Please verify that the files are valid PDFs with text.")

# Section 2: Chat UI Window
if st.session_state.document_uploaded and st.session_state.vector_db:
    # Build clean header string of active documents
    files_str = ", ".join(st.session_state.original_filenames)
    if len(files_str) > 60:
        files_str = files_str[:57] + "..."
        
    st.markdown(
        f"""
        <div class="app-title-container" style="text-align: left; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 15px;">
            <div class="app-title" style="font-size: 1.8rem; display: inline-block;">Document Q&A Chatroom</div>
            <div class="app-subtitle" style="display: inline-block; margin-left: 15px;">Chatting with <b>{files_str}</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Render all previous conversation messages on screen
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
            
            # Render sources if available in message
            if role == "ai" and "sources" in message and message["sources"]:
                st.markdown("<div style='margin-top: 15px; font-size: 0.8rem; color: #94a3b8; font-weight: 600;'>📚 Sources Cited:</div>", unsafe_allow_html=True)
                cols = st.columns(len(message["sources"]))
                for idx, src in enumerate(message["sources"]):
                    filename = src.get("source", "Unknown Document")
                    pages_list = src.get("pages", [])
                    if not pages_list and "page" in src:
                        pages_list = [src["page"]]
                    
                    if len(pages_list) == 1:
                        pages_str = f"Page {pages_list[0]}"
                    elif len(pages_list) > 1:
                        pages_str = "Pages " + ", ".join(map(str, sorted(list(set(pages_list)))))
                    else:
                        pages_str = "Page 1"
                        
                    with cols[idx]:
                        st.markdown(
                            f"""
                            <div class="sidebar-card" style="margin-bottom: 0px; padding: 10px; border: 1px solid rgba(99, 102, 241, 0.25);">
                                <div style="color: #a5b4fc; font-weight: 600; font-size: 0.8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{filename}">📄 {filename}</div>
                                <div style="color: #64748b; font-size: 0.75rem; margin-top: 2px;">{pages_str}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
    # Chat Input 
    query = st.chat_input("Ask a question about the documents...")
    
    if query:
        # Save and Display User Question
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)
            
        # Perform Similarity Search inside the Vector Database
        with st.spinner("Searching documents context and generating answer..."):
            documents = st.session_state.vector_db.similarity_search(query, k=2)
            
            # Generate Context string from retrieved chunks
            context = ""
            for doc in documents:
                context += doc.page_content + "\n\n"
                
            # Prepare Prompt template dynamically
            prompt = f"""You are a helpful assistant and you provide answers for user questions based on the provided context.
            Context: {context}
            Question: {query}"""
            
            # Generate Answer from LLM
            result = llm.invoke(prompt)
            ai_response = extract_text_content(result.content)
            
            # Extract and group source pages by PDF filename
            temp_sources = {}
            for doc in documents:
                src_name = doc.metadata.get("source", "Unknown Document")
                page_num = doc.metadata.get("page", 0) + 1
                if src_name not in temp_sources:
                    temp_sources[src_name] = set()
                temp_sources[src_name].add(page_num)
            
            sources = []
            for src_name, pages in temp_sources.items():
                sources.append({"source": src_name, "pages": sorted(list(pages))})
            
        # Save and Display AI Answer with sources
        st.session_state.messages.append({"role": "ai", "content": ai_response, "sources": sources})
        with st.chat_message("ai", avatar="🤖"):
            st.markdown(ai_response)
            
            # Display source citations on the current turn
            if sources:
                st.markdown("<div style='margin-top: 15px; font-size: 0.8rem; color: #94a3b8; font-weight: 600;'>📚 Sources Cited:</div>", unsafe_allow_html=True)
                cols = st.columns(len(sources))
                for idx, src in enumerate(sources):
                    filename = src.get("source", "Unknown Document")
                    pages_list = src.get("pages", [])
                    if len(pages_list) == 1:
                        pages_str = f"Page {pages_list[0]}"
                    elif len(pages_list) > 1:
                        pages_str = "Pages " + ", ".join(map(str, pages_list))
                    else:
                        pages_str = "Page 1"
                        
                    with cols[idx]:
                        st.markdown(
                            f"""
                            <div class="sidebar-card" style="margin-bottom: 0px; padding: 10px; border: 1px solid rgba(99, 102, 241, 0.25);">
                                <div style="color: #a5b4fc; font-weight: 600; font-size: 0.8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{filename}">📄 {filename}</div>
                                <div style="color: #64748b; font-size: 0.75rem; margin-top: 2px;">{pages_str}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

