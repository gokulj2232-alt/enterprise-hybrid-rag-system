import streamlit as st
import requests
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.document_processing.process_document import process_document
# ============================================================
# FASTAPI URL
# ============================================================
API_URL = "http://api:8000/ask"
# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Enterprise Hybrid RAG",
    page_icon="🤖",
    layout="centered"
)
# ============================================================
# SESSION STATE
# ============================================================
if "uploaded_dataset" not in st.session_state:
    st.session_state.uploaded_dataset = False
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
# ============================================================
# HEADER
# ============================================================
st.title("Enterprise Hybrid RAG")
st.write(
    "Ask questions about the enterprise knowledge base "
    "or an uploaded document."
)
# ============================================================
# DOCUMENT UPLOAD
# ============================================================
st.subheader("Upload a Document")
uploaded_file = st.file_uploader(
    "Upload TXT, CSV, Excel, or PDF",
    type=[
        "txt",
        "csv",
        "xlsx",
        "pdf"
    ]
)
if uploaded_file is not None:
    if st.button("Add Document to Knowledge Base"):
        upload_dir = (
            PROJECT_ROOT
            / "data"
            / "uploads"
        )
        upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        file_path = (
            upload_dir
            / uploaded_file.name
        )
        with open(
            file_path,
            "wb"
        ) as file:
            file.write(
                uploaded_file.getbuffer()
            )
        with st.spinner(
            "Processing document and building uploaded index..."
        ):
            try:
                result = process_document(
                    file_path
                )
                # IMPORTANT:
                # Tell the question-answering system
                # to use the uploaded knowledge base.
                st.session_state.uploaded_dataset = True
                st.session_state.uploaded_file_name = (
                    result["file_name"]
                )
                st.success(
                    f"Document added successfully: "
                    f"{result['file_name']} "
                    f"({result['chunks']} chunks)"
                )
                st.info(
                    "Questions will now search the uploaded "
                    "document index."
                )
            except Exception as e:
                st.error(
                    f"Document processing failed: {e}"
                )
# ============================================================
# CURRENT SEARCH MODE
# ============================================================
st.divider()
if st.session_state.uploaded_dataset:
    st.success(
        f"Search mode: Uploaded document "
        f"({st.session_state.uploaded_file_name})"
    )
    if st.button("Use Main Knowledge Base Instead"):
        st.session_state.uploaded_dataset = False
        st.session_state.uploaded_file_name = None
        st.rerun()
else:
    st.info(
        "Search mode: Main Enterprise Knowledge Base"
    )
# ============================================================
# QUESTION ANSWERING
# ============================================================
st.subheader("Ask a Question")
question = st.text_input(
    "Enter your question:",
    placeholder=(
        "What is retrieval augmented generation?"
    )
)
if st.button(
    "Ask",
    type="primary"
):
    if not question.strip():
        st.warning(
            "Please enter a question."
        )
    else:
        with st.spinner(
            "Searching and generating answer..."
        ):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "question": question,
                        "uploaded_dataset": (
                            st.session_state.uploaded_dataset
                        )
                    },
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("Answer")
                    st.write(
                        data["answer"]
                    )
                    # ------------------------------------------------
                    # SOURCES
                    # ------------------------------------------------
                    sources = data.get(
                        "sources",
                        []
                    )
                    if sources:
                        st.subheader("Sources")
                        for source in sources:
                            title = source.get(
                                "title",
                                "Untitled"
                            )
                            document_id = source.get(
                                "document_id",
                                "Unknown"
                            )
                            st.write(
                                f"• {title} "
                                f"({document_id})"
                            )
                else:
                    st.error(
                        f"API Error: "
                        f"{response.status_code}\n\n"
                        f"{response.text}"
                    )
            except requests.exceptions.RequestException as e:
                st.error(
                    "Could not connect to the "
                    f"FastAPI server: {e}"
                )
