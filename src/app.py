import streamlit as st
import requests

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_processing.process_document import process_document


API_URL = "http://127.0.0.1:8000/ask"


st.set_page_config(
    page_title="Enterprise Hybrid RAG",
    page_icon="🤖",
    layout="centered"
)

st.title("Enterprise Hybrid RAG")
st.write("Ask questions about the knowledge base.")

st.subheader("Upload a Document")

uploaded_file = st.file_uploader(
    "Upload TXT, CSV, Excel, or PDF",
    type=["txt", "csv", "xlsx", "pdf"]
)

if uploaded_file is not None:

    if st.button("Add Document to Knowledge Base"):

        upload_dir = PROJECT_ROOT / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / uploaded_file.name

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        with st.spinner("Processing document..."):

            try:
                result = process_document(file_path)

                st.success(
                    f"Document added successfully: "
                    f"{result['file_name']} "
                    f"({result['chunks']} chunks)"
                )

            except Exception as e:
                st.error(f"Document processing failed: {e}")


st.divider()

st.subheader("Ask a Question")

question = st.text_input(
    "Enter your question:",
    placeholder="What is retrieval augmented generation?"
)

if st.button("Ask", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching and generating answer..."):

            try:

                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=120
                )

                if response.status_code == 200:

                    data = response.json()

                    st.subheader("Answer")
                    st.write(data["answer"])

                else:

                    st.error(
                        f"API Error: {response.status_code}\n\n"
                        f"{response.text}"
                    )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Could not connect to the FastAPI server: {e}"
                )