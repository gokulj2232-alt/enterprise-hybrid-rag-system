import streamlit as st
import requests
import time

st.set_page_config(
    page_title="Enterprise Hybrid RAG",
    page_icon="🔎",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #666;
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

API_URL = "http://127.0.0.1:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    '<div class="main-title">AI-Powered Enterprise Hybrid RAG System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Semantic Search + BM25 + Hybrid Retrieval + Reranking + RAG</div>',
    unsafe_allow_html=True
)

with st.sidebar:

    st.header("System")

    try:
        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        if health_response.status_code == 200:
            health_data = health_response.json()

            if health_data.get("rag_engine_loaded"):
                st.success("RAG API Ready")
            else:
                st.warning("RAG Engine Loading")
        else:
            st.error("RAG API Error")

    except requests.exceptions.RequestException:
        st.error("RAG API Offline")

    st.markdown("### Features")

    st.write("• 50,000 documents")
    st.write("• Semantic Search")
    st.write("• BM25 Search")
    st.write("• Hybrid Retrieval")
    st.write("• Cross-Encoder Reranking")
    st.write("• Metadata Filtering")
    st.write("• Context Optimization")
    st.write("• RAG Generation")
    st.write("• Answerability Gate")
    st.write("• Grounding Validation")
    st.write("• RAG Evaluation")
    st.write("• Conversation Memory")

    st.divider()

    top_k = st.slider(
        "Retrieval Results",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input(
    "Ask a question about the 50,000 documents..."
)

if query:

    query = query.strip()

    if not query:
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        start_time = time.time()

        try:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": query,
                        "top_k": top_k
                    },
                    timeout=300
                )

            processing_time = round(
                time.time() - start_time,
                4
            )

            if response.status_code == 200:

                result = response.json()

                answer = result.get(
                    "answer",
                    "I don't know based on the provided documents."
                )

                st.markdown(
                    f"""
                    <div class="answer-box">
                    {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    f"Processing time: {processing_time} seconds"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

                with st.expander("View RAG Details"):

                    st.write("Question:", query)
                    st.write("Top-K:", top_k)

                    st.write(
                        "API Processing Time:",
                        result.get("processing_time_seconds")
                    )

                    st.json(result)

            else:

                try:
                    error_data = response.json()

                    error_message = error_data.get(
                        "detail",
                        "Unknown API error."
                    )

                except Exception:
                    error_message = response.text

                st.error(
                    f"RAG API Error: {error_message}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to the RAG API.\n\n"
                "Please make sure api.py is running."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The RAG API took too long to respond."
            )

        except Exception as e:

            st.error(
                f"Unexpected error: {str(e)}"
            )