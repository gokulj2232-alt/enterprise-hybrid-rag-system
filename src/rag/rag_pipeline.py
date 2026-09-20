import sys

sys.path.insert(0, ".")

from src.search.query_processor import process_query
from src.search.multi_query import multi_query_search
from src.search.reranker import rerank
from src.search.relevance_filter import filter_relevant_results
from src.rag.context_compressor import compress_context
from src.rag.rag_generator import RAGGenerator


rag_generator = RAGGenerator()


def build_conversation_context(conversation):
    """
    Build a compact conversation context from recent messages.
    """

    if not conversation:
        return ""

    context_parts = []

    for message in conversation[-6:]:
        role = message.get("role", "").strip()
        content = message.get("content", "").strip()

        if role and content:
            context_parts.append(
                f"{role.capitalize()}: {content}"
            )

    return "\n".join(context_parts)


def build_contextual_query(query, conversation):
    """
    Combine recent conversation with the current question.
    """

    conversation_context = build_conversation_context(
        conversation
    )

    if not conversation_context:
        return query

    return (
        f"Previous conversation:\n"
        f"{conversation_context}\n\n"
        f"Current question:\n"
        f"{query}"
    )


def rag_pipeline(query, top_k=5, conversation=None):
    """
    Complete Enterprise Hybrid RAG Pipeline.

    Flow:
    1. Conversation context
    2. Query processing
    3. Multi-query retrieval
    4. Cross-encoder reranking
    5. Relevance filtering
    6. Context compression
    7. RAG answer generation
    8. Source/evidence collection
    """

    if conversation is None:
        conversation = []

    # ---------------------------------------------------------
    # STEP 1: BUILD CONTEXTUAL QUERY
    # ---------------------------------------------------------

    contextual_query = build_contextual_query(
        query,
        conversation
    )

    # ---------------------------------------------------------
    # STEP 2: PROCESS QUERY
    # ---------------------------------------------------------

    processed = process_query(
        contextual_query
    )

    expanded_query = processed["expanded_query"]

    # ---------------------------------------------------------
    # STEP 3: MULTI-QUERY RETRIEVAL
    # ---------------------------------------------------------

    retrieved_chunks = multi_query_search(
        expanded_query,
        top_k=20
    )

    if not retrieved_chunks:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": []
        }

    # ---------------------------------------------------------
    # STEP 4: CROSS-ENCODER RERANKING
    # ---------------------------------------------------------

    reranked_chunks = rerank(
        expanded_query,
        retrieved_chunks,
        top_k=top_k
    )

    if not reranked_chunks:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": []
        }

    # ---------------------------------------------------------
    # STEP 5: RELEVANCE FILTERING
    # ---------------------------------------------------------

    filtered_chunks = filter_relevant_results(
        reranked_chunks,
        min_score=-5.0,
        min_results=1
    )

    if not filtered_chunks:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": []
        }

    # ---------------------------------------------------------
    # STEP 6: CONTEXT COMPRESSION
    # ---------------------------------------------------------

    compressed_context = compress_context(
        filtered_chunks
    )

    # ---------------------------------------------------------
    # STEP 7: RAG ANSWER GENERATION
    # ---------------------------------------------------------

    # Use the contextual query so the generator also
    # understands references such as "they", "it", "this", etc.
    result = rag_generator.generate(
        contextual_query,
        compressed_context
    )

    # ---------------------------------------------------------
    # STEP 8: SOURCE INFORMATION
    # ---------------------------------------------------------

    sources = []

    for rank, chunk in enumerate(
        filtered_chunks,
        start=1
    ):
        sources.append({
            "rank": rank,
            "document_id": chunk.get(
                "document_id",
                ""
            ),
            "category": chunk.get(
                "category",
                ""
            ),
            "title": chunk.get(
                "title",
                ""
            ),
            "reranker_score": chunk.get(
                "reranker_score",
                0
            ),
            "text": chunk.get(
                "text",
                ""
            )
        })

    return {
        "answer": result.get(
            "answer",
            "I don't know based on the provided documents."
        ),
        "sources": sources
    }


if __name__ == "__main__":

    print("=" * 70)
    print("ENTERPRISE HYBRID RAG PIPELINE TEST")
    print("=" * 70)

    query = input(
        "\nEnter your question: "
    )

    result = rag_pipeline(query)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in result["sources"]:

        print(
            f"\n[{source['rank']}] "
            f"{source['document_id']} | "
            f"{source['category']} | "
            f"{source['title']}"
        )

        print(
            f"Reranker Score: "
            f"{source['reranker_score']}"
        )

    print("\n" + "=" * 70)