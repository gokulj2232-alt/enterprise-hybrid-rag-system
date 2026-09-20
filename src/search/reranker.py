from sentence_transformers import CrossEncoder


# ==================================================
# RERANKER MODEL
# ==================================================

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print("Loading reranker model...")

reranker_model = CrossEncoder(MODEL_NAME)

print("Reranker model loaded.")


# ==================================================
# RERANK FUNCTION
# ==================================================

def rerank(query, results, top_k=5):

    if not results:
        return []

    pairs = []

    for result in results:

        title = result.get("title", "")
        text = result.get("text", "")

        document_text = (
            f"Title: {title}\n"
            f"Content: {text}"
        )

        pairs.append(
            [query, document_text]
        )

    # Calculate Cross-Encoder scores
    scores = reranker_model.predict(pairs)

    reranked_results = []

    for result, score in zip(results, scores):

        result_copy = result.copy()

        result_copy["reranker_score"] = round(
            float(score),
            4
        )

        reranked_results.append(result_copy)

    # Sort by reranker score
    reranked_results.sort(
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    return reranked_results[:top_k]


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    test_query = (
        "What is retrieval augmented generation?"
    )

    test_results = [

        {
            "title": "Retrieval Augmented Generation",
            "text": (
                "Retrieval Augmented Generation combines "
                "information retrieval with text generation."
            )
        },

        {
            "title": "Python Exception Handling",
            "text": (
                "Python provides mechanisms for "
                "handling runtime errors."
            )
        }
    ]

    print()
    print("=" * 60)
    print("RERANKER TEST")
    print("=" * 60)

    results = rerank(
        test_query,
        test_results,
        top_k=2
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print()
        print(f"Result #{rank}")
        print(
            f"Reranker Score: "
            f"{result['reranker_score']}"
        )
        print(
            f"Title: {result['title']}"
        )