from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name=MODEL_NAME):

        print(f"Loading reranker model: {model_name}")

        self.model = CrossEncoder(model_name)

        print("Reranker model loaded successfully!")

    def rerank(self, query, results, top_k=5):

        if not results:
            return []

        pairs = []

        for result in results:

            text = (
                f"{result.get('title', '')}. "
                f"{result.get('chunk_text', '')}"
            )

            pairs.append([query, text])

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):

            result = result.copy()

            result["reranker_score"] = float(score)

            reranked_results.append(result)

        reranked_results.sort(
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return reranked_results[:top_k]


if __name__ == "__main__":

    print("=" * 60)
    print("CROSS-ENCODER RERANKER TEST")
    print("=" * 60)

    reranker = Reranker()

    test_query = "What is cloud computing?"

    test_results = [
        {
            "document_id": "DOC001",
            "title": "Cloud Computing",
            "chunk_text": (
                "Cloud computing provides computing resources "
                "such as storage and processing over the internet."
            )
        },
        {
            "document_id": "DOC002",
            "title": "Machine Learning",
            "chunk_text": (
                "Machine learning allows computers to learn "
                "patterns from data."
            )
        }
    ]

    results = reranker.rerank(
        test_query,
        test_results,
        top_k=2
    )

    print("\nReranking results:")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 40)

        print(
            f"Document ID : "
            f"{result['document_id']}"
        )

        print(
            f"Title       : "
            f"{result['title']}"
        )

        print(
            f"Reranker    : "
            f"{result['reranker_score']:.4f}"
        )