from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:

    def __init__(self, model_name: str = MODEL_NAME):

        print(f"Loading Cross-Encoder reranker: {model_name}")

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

        print("Cross-Encoder reranker loaded successfully!")

    def _extract_text(self, result: Any) -> str:

        if not isinstance(result, dict):
            return str(result)

        if isinstance(result.get("chunk"), dict):

            chunk = result["chunk"]

            return str(
                chunk.get("content")
                or chunk.get("chunk_text")
                or chunk.get("text")
                or ""
            )

        return str(
            result.get("content")
            or result.get("chunk_text")
            or result.get("text")
            or ""
        )

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:

        if not results:
            return []

        query = str(query).strip()

        if not query:
            return results[:top_k]

        pairs = []
        valid_results = []

        for result in results:

            text = self._extract_text(result)

            if not text.strip():
                continue

            pairs.append((query, text))
            valid_results.append(result)

        if not pairs:
            return []

        print(f"Reranking {len(pairs)} candidates...")

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(valid_results, scores):

            item = dict(result)

            item["reranker_score"] = float(score)

            reranked.append(item)

        reranked.sort(
            key=lambda x: x.get("reranker_score", 0.0),
            reverse=True
        )

        return reranked[:top_k]


def main():

    print()
    print("=" * 70)
    print("CROSS-ENCODER RERANKER TEST")
    print("=" * 70)
    print()

    reranker = CrossEncoderReranker()

    query = "cloud computing"

    results = [

        {
            "document_id": "DOC001",
            "category": "Technology",
            "title": "Cloud Computing",
            "content": (
                "Cloud computing provides computing resources "
                "through the internet including storage and processing."
            )
        },

        {
            "document_id": "DOC002",
            "category": "Technology",
            "title": "Machine Learning",
            "content": (
                "Machine learning allows computers to learn "
                "patterns from data."
            )
        }

    ]

    reranked = reranker.rerank(
        query,
        results,
        top_k=2
    )

    print()
    print("RERANKING RESULTS")
    print()

    for index, result in enumerate(reranked, start=1):

        print(f"Result {index}")
        print("Document ID    :", result.get("document_id"))
        print("Category       :", result.get("category"))
        print("Title          :", result.get("title"))
        print(
            "Reranker Score :",
            f"{result.get('reranker_score', 0.0):.4f}"
        )
        print()


if __name__ == "__main__":
    main()