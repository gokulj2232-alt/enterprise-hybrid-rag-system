import os
import pickle

from rank_bm25 import BM25Okapi


# ============================================================
# DEFAULT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DEFAULT_BM25_PATH = os.path.join(
    MODELS_DIR,
    "bm25.pkl"
)

DEFAULT_CHUNKS_PATH = os.path.join(
    MODELS_DIR,
    "processed_chunks.pkl"
)


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Simple BM25 tokenizer.
    """

    if text is None:
        return []

    text = str(text).lower()

    return text.split()


# ============================================================
# BM25 SEARCH CLASS
# ============================================================

class BM25Search:

    def __init__(
        self,
        bm25_path=DEFAULT_BM25_PATH,
        chunks_path=DEFAULT_CHUNKS_PATH
    ):

        self.bm25_path = bm25_path
        self.chunks_path = chunks_path

        print()
        print(
            "Loading BM25 index..."
        )

        # ----------------------------------------------------
        # Load BM25 index
        # ----------------------------------------------------

        with open(
            self.bm25_path,
            "rb"
        ) as file:

            self.bm25 = pickle.load(file)

        # ----------------------------------------------------
        # Load processed chunks
        # ----------------------------------------------------

        with open(
            self.chunks_path,
            "rb"
        ) as file:

            self.chunks = pickle.load(file)

        print(
            f"BM25 chunks loaded: "
            f"{len(self.chunks):,}"
        )


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=20
    ):
        """
        Search documents using BM25.

        Returns:
            [
                {
                    "chunk": {...},
                    "bm25_score": float
                }
            ]
        """

        if not query:
            return []

        # ----------------------------------------------------
        # Tokenize query
        # ----------------------------------------------------

        query_tokens = tokenize(
            query
        )

        if not query_tokens:
            return []

        # ----------------------------------------------------
        # BM25 scores
        # ----------------------------------------------------

        scores = self.bm25.get_scores(
            query_tokens
        )

        # ----------------------------------------------------
        # Top indices
        # ----------------------------------------------------

        top_indices = scores.argsort()[::-1][
            :top_k
        ]

        results = []

        for index in top_indices:

            index = int(index)

            if index < 0:
                continue

            if index >= len(self.chunks):
                continue

            chunk = dict(
                self.chunks[index]
            )

            # ------------------------------------------------
            # Normalize chunk_text -> content
            #
            # This makes BM25 output compatible with
            # hybrid_search.py, reranker.py,
            # retrieval_quality.py and RAG.
            # ------------------------------------------------

            if not chunk.get("content"):

                chunk["content"] = (
                    chunk.get(
                        "chunk_text",
                        ""
                    )
                )

            # ------------------------------------------------
            # Add result
            # ------------------------------------------------

            results.append(
                {
                    "chunk": chunk,
                    "bm25_score": float(
                        scores[index]
                    )
                }
            )

        return results


# ============================================================
# STANDALONE TEST
# ============================================================

def main():

    print()
    print("=" * 60)
    print("BM25 KEYWORD SEARCH TEST")
    print("=" * 60)

    try:

        searcher = BM25Search()

    except Exception as error:

        print()
        print(
            "BM25 initialization failed:"
        )

        print(error)

        return

    print()
    print(
        "[1/1] Enter your search query"
    )

    query = input(
        "\nQuery: "
    ).strip()

    if not query:

        print(
            "Empty query."
        )

        return

    results = searcher.search(
        query,
        top_k=5
    )

    print()
    print("=" * 60)
    print("BM25 SEARCH RESULTS")
    print("=" * 60)

    if not results:

        print(
            "No results found."
        )

        return

    for i, result in enumerate(
        results,
        start=1
    ):

        chunk = result.get(
            "chunk",
            {}
        )

        print()
        print(
            f"Result {i}"
        )

        print(
            "-" * 40
        )

        print(
            f"Document ID : "
            f"{chunk.get('document_id', 'N/A')}"
        )

        print(
            f"Category    : "
            f"{chunk.get('category', 'N/A')}"
        )

        print(
            f"Title       : "
            f"{chunk.get('title', 'N/A')}"
        )

        print(
            f"BM25 Score  : "
            f"{result.get('bm25_score', 0.0):.4f}"
        )

        print()
        print("Content:")

        content = (
            chunk.get("content")
            or chunk.get("chunk_text")
            or ""
        )

        print(
            str(content)[:500]
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()