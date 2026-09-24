import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# UPLOADED INDEX FILES
# ============================================================

INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "uploads"
    / "index"
    / "faiss_index.bin"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "uploads"
    / "index"
    / "metadata.json"
)

BM25_FILE = (
    PROJECT_ROOT
    / "data"
    / "uploads"
    / "index"
    / "bm25.json"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = text.replace(
        "-",
        " "
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# EXTRACT FIELDS FROM DOCUMENT CONTENT
# ============================================================

def extract_fields(content):
    """
    Convert:

        product_name: Laptop Pro 15 |
        category: Electronics |
        description: High performance laptop

    into:

        {
            "product_name": "Laptop Pro 15",
            "category": "Electronics",
            "description": "High performance laptop"
        }
    """

    fields = {}

    if not content:
        return fields

    parts = str(content).split("|")

    for part in parts:

        part = part.strip()

        if ":" not in part:
            continue

        key, value = part.split(
            ":",
            1
        )

        key = key.strip().lower()

        value = value.strip()

        if key and value:
            fields[key] = value

    return fields


# ============================================================
# EXTRACT TITLE
# ============================================================

def extract_title(content):

    fields = extract_fields(
        content
    )

    possible_title_fields = [

        "title",
        "product_name",
        "product title",
        "name",
        "item_name",
        "item name",
        "document_title",
        "document title",
        "article_title",
        "article title"
    ]

    for field in possible_title_fields:

        if field in fields:

            return fields[field]

    return ""


# ============================================================
# EXTRACT CATEGORY
# ============================================================

def extract_category(content):

    fields = extract_fields(
        content
    )

    possible_category_fields = [

        "category",
        "type",
        "class",
        "classification",
        "department",
        "topic"
    ]

    for field in possible_category_fields:

        if field in fields:

            return fields[field]

    return ""


# ============================================================
# EXTRACT DESCRIPTION
# ============================================================

def extract_description(content):

    fields = extract_fields(
        content
    )

    possible_description_fields = [

        "description",
        "summary",
        "body",
        "article",
        "text"
    ]

    for field in possible_description_fields:

        if field in fields:

            return fields[field]

    return ""


# ============================================================
# LOAD UPLOADED INDEX
# ============================================================

def load_uploaded_index():

    if not INDEX_FILE.exists():

        raise FileNotFoundError(
            "Uploaded FAISS index not found. "
            "Process a dataset first."
        )

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            "Uploaded metadata not found."
        )

    print(
        "Loading uploaded FAISS index..."
    )

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    documents = [

        item.get(
            "content",
            ""
        ).lower().split()

        for item in metadata
    ]

    bm25 = BM25Okapi(
        documents
    )

    print(
        f"Uploaded documents: "
        f"{len(metadata)}"
    )

    return (
        index,
        metadata,
        bm25
    )


# ============================================================
# UPLOADED HYBRID SEARCH
# ============================================================

def uploaded_hybrid_search(
    query,
    top_k=5
):

    # --------------------------------------------------------
    # LOAD INDEX
    # --------------------------------------------------------

    index, metadata, bm25 = (
        load_uploaded_index()
    )

    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    model = SentenceTransformer(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # NORMALIZE QUERY
    # --------------------------------------------------------

    normalized_query = normalize_text(
        query
    )

    query_tokens = (
        normalized_query.split()
    )

    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # --------------------------------------------------------
    # CANDIDATE COUNT
    # --------------------------------------------------------

    candidate_k = min(
        200,
        len(metadata)
    )

    # --------------------------------------------------------
    # FAISS SEMANTIC SEARCH
    # --------------------------------------------------------

    semantic_scores, semantic_indices = (
        index.search(
            query_embedding,
            candidate_k
        )
    )

    semantic_rank = {}

    semantic_score_map = {}

    for rank, idx in enumerate(
        semantic_indices[0],
        start=1
    ):

        if idx >= 0:

            idx = int(idx)

            semantic_rank[idx] = rank

            semantic_score_map[idx] = float(
                semantic_scores[0][rank - 1]
            )

    # --------------------------------------------------------
    # BM25 KEYWORD SEARCH
    # --------------------------------------------------------

    keyword_scores = bm25.get_scores(
        query_tokens
    )

    keyword_indices = np.argsort(
        keyword_scores
    )[::-1][:candidate_k]

    keyword_rank = {}

    for rank, idx in enumerate(
        keyword_indices,
        start=1
    ):

        keyword_rank[int(idx)] = rank

    # --------------------------------------------------------
    # COMBINE CANDIDATES
    # --------------------------------------------------------

    candidates = set(
        semantic_rank.keys()
    )

    candidates.update(
        keyword_rank.keys()
    )

    # --------------------------------------------------------
    # BUILD RESULTS
    # --------------------------------------------------------

    results = []

    for idx in candidates:

        if (
            idx < 0
            or idx >= len(metadata)
        ):
            continue

        item = metadata[idx]

        content = item.get(
            "content",
            ""
        )

        text = normalize_text(
            content
        )

        # ----------------------------------------------------
        # RANKS
        # ----------------------------------------------------

        semantic_rank_value = (
            semantic_rank.get(
                idx,
                candidate_k + 1
            )
        )

        keyword_rank_value = (
            keyword_rank.get(
                idx,
                candidate_k + 1
            )
        )

        # ----------------------------------------------------
        # SCORES
        # ----------------------------------------------------

        semantic_score = (
            semantic_score_map.get(
                idx,
                0.0
            )
        )

        keyword_score = float(
            keyword_scores[idx]
        )

        # ----------------------------------------------------
        # RECIPROCAL RANK FUSION
        # ----------------------------------------------------

        rrf_score = (

            1.0 / (
                60 + semantic_rank_value
            )

            +

            1.0 / (
                60 + keyword_rank_value
            )
        )

        # ----------------------------------------------------
        # MATCHING WORD BONUS
        # ----------------------------------------------------

        matching_words = sum(

            1

            for word in query_tokens

            if (
                len(word) > 2
                and word in text
            )
        )

        word_bonus = min(
            matching_words * 0.02,
            0.12
        )

        final_score = (
            rrf_score
            + word_bonus
        )

        # ----------------------------------------------------
        # EXTRACT METADATA
        # ----------------------------------------------------

        title = extract_title(
            content
        )

        category = extract_category(
            content
        )

        description = extract_description(
            content
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        results.append({

            "document_id": item.get(
                "document_id",
                f"UPLOAD_{idx + 1:06d}"
            ),

            "title": title,

            "category": category,

            "description": description,

            "text": content,

            "source_row": item.get(
                "source_row"
            ),

            "semantic_score": round(
                semantic_score,
                4
            ),

            "keyword_score": round(
                keyword_score,
                4
            ),

            "hybrid_score": round(
                final_score,
                4
            )
        })

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    results.sort(
        key=lambda item:
            item["hybrid_score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "UPLOADED HYBRID SEARCH TEST"
    )

    print("=" * 70)

    query = input(
        "\nEnter your question: "
    )

    results = uploaded_hybrid_search(
        query,
        top_k=5
    )

    print(
        "\n" + "=" * 70
    )

    print("RESULTS")

    print(
        "=" * 70
    )

    for result in results:

        print(
            f"\nDocument ID : "
            f"{result['document_id']}"
        )

        print(
            f"Title       : "
            f"{result['title']}"
        )

        print(
            f"Category    : "
            f"{result['category']}"
        )

        print(
            f"Semantic    : "
            f"{result['semantic_score']}"
        )

        print(
            f"Keyword     : "
            f"{result['keyword_score']}"
        )

        print(
            f"Hybrid      : "
            f"{result['hybrid_score']}"
        )

        print(
            f"Text        : "
            f"{result['text']}"
        )

    print(
        "\n" + "=" * 70
    )