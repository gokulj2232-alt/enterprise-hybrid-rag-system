import json
import re
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_FILE = PROJECT_ROOT / "data" / "faiss_index.bin"
METADATA_FILE = PROJECT_ROOT / "data" / "chunk_metadata.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded.")

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_FILE))
print(f"FAISS vectors: {index.ntotal:,}")

print("Loading metadata...")
metadata = []

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            metadata.append(json.loads(line))

print(f"Metadata loaded: {len(metadata):,}")

print("Preparing BM25 keyword search...")

documents = [
    item.get("text", "").lower().split()
    for item in metadata
]

bm25 = BM25Okapi(documents)

print("BM25 ready.")


def normalize_text(text):
    text = str(text).lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def hybrid_search(query, top_k=5):

    candidate_k = 20000

    normalized_query = normalize_text(query)
    query_tokens = normalized_query.split()

    # Semantic search
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    semantic_scores, semantic_indices = index.search(
        query_embedding,
        candidate_k
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
                semantic_scores[0, rank - 1]
            )

    # BM25 search
    keyword_scores = bm25.get_scores(query_tokens)

    keyword_indices = np.argsort(
        keyword_scores
    )[::-1][:candidate_k]

    keyword_rank = {}

    for rank, idx in enumerate(
        keyword_indices,
        start=1
    ):
        keyword_rank[int(idx)] = rank

    # Candidate pool
    candidates = set(semantic_rank.keys())
    candidates.update(keyword_rank.keys())

    # Important concept phrases
    important_phrases = [
        "retrieval augmented generation",
        "question answering",
        "semantic search",
        "vector database",
        "large language model",
        "natural language processing"
    ]

    matched_phrases = [
        phrase
        for phrase in important_phrases
        if phrase in normalized_query
    ]

    # Add documents containing important concepts
    if matched_phrases:

        for idx, item in enumerate(metadata):

            text = normalize_text(
                item.get("text", "")
            )

            title = normalize_text(
                item.get("title", "")
            )

            for phrase in matched_phrases:

                if phrase in text or phrase in title:
                    candidates.add(idx)
                    break

    # Detect definition questions
    definition_question = any(
        phrase in normalized_query
        for phrase in [
            "what is",
            "what are",
            "define",
            "definition of",
            "explain"
        ]
    )

    results = []

    for idx in candidates:

        if idx < 0 or idx >= len(metadata):
            continue

        item = metadata[idx]

        text = normalize_text(
            item.get("text", "")
        )

        title = normalize_text(
            item.get("title", "")
        )

        srank = semantic_rank.get(
            idx,
            candidate_k + 1
        )

        krank = keyword_rank.get(
            idx,
            candidate_k + 1
        )

        semantic_score = semantic_score_map.get(
            idx,
            0.0
        )

        keyword_score = float(
            keyword_scores[idx]
        )

        # Reciprocal Rank Fusion
        rrf_score = (
            1.0 / (60 + srank)
            +
            1.0 / (60 + krank)
        )

        # Concept bonus
        concept_bonus = 0.0

        for phrase in matched_phrases:

            if phrase in text:
                concept_bonus += 0.35

            if phrase in title:
                concept_bonus += 0.15

        concept_bonus = min(
            concept_bonus,
            0.75
        )

        # Query word matching
        matching_words = sum(
            1
            for word in query_tokens
            if len(word) > 2 and word in text
        )

        word_bonus = min(
            matching_words * 0.02,
            0.12
        )

        # Definition bonus
        definition_bonus = 0.0

        if definition_question:

            definition_patterns = [
                "is an approach",
                "is a method",
                "is a technique",
                "is the process",
                "is a system",
                "is a framework",
                "is a practice",
                "refers to",
                "means",
                "combines",
                "consists of",
                "is used to"
            ]

            for pattern in definition_patterns:

                if pattern in text:
                    definition_bonus += 0.10

            definition_bonus = min(
                definition_bonus,
                0.50
            )

        # Explanation bonus
        explanation_bonus = 0.0

        explanation_terms = [
            "when a user",
            "the system",
            "knowledge base",
            "retrieved documents",
            "language model",
            "for example",
            "typically",
            "includes",
            "useful when",
            "allows",
            "helps"
        ]

        for term in explanation_terms:

            if term in text:
                explanation_bonus += 0.025

        explanation_bonus = min(
            explanation_bonus,
            0.20
        )

        # Content quality
        text_length = len(text)

        quality_bonus = 0.0

        if 450 <= text_length <= 1000:
            quality_bonus = 0.10

        elif text_length >= 300:
            quality_bonus = 0.05

        # Title bonus
        title_bonus = 0.0

        for word in query_tokens:

            if len(word) > 2 and word in title:
                title_bonus += 0.025

        title_bonus = min(
            title_bonus,
            0.10
        )

        # Final score
        final_score = (
            rrf_score
            + concept_bonus
            + word_bonus
            + definition_bonus
            + explanation_bonus
            + quality_bonus
            + title_bonus
        )

        results.append({
            "index": idx,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "rrf_score": rrf_score,
            "final_score": final_score
        })

    # Sort
    results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    # Remove duplicate content
    seen_content = set()
    final_results = []

    for result in results:

        item = metadata[result["index"]]

        content_key = normalize_text(
            item.get("text", "")
        )

        if content_key in seen_content:
            continue

        seen_content.add(content_key)

        output = item.copy()

        output["semantic_score"] = round(
            result["semantic_score"],
            4
        )

        output["keyword_score"] = round(
            result["keyword_score"],
            4
        )

        output["hybrid_score"] = round(
            result["final_score"],
            4
        )

        final_results.append(output)

        if len(final_results) >= top_k:
            break

    return final_results