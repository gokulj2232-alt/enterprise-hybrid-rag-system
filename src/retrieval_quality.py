import re


# ============================================================
# GENERIC / TEMPLATE PATTERNS
# ============================================================

GENERIC_PATTERNS = [
    "this document explains",
    "this document contains useful information",
    "this document provides information",
    "it provides information about",
    "the information is intended to help users",
    "users can refer to this document",
    "this document describes",
    "this document contains",
    "frequently used terms",
    "common questions",
    "important features procedures benefits",
    "practical considerations",
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Normalize text for retrieval-quality analysis.
    """

    if not text:
        return ""

    return " ".join(
        str(text).lower().split()
    )


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Convert text into a set of alphanumeric tokens.
    """

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            clean_text(text)
        )
    )


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def keyword_overlap(query, text):
    """
    Calculate query keyword overlap with evidence.
    """

    query_tokens = tokenize(query)
    text_tokens = tokenize(text)

    if not query_tokens:
        return 0.0

    overlap = query_tokens.intersection(
        text_tokens
    )

    return len(overlap) / len(
        query_tokens
    )


# ============================================================
# INFORMATION DENSITY
# ============================================================

def information_density(text):
    """
    Estimate vocabulary diversity and information density.
    """

    text = clean_text(text)

    if not text:
        return 0.0

    words = text.split()

    if len(words) < 5:
        return 0.1

    unique_ratio = (
        len(set(words)) /
        len(words)
    )

    return min(
        1.0,
        unique_ratio
    )


# ============================================================
# GENERIC PENALTY
# ============================================================

def generic_penalty(text):
    """
    Detect generic/template language.

    0.0 = no obvious template language
    1.0 = strong template language
    """

    text = clean_text(text)

    if not text:
        return 1.0

    matches = 0

    for pattern in GENERIC_PATTERNS:

        if pattern in text:
            matches += 1

    if matches == 0:
        return 0.0

    return min(
        1.0,
        matches / 3
    )


# ============================================================
# LOW-INFORMATION DETECTOR
# ============================================================

def is_low_information(text):
    """
    Detect chunks that contain mostly generic/template
    content instead of useful factual evidence.
    """

    text = clean_text(text)

    # Empty content
    if not text:
        return True

    words = text.split()

    # Very short content
    if len(words) <= 8:
        return True

    penalty = generic_penalty(
        text
    )

    density = information_density(
        text
    )

    # Strong template signal
    if penalty >= 0.66:
        return True

    # Very low vocabulary diversity
    if len(words) >= 15 and density < 0.35:
        return True

    return False


# ============================================================
# EVIDENCE QUALITY SCORE
# ============================================================

def calculate_quality_score(
    query,
    result
):
    """
    Calculate evidence quality.

    Components:

    35% CrossEncoder relevance
    30% Keyword overlap
    25% Information density
    -30% Generic/template penalty
    """

    title = result.get(
        "title",
        ""
    )

    content = result.get(
        "chunk_text",
        ""
    )

    combined_text = (
        f"{title} {content}"
    )

    # Keyword relevance
    overlap = keyword_overlap(
        query,
        combined_text
    )

    # Information density
    density = information_density(
        content
    )

    # Generic/template penalty
    penalty = generic_penalty(
        content
    )

    # Low-information status
    low_information = is_low_information(
        content
    )

    # CrossEncoder score
    reranker_score = float(
        result.get(
            "reranker_score",
            0.0
        )
    )

    # Normalize CrossEncoder score
    reranker_normalized = (
        1.0 /
        (
            1.0 +
            pow(
                2.71828,
                -reranker_score
            )
        )
    )

    # Final quality score
    quality_score = (
        0.35 * reranker_normalized
        + 0.30 * overlap
        + 0.25 * density
        - 0.30 * penalty
    )

    # Strongly reduce template evidence
    if low_information:

        quality_score *= 0.35

    return max(
        0.0,
        quality_score
    )


# ============================================================
# RANK BY EVIDENCE QUALITY
# ============================================================

def rank_by_evidence_quality(
    query,
    results
):
    """
    Add quality scores and sort results.
    """

    scored_results = []

    for result in results:

        score = calculate_quality_score(
            query,
            result
        )

        updated_result = dict(
            result
        )

        updated_result[
            "quality_score"
        ] = score

        updated_result[
            "low_information"
        ] = is_low_information(
            result.get(
                "chunk_text",
                ""
            )
        )

        scored_results.append(
            updated_result
        )

    scored_results.sort(
        key=lambda x:
            x["quality_score"],
        reverse=True
    )

    return scored_results


# ============================================================
# FILTER LOW-INFORMATION RESULTS
# ============================================================

def filter_low_information(
    results
):
    """
    Remove generic/template evidence.

    Important:
    This function is used by hybrid_search.py.
    """

    filtered = []

    for result in results:

        content = result.get(
            "chunk_text",
            ""
        )

        if not is_low_information(
            content
        ):

            filtered.append(
                result
            )

    return filtered


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RETRIEVAL QUALITY TEST")
    print("=" * 60)

    test_results = [

        {
            "document_id": "DOC001",
            "title": "Cloud Computing Resources",
            "chunk_text": (
                "Cloud computing provides "
                "computing resources over the internet "
                "including storage and processing "
                "resources."
            ),
            "reranker_score": 8.2
        },

        {
            "document_id": "DOC002",
            "title": "Cloud Computing",
            "chunk_text": (
                "This document explains cloud computing. "
                "It provides information about the topic."
            ),
            "reranker_score": 8.1
        },

        {
            "document_id": "DOC003",
            "title": "Cloud Computing",
            "chunk_text": (
                "cloud computing"
            ),
            "reranker_score": 7.9
        }
    ]

    query = "What is cloud computing?"

    print("\nOriginal Results:")

    for result in test_results:

        print(
            f"\nDocument ID: "
            f"{result['document_id']}"
        )

        print(
            f"Title: "
            f"{result['title']}"
        )

        print(
            f"Content: "
            f"{result['chunk_text']}"
        )

    # --------------------------------------------------------
    # Low-information detection
    # --------------------------------------------------------

    print("\n" + "-" * 60)

    print(
        "\nLOW-INFORMATION DETECTION"
    )

    print("-" * 60)

    for result in test_results:

        status = is_low_information(
            result["chunk_text"]
        )

        print(
            f"{result['document_id']} "
            f"-> Low Information: {status}"
        )

    # --------------------------------------------------------
    # Quality ranking
    # --------------------------------------------------------

    print("\n" + "-" * 60)

    ranked_results = rank_by_evidence_quality(
        query,
        test_results
    )

    print(
        "\nEVIDENCE QUALITY RANKING"
    )

    print("-" * 60)

    for result in ranked_results:

        print(
            f"{result['document_id']} "
            f"| Quality Score: "
            f"{result['quality_score']:.4f} "
            f"| Low Information: "
            f"{result['low_information']}"
        )

    # --------------------------------------------------------
    # Filtering
    # --------------------------------------------------------

    print("\n" + "-" * 60)

    filtered_results = filter_low_information(
        test_results
    )

    print(
        "\nAFTER LOW-INFORMATION FILTERING"
    )

    print("-" * 60)

    for result in filtered_results:

        print(
            f"{result['document_id']} "
            f"| {result['title']}"
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "RETRIEVAL QUALITY TEST COMPLETED"
    )

    print("=" * 60)