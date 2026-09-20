import re


def clean_context_text(text):
    """
    Clean text before sending it to the LLM.
    """

    if not text:
        return ""

    text = str(text).strip()

    # Remove excessive whitespace
    text = " ".join(text.split())

    return text


def split_sentences(text):
    """
    Split document text into reasonably sized sentences.
    """

    text = clean_context_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 10:
            continue

        cleaned.append(sentence)

    return cleaned


def tokenize(text):
    """
    Convert text into normalized tokens.
    """

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        str(text).lower()
    )

    return set(words)


def sentence_relevance(sentence, query):
    """
    Calculate simple lexical relevance between
    the query and a sentence.
    """

    if not query:
        return 0.0

    query_tokens = tokenize(query)
    sentence_tokens = tokenize(sentence)

    if not query_tokens or not sentence_tokens:
        return 0.0

    overlap = (
        query_tokens.intersection(
            sentence_tokens
        )
    )

    return len(overlap) / len(query_tokens)


def is_generic_sentence(sentence):
    """
    Detect weak/generic document sentences that
    usually provide little useful evidence.
    """

    text = sentence.lower().strip()

    generic_patterns = [

        "this document explains",

        "this document contains",

        "this document provides",

        "this document describes",

        "this document covers",

        "this document discusses",

        "it provides information",

        "it contains information",

        "it describes important",

        "the information is intended",

        "users can refer to this document",

        "keywords:",

        "frequently used terms",

        "common questions"
    ]

    for pattern in generic_patterns:

        if pattern in text:
            return True

    return False


def information_density(sentence):
    """
    Estimate how much useful information a sentence contains.
    """

    tokens = tokenize(sentence)

    if not tokens:
        return 0.0

    # Longer informative sentences receive a small bonus.
    length_score = min(
        len(sentence) / 150,
        1.0
    )

    # Unique vocabulary bonus.
    vocabulary_score = min(
        len(tokens) / 25,
        1.0
    )

    return (
        0.5 * length_score
        + 0.5 * vocabulary_score
    )


def score_sentence(sentence, query=None):
    """
    Combined relevance score.
    """

    lexical_score = sentence_relevance(
        sentence,
        query
    )

    density_score = information_density(
        sentence
    )

    generic_penalty = (
        0.35
        if is_generic_sentence(sentence)
        else 0.0
    )

    score = (
        0.65 * lexical_score
        + 0.35 * density_score
        - generic_penalty
    )

    return max(
        0.0,
        score
    )


def optimize_chunk(
    chunk_text,
    query=None,
    max_sentences=5
):
    """
    Select the most useful sentences from a chunk.

    Generic sentences are deprioritized.
    Query-relevant sentences are prioritized.
    """

    sentences = split_sentences(
        chunk_text
    )

    if not sentences:
        return ""

    scored_sentences = []

    for index, sentence in enumerate(
        sentences
    ):

        score = score_sentence(
            sentence,
            query
        )

        scored_sentences.append(
            (
                score,
                index,
                sentence
            )
        )

    # Highest scoring sentences first
    scored_sentences.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = (
        scored_sentences[:max_sentences]
    )

    # Restore original document order
    selected.sort(
        key=lambda item: item[1]
    )

    return " ".join(
        item[2]
        for item in selected
    )


def build_context(
    results,
    max_results=5,
    max_characters=6000,
    query=None
):
    """
    Build optimized context from reranked results.

    Features:
    - Query-aware sentence selection
    - Generic sentence reduction
    - Information-density scoring
    - Character limit
    - Source metadata preservation
    """

    if not results:
        return ""

    context_parts = []
    total_characters = 0

    for rank, result in enumerate(
        results[:max_results],
        start=1
    ):

        title = clean_context_text(
            result.get(
                "title",
                ""
            )
        )

        category = clean_context_text(
            result.get(
                "category",
                ""
            )
        )

        document_id = clean_context_text(
            result.get(
                "document_id",
                ""
            )
        )

        chunk_id = clean_context_text(
            result.get(
                "chunk_id",
                ""
            )
        )

        raw_content = clean_context_text(
            result.get(
                "chunk_text",
                ""
            )
        )

        # Optimize the actual document content
        content = optimize_chunk(
            raw_content,
            query=query,
            max_sentences=5
        )

        if not content:
            continue

        context_block = (
            f"[Source {rank}]\n"
            f"Document ID: {document_id}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Category: {category}\n"
            f"Title: {title}\n"
            f"Content: {content}\n"
        )

        if (
            total_characters
            + len(context_block)
            > max_characters
        ):
            break

        context_parts.append(
            context_block
        )

        total_characters += len(
            context_block
        )

    return "\n".join(
        context_parts
    )


def build_source_citations(results):
    """
    Create source information for the final answer.
    """

    citations = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        citations.append({

            "source": f"Source {rank}",

            "document_id": result.get(
                "document_id",
                ""
            ),

            "chunk_id": result.get(
                "chunk_id",
                ""
            ),

            "title": result.get(
                "title",
                ""
            ),

            "category": result.get(
                "category",
                ""
            )
        })

    return citations


if __name__ == "__main__":

    print("=" * 60)
    print("ADVANCED CONTEXT OPTIMIZER TEST")
    print("=" * 60)

    test_results = [

        {
            "document_id": "DOC001",
            "chunk_id": "DOC001_CHUNK_001",
            "category": "Technology",
            "title": "Cloud Computing",

            "chunk_text": (
                "cloud computing. "
                "This document explains cloud computing. "
                "Cloud computing provides computing "
                "resources over the internet. "
                "It provides information about the topic. "
                "Cloud services can provide scalable "
                "computing resources to users."
            ),

            "reranker_score": 8.5
        },

        {
            "document_id": "DOC002",
            "chunk_id": "DOC002_CHUNK_001",
            "category": "Technology",
            "title": "Cloud Infrastructure",

            "chunk_text": (
                "This document contains useful "
                "information about cloud infrastructure. "
                "Cloud infrastructure includes servers, "
                "storage, and networking. "
                "These components support cloud-based "
                "applications and services."
            ),

            "reranker_score": 7.8
        }
    ]

    test_query = (
        "What is cloud computing?"
    )

    print(
        f"\nQuery: {test_query}"
    )

    context = build_context(
        test_results,
        max_results=2,
        max_characters=6000,
        query=test_query
    )

    print(
        "\nOptimized Context:"
    )

    print("-" * 60)

    print(context)

    citations = build_source_citations(
        test_results
    )

    print(
        "\nSource Citations:"
    )

    print("-" * 60)

    for citation in citations:

        print(citation)

    print(
        "\n" + "=" * 60
    )

    print(
        "CONTEXT OPTIMIZER TEST COMPLETED"
    )

    print("=" * 60)