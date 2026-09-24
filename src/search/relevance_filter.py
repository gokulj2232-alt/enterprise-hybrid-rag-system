# ==================================================
# RELEVANCE FILTER
# ==================================================
def _to_float(value, default=None):
    """Safely convert a value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
def filter_relevant_results(
    results,
    min_score=-5.0,
    min_results=1
):
    """
    Remove weakly relevant reranked results safely.
    """
    if not results:
        return []
    threshold = _to_float(
        min_score,
        default=-5.0
    )
    filtered_results = []
    for result in results:
        if not isinstance(result, dict):
            continue
        score = _to_float(
            result.get("reranker_score"),
            default=None
        )
        if score is None:
            continue
        result_copy = result.copy()
        result_copy["reranker_score"] = score
        if score >= threshold:
            filtered_results.append(result_copy)
    return filtered_results
# ==================================================
# TEST
# ==================================================
if __name__ == "__main__":
    test_results = [
        {
            "title": "RAG",
            "reranker_score": 7.62
        },
        {
            "title": "Context Retrieval",
            "reranker_score": 2.75
        },
        {
            "title": "Document Chunking",
            "reranker_score": 1.84
        },
        {
            "title": "K Means Clustering",
            "reranker_score": -10.87
        },
        {
            "title": "String Score Test",
            "reranker_score": "3.50"
        }
    ]
    print()
    print("=" * 60)
    print("RELEVANCE FILTER TEST")
    print("=" * 60)
    filtered = filter_relevant_results(
        test_results,
        min_score="-5.0"
    )
    print()
    print(f"Original results: {len(test_results)}")
    print(f"Filtered results: {len(filtered)}")
    for rank, result in enumerate(
        filtered,
        start=1
    ):
        print()
        print(f"Result #{rank}")
        print(f"Title: {result['title']}")
        print(f"Score: {result['reranker_score']}")
