
# ==================================================
# RELEVANCE FILTER
# ==================================================

def filter_relevant_results(
    results,
    min_score=-5.0,
    min_results=1
):
    """
    Remove weakly relevant reranked results.

    Parameters
    ----------
    results : list
        Reranked search results.

    min_score : float
        Minimum Cross-Encoder score required.

    min_results : int
        Kept for compatibility with the pipeline.
        No weak result is forced into the final context.

    Returns
    -------
    list
        Filtered relevant results.
    """

    if not results:
        return []

    # Keep only results that meet the relevance threshold
    filtered_results = [
        result
        for result in results
        if result.get("reranker_score", -999) >= min_score
    ]

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
        }
    ]

    print()
    print("=" * 60)
    print("RELEVANCE FILTER TEST")
    print("=" * 60)

    filtered = filter_relevant_results(
        test_results,
        min_score=-5.0
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
        print(
            f"Score: "
            f"{result['reranker_score']}"
        )