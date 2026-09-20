def normalize_filter_value(value):
    """
    Normalize a filter value for comparison.
    """

    if value is None:
        return None

    return str(value).strip().lower()


def filter_results(
    results,
    category=None,
    keyword=None,
    document_id=None
):
    """
    Filter search results using metadata.

    Supported filters:
    - category
    - keyword
    - document_id
    """

    filtered_results = []

    category = normalize_filter_value(category)
    keyword = normalize_filter_value(keyword)
    document_id = normalize_filter_value(document_id)

    for result in results:

        # Category filter
        if category:

            result_category = normalize_filter_value(
                result.get("category", "")
            )

            if result_category != category:
                continue

        # Keyword filter
        if keyword:

            result_keyword = normalize_filter_value(
                result.get("keyword", "")
            )

            if keyword not in result_keyword:
                continue

        # Document ID filter
        if document_id:

            result_document_id = normalize_filter_value(
                result.get("document_id", "")
            )

            if result_document_id != document_id:
                continue

        filtered_results.append(result)

    return filtered_results


def get_available_categories(results):
    """
    Return unique categories available
    in the provided results.
    """

    categories = set()

    for result in results:

        category = result.get("category", "")

        if category:
            categories.add(category)

    return sorted(categories)


if __name__ == "__main__":

    print("=" * 60)
    print("METADATA FILTER TEST")
    print("=" * 60)

    test_results = [
        {
            "document_id": "DOC001",
            "category": "Technology",
            "keyword": "cloud, computing, technology",
            "title": "Cloud Computing"
        },
        {
            "document_id": "DOC002",
            "category": "Healthcare",
            "keyword": "health, medicine",
            "title": "Healthcare Systems"
        },
        {
            "document_id": "DOC003",
            "category": "Technology",
            "keyword": "python, programming",
            "title": "Python Programming"
        }
    ]

    print("\nOriginal results:")
    print(len(test_results))

    filtered = filter_results(
        test_results,
        category="Technology"
    )

    print("\nCategory = Technology")
    print(f"Results: {len(filtered)}")

    for result in filtered:
        print(
            result["document_id"],
            "-",
            result["title"]
        )

    filtered = filter_results(
        test_results,
        keyword="cloud"
    )

    print("\nKeyword = cloud")
    print(f"Results: {len(filtered)}")

    for result in filtered:
        print(
            result["document_id"],
            "-",
            result["title"]
        )