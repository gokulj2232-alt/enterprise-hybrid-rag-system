import re
import html


def clean_text(text: str) -> str:
    """
    Clean and normalize document text.
    """

    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    # Keep letters, numbers and basic punctuation
    text = re.sub(r"[^a-z0-9\s.,!?;:()\-]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_search_text(
    title: str,
    content: str,
    keyword: str
) -> str:
    """
    Create a combined searchable representation
    of a document.
    """

    title = clean_text(title)
    content = clean_text(content)
    keyword = clean_text(keyword)

    return f"{title}. {content}. Keywords: {keyword}".strip()


def preprocess_documents(df):
    """
    Apply preprocessing to the complete document dataset.
    """

    df = df.copy()

    df["search_text"] = df.apply(
        lambda row: create_search_text(
            row["title"],
            row["content"],
            row["keyword"]
        ),
        axis=1
    )

    # Remove documents that became empty
    df = df[df["search_text"].str.strip() != ""]

    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":

    from data_loader import load_documents

    dataset_path = "data/NLP_50K_Document_Dataset.xlsx"

    documents = load_documents(dataset_path)

    processed_documents = preprocess_documents(documents)

    print("\nPreprocessing completed!")

    print(
        f"Documents after preprocessing: "
        f"{len(processed_documents)}"
    )

    print("\nOriginal content:")
    print(documents.iloc[0]["content"])

    print("\nProcessed search text:")
    print(processed_documents.iloc[0]["search_text"])