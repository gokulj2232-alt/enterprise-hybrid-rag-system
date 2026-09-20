import re


def split_into_sentences(text: str):
    """
    Split text into simple sentences.
    """

    if not isinstance(text, str):
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def create_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
):
    """
    Create overlapping text chunks.

    chunk_size:
        Maximum approximate number of words per chunk.

    overlap:
        Number of words shared between consecutive chunks.
    """

    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def chunk_documents(df):
    """
    Convert documents into overlapping chunks
    while preserving document metadata.
    """

    chunk_records = []

    for _, row in df.iterrows():

        text = row["search_text"]

        chunks = create_chunks(
            text=text,
            chunk_size=500,
            overlap=100
        )

        for chunk_number, chunk_text in enumerate(
            chunks,
            start=1
        ):

            chunk_id = (
                f"{row['document_id']}_"
                f"CHUNK_{chunk_number:03d}"
            )

            chunk_records.append({
                "chunk_id": chunk_id,
                "document_id": row["document_id"],
                "category": row["category"],
                "title": row["title"],
                "keyword": row["keyword"],
                "chunk_number": chunk_number,
                "chunk_text": chunk_text
            })

    return chunk_records


if __name__ == "__main__":

    from data_loader import load_documents
    from preprocessing import preprocess_documents

    dataset_path = "data/NLP_50K_Document_Dataset.xlsx"

    documents = load_documents(dataset_path)

    processed_documents = preprocess_documents(
        documents
    )

    chunks = chunk_documents(
        processed_documents
    )

    print("\nChunking completed!")

    print(
        f"Total documents: "
        f"{len(processed_documents)}"
    )

    print(
        f"Total chunks: "
        f"{len(chunks)}"
    )

    print("\nFirst chunk:")

    print(chunks[0])