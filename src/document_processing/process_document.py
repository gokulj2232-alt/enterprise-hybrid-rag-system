from pathlib import Path

from src.document_processing.document_loader import load_document
from src.document_processing.preprocessing.clean_text import clean_text
from src.document_processing.chunker import chunk_text
from src.document_processing.embed_document import create_embeddings
from src.document_processing.update_index import update_faiss_index


def process_document(file_path):
    file_path = Path(file_path)

    print(f"Processing: {file_path.name}")

    # 1. Load document
    document = load_document(file_path)

    # 2. Clean text
    text = clean_text(document["text"])

    if not text:
        raise ValueError("Document contains no usable text.")

    # 3. Create chunks
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No chunks were created.")

    print(f"Created {len(chunks)} chunks.")

    # 4. Create embeddings
    embeddings = create_embeddings(chunks)

    # 5. Create metadata
    metadata = []

    for index, chunk in enumerate(chunks):
        metadata.append({
            "chunk_id": f"{document['document_id']}_chunk_{index}",
            "document_id": document["document_id"],
            "category": "Uploaded Document",
            "title": document["file_name"],
            "keyword": "",
            "chunk_index": index,
            "text": chunk
        })

    # 6. Update FAISS and metadata
    update_faiss_index(embeddings, metadata)

    print("Document successfully added to the knowledge base.")

    return {
        "document_id": document["document_id"],
        "file_name": document["file_name"],
        "chunks": len(chunks)
    }