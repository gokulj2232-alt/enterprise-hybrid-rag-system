import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
def atomic_replace(source, destination):
    os.replace(source, destination)
def build_uploaded_index(
    documents_file="data/uploads/uploaded_documents.jsonl",
    output_dir="data/uploads/index"
):
    os.makedirs(output_dir, exist_ok=True)
    faiss_path = os.path.join(output_dir, "faiss_index.bin")
    metadata_path = os.path.join(output_dir, "metadata.json")
    bm25_path = os.path.join(output_dir, "bm25.json")
    temp_faiss = faiss_path + ".tmp"
    temp_metadata = metadata_path + ".tmp"
    temp_bm25 = bm25_path + ".tmp"
    # Load uploaded documents
    documents = []
    with open(documents_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
    if not documents:
        raise ValueError("No documents found.")
    existing_documents = []
    existing_index = None
    existing_bm25 = []
    # Try loading existing index
    if (
        os.path.exists(faiss_path)
        and os.path.exists(metadata_path)
        and os.path.exists(bm25_path)
    ):
        try:
            existing_index = faiss.read_index(faiss_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_documents = json.load(f)
            with open(bm25_path, "r", encoding="utf-8") as f:
                existing_bm25 = json.load(f)
            valid = (
                existing_index.ntotal == len(existing_documents)
                and len(existing_documents) == len(existing_bm25)
                and len(existing_documents) <= len(documents)
            )
            if valid:
                for i, old_document in enumerate(existing_documents):
                    if old_document != documents[i]:
                        valid = False
                        break
            if not valid:
                print("Existing uploaded index is inconsistent.")
                print("Rebuilding uploaded index from scratch...")
                existing_index = None
                existing_documents = []
                existing_bm25 = []
        except Exception as e:
            print(f"Could not load existing uploaded index: {e}")
            print("Rebuilding uploaded index from scratch...")
            existing_index = None
            existing_documents = []
            existing_bm25 = []
    start_index = len(existing_documents)
    new_documents = documents[start_index:]
    print(f"Existing indexed documents: {start_index}")
    print(f"New documents to embed: {len(new_documents)}")
    # Create initial index
    if existing_index is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        texts = [doc["content"] for doc in documents]
        print(f"Creating embeddings for {len(texts)} documents...")
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        existing_documents = documents.copy()
        existing_bm25 = [
            doc["content"].lower().split()
            for doc in documents
        ]
    # Add only new documents
    elif new_documents:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        new_texts = [
            doc["content"]
            for doc in new_documents
        ]
        print(f"Creating embeddings for {len(new_texts)} new documents...")
        new_embeddings = model.encode(
            new_texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        new_embeddings = np.asarray(
            new_embeddings,
            dtype="float32"
        )
        existing_index.add(new_embeddings)
        existing_documents.extend(new_documents)
        existing_bm25.extend(
            [text.lower().split() for text in new_texts]
        )
        index = existing_index
    else:
        index = existing_index
    # Write everything to temporary files first.
    # This prevents a failed/interrupted write from corrupting the index.
    faiss.write_index(index, temp_faiss)
    with open(temp_metadata, "w", encoding="utf-8") as f:
        json.dump(
            existing_documents,
            f,
            ensure_ascii=False
        )
    with open(temp_bm25, "w", encoding="utf-8") as f:
        json.dump(
            existing_bm25,
            f,
            ensure_ascii=False
        )
    # Replace old files only after successful writes.
    atomic_replace(temp_faiss, faiss_path)
    atomic_replace(temp_metadata, metadata_path)
    atomic_replace(temp_bm25, bm25_path)
    print(f"Uploaded index ready: {index.ntotal} vectors")
    return {
        "documents": len(existing_documents),
        "new_documents": len(new_documents),
        "dimension": index.d,
        "faiss_index": faiss_path,
        "metadata": metadata_path,
        "bm25": bm25_path
    }
if __name__ == "__main__":
    result = build_uploaded_index()
    print("\nUploaded index build completed.")
    print(result)
