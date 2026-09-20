from pathlib import Path
import pickle
import time

from src.data_loader import load_documents
from src.preprocessing import preprocess_documents
from src.chunking import chunk_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


# ============================================================
# DATASET CONFIGURATION
# ============================================================

# Use the newly upgraded 50K dataset
DATASET_PATH = "data/NLP_50K_Upgraded_Final.xlsx"

INDEX_PATH = "models/faiss.index"
METADATA_PATH = "models/chunk_metadata.pkl"


def main():

    start_time = time.time()

    print("=" * 60)
    print("AI SEMANTIC SEARCH - INDEX BUILDING")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------

    print("\n[1/5] Loading documents...")

    documents = load_documents(DATASET_PATH)

    print(
        f"Loaded {len(documents):,} documents."
    )

    # --------------------------------------------------
    # 2. Preprocess documents
    # --------------------------------------------------

    print("\n[2/5] Preprocessing documents...")

    documents = preprocess_documents(documents)

    print(
        f"Preprocessed {len(documents):,} documents."
    )

    # --------------------------------------------------
    # 3. Create chunks
    # --------------------------------------------------

    print("\n[3/5] Creating document chunks...")

    chunks = chunk_documents(documents)

    print(
        f"Created {len(chunks):,} chunks."
    )

    # --------------------------------------------------
    # 4. Generate embeddings
    # --------------------------------------------------

    print("\n[4/5] Generating embeddings...")

    embedding_model = EmbeddingModel()

    texts = [
        chunk["chunk_text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        batch_size=32
    )

    print(
        f"Embedding matrix shape: {embeddings.shape}"
    )

    # --------------------------------------------------
    # 5. Build FAISS index
    # --------------------------------------------------

    print("\n[5/5] Building FAISS vector index...")

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add_vectors(
        embeddings,
        chunks
    )

    vector_store.save(
        INDEX_PATH,
        METADATA_PATH
    )

    # --------------------------------------------------
    # Save processed chunks separately
    # --------------------------------------------------

    processed_chunks_path = Path(
        "models/processed_chunks.pkl"
    )

    with open(
        processed_chunks_path,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print(
        f"\nDocuments : {len(documents):,}"
    )

    print(
        f"Chunks    : {len(chunks):,}"
    )

    print(
        f"Vectors   : {vector_store.index.ntotal:,}"
    )

    print(
        f"Dimension : {embeddings.shape[1]}"
    )

    print(
        f"Time      : {elapsed_time / 60:.2f} minutes"
    )

    print("\nSaved files:")

    print(f"  {INDEX_PATH}")
    print(f"  {METADATA_PATH}")
    print(f"  {processed_chunks_path}")


if __name__ == "__main__":
    main()