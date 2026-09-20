import pickle

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


INDEX_PATH = "models/faiss.index"
METADATA_PATH = "models/chunk_metadata.pkl"


def main():

    print("=" * 60)
    print("SEMANTIC SEARCH TEST")
    print("=" * 60)

    print("\n[1/3] Loading embedding model...")
    embedding_model = EmbeddingModel()

    print("\n[2/3] Loading FAISS index...")
    vector_store = VectorStore.load(
        INDEX_PATH,
        METADATA_PATH
    )

    print(f"Loaded vectors: {len(vector_store):,}")

    print("\n[3/3] Enter your search query")

    query = input("\nQuery: ")

    query_vector = embedding_model.encode_query(query)

    # Retrieve more candidates first
    results = vector_store.search(
        query_vector,
        top_k=20
    )

   

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    for i, result in enumerate(results[:5], start=1):
        print(f"\nResult {i}")
        print("-" * 40)

        print(f"Document ID : {result['document_id']}")
        print(f"Category    : {result['category']}")
        print(f"Title       : {result['title']}")
        print(f"Score       : {result['score']:.4f}")

        print("\nContent:")
        print(result["chunk_text"][:500])


if __name__ == "__main__":
    main()