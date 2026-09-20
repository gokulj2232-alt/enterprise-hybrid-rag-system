from src.search.hybrid_search import hybrid_search
from src.search.reranker import rerank
from src.evidence_extractor import EvidenceExtractor

query = "What is Retrieval-Augmented Generation?"

print("Retrieving...")
results = hybrid_search(query, top_k=30)

print("Reranking...")
reranked = rerank(query, results, top_k=10)

print("Extracting evidence...")
extractor = EvidenceExtractor(max_sentences=8)
evidence = extractor.extract(query, reranked)

print("\n" + "=" * 60)
print("EXTRACTED EVIDENCE")
print("=" * 60)

for i, item in enumerate(evidence, start=1):
    print(f"\n{i}. {item.get('document_id', '')} | {item.get('title', '')}")
    print(item.get('sentence', ''))

print("\n" + "=" * 60)