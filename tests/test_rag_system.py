def test_imports():
    from src.search.hybrid_search import hybrid_search
    from src.search.reranker import rerank
    from src.evidence_extractor import EvidenceExtractor
    from src.rag.rag_pipeline import rag_pipeline
    assert hybrid_search is not None
    assert rerank is not None
    assert EvidenceExtractor is not None
    assert rag_pipeline is not None
def test_hybrid_search():
    from src.search.hybrid_search import hybrid_search
    results = hybrid_search(
        "What are embeddings?",
        top_k=5
    )
    assert isinstance(results, list)
    assert len(results) > 0
def test_reranker():
    from src.search.hybrid_search import hybrid_search
    from src.search.reranker import rerank
    results = hybrid_search(
        "What are embeddings?",
        top_k=5
    )
    reranked = rerank(
        "What are embeddings?",
        results,
        top_k=3
    )
    assert isinstance(reranked, list)
    assert len(reranked) > 0
    assert "reranker_score" in reranked[0]
def test_evidence_extraction():
    from src.search.hybrid_search import hybrid_search
    from src.search.reranker import rerank
    from src.evidence_extractor import EvidenceExtractor
    results = hybrid_search(
        "What are embeddings?",
        top_k=10
    )
    reranked = rerank(
        "What are embeddings?",
        results,
        top_k=5
    )
    evidence = EvidenceExtractor().extract(
        "What are embeddings?",
        reranked
    )
    assert isinstance(evidence, list)
    assert len(evidence) > 0
def test_rag_pipeline():
    from src.rag.rag_pipeline import rag_pipeline
    result = rag_pipeline(
        "What are embeddings?",
        top_k=5
    )
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) > 0
