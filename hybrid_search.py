import os
import re
import json
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.query_processor import QueryProcessor
from src.context_builder import build_context
from src.rag_generator import RAGGenerator
from src.retrieval_quality import (
    rank_by_evidence_quality,
    filter_low_information,
)
from src.memory import ConversationMemory
from src.rag_evaluator import RAGEvaluator

from bm25_search import BM25Search


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAISS_INDEX_PATH = os.path.join(
    BASE_DIR,
    "models",
    "faiss.index"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "models",
    "chunk_metadata.pkl"
)

BM25_INDEX_PATH = os.path.join(
    BASE_DIR,
    "models",
    "bm25.pkl"
)

CHUNKS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "processed_chunks.pkl"
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

SEMANTIC_TOP_K = 20
BM25_TOP_K = 20

RRF_K = 60

RERANK_CANDIDATES = 50
RERANK_TOP_K = 10

FALLBACK_ANSWER = (
    "I don't know based on the provided documents."
)


# ============================================================
# RESULT NORMALIZATION
# ============================================================

def normalize_result(item: Any) -> Dict:

    if not isinstance(item, dict):
        return {}

    if isinstance(item.get("chunk"), dict):

        result = dict(item["chunk"])

        for key, value in item.items():

            if key != "chunk" and key not in result:
                result[key] = value

    else:

        result = dict(item)

    if "content" not in result:

        result["content"] = (
            result.get("chunk_text")
            or result.get("text")
            or ""
        )

    if "chunk_text" not in result:

        result["chunk_text"] = result.get(
            "content",
            ""
        )

    if "document_id" not in result:

        result["document_id"] = (
            result.get("doc_id")
            or result.get("id")
            or ""
        )

    return result


# ============================================================
# RESULT HELPERS
# ============================================================

def get_content(item: Any) -> str:

    result = normalize_result(item)

    return str(
        result.get("content")
        or result.get("chunk_text")
        or result.get("text")
        or ""
    ).strip()


def get_document_id(item: Any) -> str:

    result = normalize_result(item)

    return str(
        result.get("document_id")
        or result.get("doc_id")
        or ""
    )


def get_chunk_id(item: Any) -> str:

    result = normalize_result(item)

    return str(
        result.get("chunk_id")
        or result.get("id")
        or ""
    )


def get_title(item: Any) -> str:

    result = normalize_result(item)

    return str(
        result.get("title")
        or ""
    )


def get_category(item: Any) -> str:

    result = normalize_result(item)

    return str(
        result.get("category")
        or ""
    )


# ============================================================
# STABLE RESULT ID
# ============================================================

def stable_id(item: Any) -> str:

    chunk_id = get_chunk_id(item)

    if chunk_id:
        return "chunk:" + chunk_id

    document_id = get_document_id(item)

    if document_id:
        return "document:" + document_id

    content = get_content(item).lower()

    return "content:" + " ".join(
        content.split()
    )


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicates(
    results: List[Dict]
) -> List[Dict]:

    unique = []
    seen = set()

    for item in results:

        result = normalize_result(item)

        if not result:
            continue

        key = stable_id(result)

        if key in seen:
            continue

        seen.add(key)
        unique.append(result)

    return unique


def remove_content_duplicates(
    results: List[Dict]
) -> List[Dict]:

    unique = []
    seen = set()

    for item in results:

        result = normalize_result(item)

        content = get_content(result)

        normalized = " ".join(
            content.lower().split()
        )

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        unique.append(result)

    return unique


# ============================================================
# METADATA FILTERING
# ============================================================

def apply_metadata_filters(
    results: List[Dict],
    filters: Optional[Dict]
) -> List[Dict]:

    if not filters:
        return results

    filtered = []

    for item in results:

        result = normalize_result(item)

        keep = True

        # Category
        if filters.get("category"):

            expected = str(
                filters["category"]
            ).lower()

            actual = str(
                result.get("category", "")
            ).lower()

            if actual != expected:
                keep = False

        # Document ID
        if filters.get("document_id"):

            expected = str(
                filters["document_id"]
            ).lower()

            actual = str(
                result.get("document_id", "")
            ).lower()

            if actual != expected:
                keep = False

        # Keyword
        if filters.get("keyword"):

            expected = str(
                filters["keyword"]
            ).lower()

            keyword_text = str(
                result.get("keyword")
                or result.get("keywords")
                or ""
            ).lower()

            content = get_content(
                result
            ).lower()

            if (
                expected not in keyword_text
                and expected not in content
            ):
                keep = False

        if keep:
            filtered.append(result)

    return filtered


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    k: int = RRF_K
) -> List[Dict]:

    scores = {}
    stored = {}

    for ranked_list in ranked_lists:

        for rank, item in enumerate(
            ranked_list,
            start=1
        ):

            result = normalize_result(item)

            if not result:
                continue

            key = stable_id(result)

            score = 1.0 / (
                k + rank
            )

            scores[key] = (
                scores.get(key, 0.0)
                + score
            )

            if key not in stored:
                stored[key] = result

    fused = []

    for key, score in scores.items():

        result = dict(
            stored[key]
        )

        result["rrf_score"] = score

        fused.append(result)

    fused.sort(
        key=lambda x: x.get(
            "rrf_score",
            0.0
        ),
        reverse=True
    )

    return fused


# ============================================================
# CROSS ENCODER RERANKER
# ============================================================

class LocalCrossEncoderReranker:

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME
    ):

        print(
            "Loading Cross-Encoder model..."
        )

        print(
            "Model:",
            model_name
        )

        self.model = CrossEncoder(
            model_name
        )

        print(
            "Cross-Encoder loaded successfully!"
        )

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:

        if not results:
            return []

        pairs = []
        valid_results = []

        for result in results:

            content = get_content(result)

            if not content:
                continue

            pairs.append(
                (
                    query,
                    content
                )
            )

            valid_results.append(result)

        if not pairs:
            return []

        print(
            "Cross-Encoder candidates:",
            len(pairs)
        )

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for result, score in zip(
            valid_results,
            scores
        ):

            item = dict(result)

            item["reranker_score"] = float(
                score
            )

            reranked.append(item)

        reranked.sort(
            key=lambda x: x.get(
                "reranker_score",
                0.0
            ),
            reverse=True
        )

        return reranked[:top_k]


# ============================================================
# HYBRID RAG SEARCH ENGINE
# ============================================================

class HybridRAGSearchEngine:

    def __init__(self):

        print()
        print("=" * 70)
        print("AI-POWERED ENTERPRISE HYBRID RAG SYSTEM")
        print("=" * 70)
        print()

        # ====================================================
        # 1. EMBEDDING MODEL
        # ====================================================

        print(
            "[1/8] Loading embedding model..."
        )

        self.embedding_model = EmbeddingModel(
            EMBEDDING_MODEL_NAME
        )

        print(
            "Embedding model ready."
        )

        print()

        # ====================================================
        # 2. FAISS
        # ====================================================

        print(
            "[2/8] Loading FAISS vector database..."
        )

        self.vector_store = VectorStore.load(
            FAISS_INDEX_PATH,
            METADATA_PATH
        )

        print(
            "FAISS vector database ready."
        )

        print()

        # ====================================================
        # 3. BM25
        # ====================================================

        print(
            "[3/8] Loading BM25 index..."
        )

        self.bm25 = BM25Search(
            BM25_INDEX_PATH,
            CHUNKS_PATH
        )

        print(
            "BM25 search ready."
        )

        print()

        # ====================================================
        # 4. CROSS ENCODER
        # ====================================================

        print(
            "[4/8] Loading Cross-Encoder reranker..."
        )

        self.reranker = (
            LocalCrossEncoderReranker()
        )

        print()

        # ====================================================
        # 5. QUERY PROCESSOR
        # ====================================================

        print(
            "[5/8] Loading query processor..."
        )

        self.query_processor = (
            QueryProcessor()
        )

        print(
            "Query processor ready."
        )

        print()

        # ====================================================
        # 6. RAG GENERATOR
        # ====================================================

        print(
            "[6/8] Loading RAG generator..."
        )

        self.rag_generator = (
            RAGGenerator()
        )

        print(
            "RAG generator ready."
        )

        print()

        # ====================================================
        # 7. MEMORY
        # ====================================================

        print(
            "[7/8] Loading conversation memory..."
        )

        self.memory = (
            ConversationMemory()
        )

        print(
            "Conversation memory ready."
        )

        print()

        # ====================================================
        # 8. EVALUATOR
        # ====================================================

        print(
            "[8/8] Loading RAG evaluator..."
        )

        self.rag_evaluator = (
            RAGEvaluator()
        )

        print(
            "RAG evaluator ready."
        )

        print()

        print("=" * 70)
        print("SYSTEM READY")
        print("=" * 70)
        print()


    # ========================================================
    # MEMORY
    # ========================================================

    def get_memory_history(self) -> List[Dict]:

        try:

            if hasattr(
                self.memory,
                "get_history"
            ):

                history = (
                    self.memory.get_history()
                )

                if history is None:
                    return []

                if isinstance(
                    history,
                    list
                ):
                    return history

                try:
                    return list(history)
                except Exception:
                    pass

        except Exception as error:

            print(
                "get_history() error:",
                error
            )

        try:

            if hasattr(
                self.memory,
                "history"
            ):

                history = self.memory.history

                if isinstance(
                    history,
                    list
                ):
                    return history

        except Exception:
            pass

        try:

            if hasattr(
                self.memory,
                "turns"
            ):

                history = self.memory.turns

                if isinstance(
                    history,
                    list
                ):
                    return history

        except Exception:
            pass

        return []


    def get_last_memory_turn(
        self
    ) -> Optional[Dict]:

        history = (
            self.get_memory_history()
        )

        if not history:
            return None

        last_turn = history[-1]

        if isinstance(
            last_turn,
            dict
        ):
            return last_turn

        return None


    # ========================================================
    # MEMORY QUESTION DETECTION
    # ========================================================

    def is_memory_question(
        self,
        query: str
    ) -> bool:

        normalized = (
            query.lower()
            .strip()
        )

        patterns = [

            "what topic did i just ask about",

            "what topic did i ask about",

            "what did i just ask",

            "what did i ask",

            "what was my previous question",

            "what was my last question",

            "what did i ask previously",

            "what topic was i asking about",

            "what were we talking about",

            "what are we talking about",

            "what was the previous topic",

            "what is the previous topic",
        ]

        return any(
            pattern in normalized
            for pattern in patterns
        )


    def answer_from_memory(
        self,
        query: str
    ) -> Optional[str]:

        if not self.is_memory_question(query):
            return None

        history = (
            self.get_memory_history()
        )

        if not history:

            return (
                "I don't have any previous "
                "conversation history."
            )

        last_user_question = None

        for turn in reversed(history):

            if not isinstance(
                turn,
                dict
            ):
                continue

            user_text = (
                turn.get("user")
                or turn.get("query")
                or turn.get("question")
                or ""
            )

            user_text = str(
                user_text
            ).strip()

            if user_text:

                last_user_question = (
                    user_text
                )

                break

        if not last_user_question:

            return (
                "I don't have a previous "
                "user question in memory."
            )

        topic_question = last_user_question

        topic_question = re.sub(
            r"^\s*what\s+is\s+",
            "",
            topic_question,
            flags=re.IGNORECASE
        )

        topic_question = re.sub(
            r"^\s*what\s+are\s+",
            "",
            topic_question,
            flags=re.IGNORECASE
        )

        topic_question = re.sub(
            r"^\s*who\s+is\s+",
            "",
            topic_question,
            flags=re.IGNORECASE
        )

        topic_question = re.sub(
            r"^\s*who\s+are\s+",
            "",
            topic_question,
            flags=re.IGNORECASE
        )

        topic_question = re.sub(
            r"[?!.]+$",
            "",
            topic_question
        )

        topic_question = (
            topic_question.strip()
        )

        if topic_question:

            return (
                f"You just asked about "
                f"{topic_question}."
            )

        return (
            f"Your previous question was: "
            f"{last_user_question}"
        )


    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    def semantic_search(
        self,
        query: str,
        top_k: int = SEMANTIC_TOP_K
    ) -> List[Dict]:

        embedding = (
            self.embedding_model.encode(
                [query]
            )
        )

        results = (
            self.vector_store.search(
                embedding,
                top_k=top_k
            )
        )

        normalized = []

        for item in results:

            result = normalize_result(item)

            if result:
                normalized.append(result)

        return normalized


    # ========================================================
    # BM25 SEARCH
    # ========================================================

    def bm25_search(
        self,
        query: str,
        top_k: int = BM25_TOP_K
    ) -> List[Dict]:

        try:

            results = self.bm25.search(
                query,
                top_k=top_k
            )

            normalized = []

            for item in results:

                result = normalize_result(item)

                if result:
                    normalized.append(result)

            return normalized

        except Exception as error:

            print(
                "BM25 error:",
                error
            )

            return []


    # ========================================================
    # HYBRID RETRIEVAL
    # ========================================================

    def hybrid_retrieve(
        self,
        query: str,
        metadata_filters: Optional[Dict] = None
    ) -> List[Dict]:

        print()
        print("=" * 70)
        print("HYBRID RETRIEVAL")
        print("=" * 70)

        processed = (
            self.query_processor.process(
                query
            )
        )

        expanded_queries = (
            processed.get(
                "expanded_queries",
                [query]
            )
        )

        if not expanded_queries:
            expanded_queries = [query]

        print(
            "Expanded queries:",
            len(expanded_queries)
        )

        semantic_lists = []
        bm25_lists = []

        for expanded_query in expanded_queries:

            print()
            print(
                "Query:",
                expanded_query
            )

            semantic_results = (
                self.semantic_search(
                    expanded_query,
                    SEMANTIC_TOP_K
                )
            )

            bm25_results = (
                self.bm25_search(
                    expanded_query,
                    BM25_TOP_K
                )
            )

            semantic_lists.append(
                semantic_results
            )

            bm25_lists.append(
                bm25_results
            )

        print()
        print(
            "Semantic candidates:",
            sum(
                len(items)
                for items in semantic_lists
            )
        )

        print(
            "BM25 candidates:",
            sum(
                len(items)
                for items in bm25_lists
            )
        )

        ranked_lists = []

        ranked_lists.extend(
            semantic_lists
        )

        ranked_lists.extend(
            bm25_lists
        )

        fused_results = (
            reciprocal_rank_fusion(
                ranked_lists
            )
        )

        print(
            "RRF candidates:",
            len(fused_results)
        )

        fused_results = (
            apply_metadata_filters(
                fused_results,
                metadata_filters
            )
        )

        print(
            "After metadata filtering:",
            len(fused_results)
        )

        fused_results = (
            remove_duplicates(
                fused_results
            )
        )

        print(
            "After exact duplicate removal:",
            len(fused_results)
        )

        fused_results = (
            remove_content_duplicates(
                fused_results
            )
        )

        print(
            "After content duplicate removal:",
            len(fused_results)
        )

        if not fused_results:
            return []

        print()
        print("=" * 70)
        print("CROSS-ENCODER RERANKING")
        print("=" * 70)

        candidates = fused_results[
            :RERANK_CANDIDATES
        ]

        try:

            reranked = (
                self.reranker.rerank(
                    query,
                    candidates,
                    top_k=RERANK_TOP_K
                )
            )

        except Exception as error:

            print(
                "Reranker error:",
                error
            )

            reranked = candidates[
                :RERANK_TOP_K
            ]

        print(
            "Reranked results:",
            len(reranked)
        )

        if not reranked:
            return []

        print()
        print("=" * 70)
        print("EVIDENCE QUALITY FILTERING")
        print("=" * 70)

        try:

            filtered = (
                filter_low_information(
                    reranked
                )
            )

        except Exception as error:

            print(
                "Quality filter error:",
                error
            )

            filtered = reranked

        print(
            "After low-information filtering:",
            len(filtered)
        )

        if not filtered:

            print(
                "All candidates were classified "
                "as low information."
            )

            filtered = reranked

        try:

            quality_ranked = (
                rank_by_evidence_quality(
                    query,
                    filtered
                )
            )

        except Exception as error:

            print(
                "Evidence quality ranking error:",
                error
            )

            quality_ranked = filtered

        final_results = []

        for item in quality_ranked[:RERANK_TOP_K]:

            result = normalize_result(item)

            if result:
                final_results.append(result)

        return final_results


    # ========================================================
    # RAG EVALUATION
    # ========================================================

    def evaluate_rag(
        self,
        query: str,
        results: List[Dict],
        context: str,
        answer: str,
        answerable: bool
    ) -> Dict[str, Any]:

        print()
        print("=" * 70)
        print("RAG EVALUATION")
        print("=" * 70)

        try:

            evaluation = (
                self.rag_evaluator.evaluate(
                    query=query,
                    retrieved_results=results,
                    context=context,
                    answer=answer,
                    answerable=answerable,
                    top_k=5
                )
            )

        except Exception as error:

            print(
                "RAG evaluation error:",
                error
            )

            evaluation = {
                "retrieval_precision": 0.0,
                "context_relevance": 0.0,
                "answer_relevance": 0.0,
                "faithfulness": 0.0,
                "answerability": 0.0,
                "overall_rag_score": 0.0,
            }

        return evaluation


    # ========================================================
    # ANSWERABILITY
    # ========================================================

    def extract_answerability(
        self,
        rag_result: Any
    ) -> bool:

        if not isinstance(
            rag_result,
            dict
        ):
            return False

        possible_keys = [

            "answerable",

            "is_answerable",

            "answerability",

            "answerability_result",

        ]

        for key in possible_keys:

            if key not in rag_result:
                continue

            value = rag_result.get(key)

            if isinstance(
                value,
                bool
            ):
                return value

            if isinstance(
                value,
                dict
            ):

                nested = (
                    value.get("answerable")
                    if "answerable" in value
                    else value.get("is_answerable")
                )

                if isinstance(
                    nested,
                    bool
                ):
                    return nested

        answer = (
            rag_result.get("answer")
            or rag_result.get("response")
            or rag_result.get("final_answer")
            or ""
        )

        answer = str(
            answer
        ).strip()

        if (
            not answer
            or answer == FALLBACK_ANSWER
        ):
            return False

        return True


    # ========================================================
    # MAKE JSON SAFE
    # ========================================================

    def make_json_safe(
        self,
        value: Any
    ) -> Any:

        try:

            json.dumps(value)

            return value

        except Exception:

            if isinstance(
                value,
                dict
            ):

                return {
                    str(key): self.make_json_safe(
                        item
                    )
                    for key, item in value.items()
                }

            if isinstance(
                value,
                list
            ):

                return [
                    self.make_json_safe(item)
                    for item in value
                ]

            if isinstance(
                value,
                tuple
            ):

                return [
                    self.make_json_safe(item)
                    for item in value
                ]

            return str(value)


    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        query: str,
        metadata_filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:

        query = str(query).strip()

        if not query:

            return {
                "answer": FALLBACK_ANSWER,
                "rag_details": {
                    "error": "Query cannot be empty."
                }
            }

        print()
        print("=" * 70)
        print("USER QUERY")
        print("=" * 70)
        print(query)

        # ====================================================
        # MEMORY
        # ====================================================

        memory_answer = (
            self.answer_from_memory(
                query
            )
        )

        if memory_answer is not None:

            try:

                self.memory.add_turn(
                    query,
                    memory_answer
                )

            except Exception as error:

                print(
                    "Memory storage error:",
                    error
                )

            return {
                "answer": memory_answer,

                "rag_details": {
                    "pipeline": "conversation_memory",
                    "query": query,
                    "query_processing": {},
                    "retrieval": {},
                    "retrieval_results": [],
                    "context": {
                        "characters": 0,
                        "text": ""
                    },
                    "generation": {},
                    "answerability": True,
                    "evaluation": {},
                    "memory_used": True
                }
            }

        # ====================================================
        # QUERY PROCESSING
        # ====================================================

        print()
        print("=" * 70)
        print("QUERY PROCESSING")
        print("=" * 70)

        processed = (
            self.query_processor.process(
                query
            )
        )

        expanded_queries = (
            processed.get(
                "expanded_queries",
                [query]
            )
        )

        if not expanded_queries:
            expanded_queries = [query]

        print(
            "Query Type:",
            processed.get("query_type")
        )

        print(
            "Keywords:",
            processed.get("keywords")
        )

        print(
            "Retrieval Strategy:",
            processed.get(
                "retrieval_strategy"
            )
        )

        # ====================================================
        # RETRIEVAL
        # ====================================================

        results = (
            self.hybrid_retrieve(
                query,
                metadata_filters
            )
        )

        display_results = results[:top_k]

        # ====================================================
        # NO RESULTS
        # ====================================================

        if not results:

            answer = FALLBACK_ANSWER

            evaluation = (
                self.evaluate_rag(
                    query=query,
                    results=[],
                    context="",
                    answer=answer,
                    answerable=False
                )
            )

            try:

                self.memory.add_turn(
                    query,
                    answer
                )

            except Exception as error:

                print(
                    "Memory storage error:",
                    error
                )

            return {
                "answer": answer,

                "rag_details": {
                    "pipeline": "hybrid_rag",
                    "query": query,

                    "query_processing": {
                        "query_type": processed.get(
                            "query_type"
                        ),
                        "keywords": processed.get(
                            "keywords"
                        ),
                        "retrieval_strategy": processed.get(
                            "retrieval_strategy"
                        ),
                        "expanded_queries": expanded_queries
                    },

                    "retrieval": {
                        "total_results": 0,
                        "displayed_results": 0,
                        "top_k": top_k
                    },

                    "retrieval_results": [],

                    "context": {
                        "characters": 0,
                        "text": ""
                    },

                    "generation": {},

                    "answerability": False,

                    "evaluation": evaluation,

                    "memory_used": False
                }
            }

        # ====================================================
        # RETRIEVAL DETAILS
        # ====================================================

        retrieval_details = []

        for index, result in enumerate(
            display_results,
            start=1
        ):

            detail = {

                "rank": index,

                "document_id": get_document_id(
                    result
                ),

                "chunk_id": get_chunk_id(
                    result
                ),

                "category": get_category(
                    result
                ),

                "title": get_title(
                    result
                ),

                "content": get_content(
                    result
                ),

                "rrf_score": float(
                    result.get(
                        "rrf_score",
                        0.0
                    )
                ),

                "reranker_score": float(
                    result.get(
                        "reranker_score",
                        0.0
                    )
                )
            }

            if "quality_score" in result:

                try:

                    detail["quality_score"] = float(
                        result.get(
                            "quality_score",
                            0.0
                        )
                    )

                except Exception:

                    detail["quality_score"] = str(
                        result.get(
                            "quality_score"
                        )
                    )

            retrieval_details.append(
                detail
            )

        # ====================================================
        # CONTEXT OPTIMIZATION
        # ====================================================

        print()
        print("=" * 70)
        print("CONTEXT OPTIMIZATION")
        print("=" * 70)

        context = ""

        try:

            context = build_context(
                results,
                max_results=top_k,
                max_characters=6000,
                query=query
            )

            print(
                "Optimized context created."
            )

            print(
                "Context size:",
                len(context),
                "characters"
            )

        except Exception as error:

            print(
                "Context optimization error:",
                error
            )

            context_parts = []

            for result in results[:top_k]:

                content = get_content(
                    result
                )

                if content:
                    context_parts.append(
                        content
                    )

            context = "\n".join(
                context_parts
            )[:6000]

        # ====================================================
        # RAG GENERATION
        # ====================================================

        print()
        print("=" * 70)
        print("RAG GENERATION")
        print("=" * 70)

        rag_result = None
        generation_error = None

        try:

            rag_result = (
                self.rag_generator.generate(
                    query,
                    results
                )
            )

            if isinstance(
                rag_result,
                dict
            ):

                answer = (
                    rag_result.get("answer")
                    or rag_result.get("response")
                    or rag_result.get("final_answer")
                    or FALLBACK_ANSWER
                )

            elif isinstance(
                rag_result,
                str
            ):

                answer = rag_result

            else:

                answer = str(
                    rag_result
                )

        except Exception as error:

            generation_error = str(error)

            print(
                "RAG generation error:",
                generation_error
            )

            answer = FALLBACK_ANSWER

        if not str(answer).strip():

            answer = FALLBACK_ANSWER

        answer = str(answer)

        # ====================================================
        # ANSWERABILITY
        # ====================================================

        answerable = (
            self.extract_answerability(
                rag_result
            )
        )

        if answer == FALLBACK_ANSWER:

            answerable = False

        # ====================================================
        # EVALUATION
        # ====================================================

        evaluation = (
            self.evaluate_rag(
                query=query,
                results=results,
                context=context,
                answer=answer,
                answerable=answerable
            )
        )

        # ====================================================
        # MEMORY
        # ====================================================

        try:

            self.memory.add_turn(
                query,
                answer
            )

        except Exception as error:

            print(
                "Memory storage error:",
                error
            )

        # ====================================================
        # GENERATION DETAILS
        # ====================================================

        generation_details = {}

        if isinstance(
            rag_result,
            dict
        ):

            generation_details = (
                self.make_json_safe(
                    rag_result
                )
            )

        elif rag_result is not None:

            generation_details = {
                "raw_result": str(
                    rag_result
                )
            }

        if generation_error:

            generation_details["error"] = (
                generation_error
            )

        # ====================================================
        # FINAL RAG DETAILS
        # ====================================================

        rag_details = {

            "pipeline": (
                "Query Processing → "
                "Semantic Search + BM25 → "
                "Hybrid Retrieval → "
                "RRF → "
                "Cross-Encoder Reranking → "
                "Evidence Quality → "
                "Context Optimization → "
                "RAG Generation → "
                "Answerability → "
                "Grounding → "
                "Evaluation → "
                "Conversation Memory"
            ),

            "query": query,

            "query_processing": {
                "query_type": processed.get(
                    "query_type"
                ),

                "keywords": processed.get(
                    "keywords"
                ),

                "retrieval_strategy": processed.get(
                    "retrieval_strategy"
                ),

                "expanded_queries": expanded_queries
            },

            "retrieval": {

                "total_results": len(
                    results
                ),

                "displayed_results": len(
                    retrieval_details
                ),

                "top_k": top_k
            },

            "retrieval_results": (
                retrieval_details
            ),

            "context": {

                "characters": len(
                    context
                ),

                "text": context
            },

            "generation": generation_details,

            "answerability": answerable,

            "evaluation": evaluation,

            "memory_used": False
        }

        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        final_result = {

            "answer": answer,

            "rag_details": rag_details
        }

        return self.make_json_safe(
            final_result
        )


    # ========================================================
    # DISPLAY MEMORY
    # ========================================================

    def display_memory(self):

        print()
        print("=" * 70)
        print("CONVERSATION MEMORY")
        print("=" * 70)

        history = (
            self.get_memory_history()
        )

        if not history:

            print(
                "No conversation history."
            )

            return

        print(
            "Stored conversation turns:",
            len(history)
        )

        for index, turn in enumerate(
            history,
            start=1
        ):

            print()
            print(
                f"Turn {index}"
            )

            if not isinstance(
                turn,
                dict
            ):

                print(
                    "Memory entry:",
                    turn
                )

                continue

            user_text = (
                turn.get("user")
                or turn.get("query")
                or turn.get("question")
                or ""
            )

            assistant_text = (
                turn.get("assistant")
                or turn.get("answer")
                or turn.get("response")
                or ""
            )

            print(
                "User:",
                user_text
            )

            print(
                "Assistant:",
                assistant_text
            )


    # ========================================================
    # INTERACTIVE MODE
    # ========================================================

    def interactive(self):

        print()
        print("=" * 70)
        print("INTERACTIVE RAG MODE")
        print("=" * 70)

        print(
            "Ask questions about the 50,000 documents."
        )

        print(
            "Conversation memory is enabled."
        )

        print(
            "RAG evaluation metrics are enabled."
        )

        print(
            "Type 'exit' to stop."
        )

        while True:

            try:

                query = input(
                    "\nAsk a question: "
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError
            ):

                print()
                print(
                    "Exiting..."
                )

                break

            if not query:
                continue

            if query.lower() in {
                "exit",
                "quit"
            }:

                print()
                print(
                    "Goodbye!"
                )

                break

            result = self.ask(query)

            print()
            print("=" * 70)
            print("FINAL RAG ANSWER")
            print("=" * 70)

            print()
            print(
                result.get(
                    "answer",
                    FALLBACK_ANSWER
                )
            )

            print()

            print("=" * 70)
            print("RAG DETAILS")
            print("=" * 70)

            try:

                print(
                    json.dumps(
                        result.get(
                            "rag_details",
                            {}
                        ),
                        indent=2,
                        ensure_ascii=False
                    )
                )

            except Exception:

                print(
                    result.get(
                        "rag_details",
                        {}
                    )
                )

            self.display_memory()


# ============================================================
# MAIN
# ============================================================

def main():

    engine = HybridRAGSearchEngine()

    engine.interactive()


if __name__ == "__main__":

    main()