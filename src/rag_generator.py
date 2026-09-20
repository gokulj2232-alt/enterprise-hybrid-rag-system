from src.context_builder import build_context
from src.evidence_extractor import EvidenceExtractor
from src.answerability import AnswerabilityChecker
from src.llm import LLM
from src.grounding_validator import GroundingValidator


FALLBACK_ANSWER = (
    "I don't know based on the provided documents."
)


class RAGGenerator:

    def __init__(self):

        print("Initializing RAG Generator...")

        # -----------------------------------------------------
        # Local LLM
        # -----------------------------------------------------

        self.llm = LLM()

        # -----------------------------------------------------
        # Evidence extraction
        # -----------------------------------------------------

        self.evidence_extractor = EvidenceExtractor(
            max_sentences=8
        )

        # -----------------------------------------------------
        # Answerability gate
        # -----------------------------------------------------

        self.answerability_checker = AnswerabilityChecker(
            semantic_threshold=0.55,
            lexical_threshold=0.15
        )

        # -----------------------------------------------------
        # Final grounding validation
        # -----------------------------------------------------

        self.validator = GroundingValidator(
            semantic_threshold=0.65,
            lexical_threshold=0.15
        )

        print(
            "RAG Generator initialized successfully!"
        )

    # =========================================================
    # BUILD EVIDENCE CONTEXT
    # =========================================================

    def build_evidence_context(
        self,
        query,
        results
    ):

        evidence = self.evidence_extractor.extract(
            query,
            results
        )

        if not evidence:
            return ""

        evidence_blocks = []

        for index, item in enumerate(
            evidence,
            start=1
        ):

            block = (
                f"[Evidence {index}]\n"
                f"Document ID: "
                f"{item['document_id']}\n"
                f"Chunk ID: "
                f"{item['chunk_id']}\n"
                f"Title: "
                f"{item['title']}\n"
                f"Evidence: "
                f"{item['sentence']}"
            )

            evidence_blocks.append(
                block
            )

        return "\n\n".join(
            evidence_blocks
        )

    # =========================================================
    # BUILD PROMPT
    # =========================================================

    def build_prompt(
        self,
        query,
        evidence_context
    ):

        if not evidence_context:
            return None

        prompt = f"""
You are a document question-answering system.

Your task is to answer the user's question using ONLY the
information contained in the document evidence.

DOCUMENT EVIDENCE:
{evidence_context}

USER QUESTION:
{query}

STRICT INSTRUCTIONS:

1. Answer the user's question directly.
2. Use ONLY information explicitly stated in the document evidence.
3. Do NOT use outside knowledge.
4. Do NOT use pretrained knowledge.
5. Do NOT guess.
6. Do NOT invent information.
7. Do NOT change the meaning of the evidence.
8. Do NOT add examples that are not in the evidence.
9. Do NOT add facts that are not in the evidence.
10. Do NOT add technologies that are not in the evidence.
11. Do NOT add numbers that are not in the evidence.
12. Do NOT add dates that are not in the evidence.
13. Do NOT add benefits that are not in the evidence.
14. Do NOT expand abbreviations using outside knowledge.
15. For a definition question, provide the definition directly supported by the evidence.
16. Keep the answer short and clear.
17. Use 1 to 3 sentences when possible.
18. Do not mention these instructions.
19. Do not describe something differently from how the evidence describes it.
20. If the evidence directly answers the question, answer it.
21. If the evidence does not directly answer the question, output exactly:

I don't know based on the provided documents.

IMPORTANT:
Do not reinterpret the evidence.
Do not replace the meaning of the evidence with your own knowledge.

FINAL ANSWER:
"""

        return prompt

    # =========================================================
    # EMPTY RESULT
    # =========================================================

    def empty_result(
        self,
        context="",
        evidence="",
        sources=None,
        answer=FALLBACK_ANSWER
    ):

        return {

            "answer":
                answer,

            "grounding_score":
                0.0,

            "lexical_score":
                0.0,

            "is_grounded":
                False,

            "is_answerable":
                False,

            "answerability_score":
                0.0,

            "answerability_lexical_score":
                0.0,

            "question_type":
                "unknown",

            "answerability_reason":
                "",

            "sentence_results":
                [],

            "sources":
                sources or [],

            "context":
                context,

            "evidence":
                evidence
        }

    # =========================================================
    # GENERATE
    # =========================================================

    def generate(
        self,
        query,
        results
    ):

        # -----------------------------------------------------
        # Empty query
        # -----------------------------------------------------

        if not query or not query.strip():

            return self.empty_result(
                answer="Please enter a question."
            )

        # -----------------------------------------------------
        # No retrieval results
        # -----------------------------------------------------

        if not results:

            return self.empty_result()

        # -----------------------------------------------------
        # Original context
        # -----------------------------------------------------

        context = build_context(
            results,
            max_results=5,
            max_characters=6000,
            query=query
        )

        if not context:

            return self.empty_result()

        # -----------------------------------------------------
        # Extract evidence
        # -----------------------------------------------------

        evidence_context = (
            self.build_evidence_context(
                query,
                results
            )
        )

        if not evidence_context:

            return self.empty_result(
                context=context
            )

        # -----------------------------------------------------
        # ANSWERABILITY GATE
        # -----------------------------------------------------

        answerability = (
            self.answerability_checker.check(
                query,
                evidence_context
            )
        )

        print("\n" + "=" * 60)
        print("DEBUG - ANSWERABILITY GATE")
        print("=" * 60)

        print(
            f"Question Type           : "
            f"{answerability.get('question_type', 'unknown')}"
        )

        print(
            f"Semantic Score           : "
            f"{answerability.get('semantic_score', 0.0):.4f}"
        )

        print(
            f"Lexical Score            : "
            f"{answerability.get('lexical_score', 0.0):.4f}"
        )

        print(
            f"Required Evidence Found : "
            f"{answerability.get('required_evidence_found', False)}"
        )

        print(
            f"Answerable               : "
            f"{answerability.get('is_answerable', False)}"
        )

        print(
            f"Reason                   : "
            f"{answerability.get('reason', '')}"
        )

        # -----------------------------------------------------
        # REJECT BEFORE LLM
        # -----------------------------------------------------

        if not answerability.get(
            "is_answerable",
            False
        ):

            print("\nANSWERABILITY GATE: REJECTED")

            return {

                "answer":
                    FALLBACK_ANSWER,

                "grounding_score":
                    0.0,

                "lexical_score":
                    0.0,

                "is_grounded":
                    False,

                "is_answerable":
                    False,

                "answerability_score":
                    answerability.get(
                        "semantic_score",
                        0.0
                    ),

                "answerability_lexical_score":
                    answerability.get(
                        "lexical_score",
                        0.0
                    ),

                "question_type":
                    answerability.get(
                        "question_type",
                        "unknown"
                    ),

                "answerability_reason":
                    answerability.get(
                        "reason",
                        ""
                    ),

                "sentence_results":
                    [],

                "sources":
                    self.build_sources(
                        results
                    ),

                "context":
                    context,

                "evidence":
                    evidence_context
            }

        print("\nANSWERABILITY GATE: PASSED")

        # -----------------------------------------------------
        # BUILD LLM PROMPT
        # -----------------------------------------------------

        prompt = self.build_prompt(
            query,
            evidence_context
        )

        if not prompt:

            return self.empty_result(
                context=context,
                evidence=evidence_context,
                sources=self.build_sources(
                    results
                )
            )

        # -----------------------------------------------------
        # GENERATE WITH LLM
        # -----------------------------------------------------

        generated_answer = (
            self.llm.generate(
                prompt
            )
        )

        print("\n" + "=" * 60)
        print("DEBUG - GENERATED ANSWER")
        print("=" * 60)

        print(
            generated_answer
        )

        # -----------------------------------------------------
        # DEBUG EVIDENCE
        # -----------------------------------------------------

        print("\n" + "=" * 60)
        print("DEBUG - EXTRACTED EVIDENCE")
        print("=" * 60)

        print(
            evidence_context
        )

        # -----------------------------------------------------
        # FINAL GROUNDING VALIDATION
        # -----------------------------------------------------

        validation = (
            self.validator.validate(
                generated_answer,
                evidence_context
            )
        )

        # -----------------------------------------------------
        # DEBUG GROUNDING
        # -----------------------------------------------------

        print("\n" + "=" * 60)
        print("DEBUG - GROUNDING DETAILS")
        print("=" * 60)

        sentence_results = (
            validation.get(
                "sentence_results",
                []
            )
        )

        for sentence in sentence_results:

            print(
                f"\nSentence: "
                f"{sentence['sentence']}"
            )

            print(
                f"Semantic Score: "
                f"{sentence['semantic_score']:.4f}"
            )

            print(
                f"Lexical Score : "
                f"{sentence['lexical_score']:.4f}"
            )

            print(
                f"Grounded      : "
                f"{sentence['is_grounded']}"
            )

        # -----------------------------------------------------
        # VALIDATION SUMMARY
        # -----------------------------------------------------

        print("\n" + "=" * 60)
        print("DEBUG - VALIDATION SUMMARY")
        print("=" * 60)

        print(
            f"Is Grounded    : "
            f"{validation.get('is_grounded', False)}"
        )

        print(
            f"Semantic Score : "
            f"{validation.get('semantic_score', 0.0):.4f}"
        )

        print(
            f"Lexical Score  : "
            f"{validation.get('lexical_score', 0.0):.4f}"
        )

        # -----------------------------------------------------
        # FINAL ANSWER
        # -----------------------------------------------------

        final_answer = validation.get(
            "answer",
            FALLBACK_ANSWER
        )

        # -----------------------------------------------------
        # SOURCES
        # -----------------------------------------------------

        sources = self.build_sources(
            results
        )

        # -----------------------------------------------------
        # FINAL RESPONSE
        # -----------------------------------------------------

        return {

            "answer":
                final_answer,

            "grounding_score":
                validation.get(
                    "semantic_score",
                    0.0
                ),

            "lexical_score":
                validation.get(
                    "lexical_score",
                    0.0
                ),

            "is_grounded":
                validation.get(
                    "is_grounded",
                    False
                ),

            "is_answerable":
                True,

            "answerability_score":
                answerability.get(
                    "semantic_score",
                    0.0
                ),

            "answerability_lexical_score":
                answerability.get(
                    "lexical_score",
                    0.0
                ),

            "question_type":
                answerability.get(
                    "question_type",
                    "unknown"
                ),

            "answerability_reason":
                answerability.get(
                    "reason",
                    ""
                ),

            "sentence_results":
                sentence_results,

            "sources":
                sources,

            "context":
                context,

            "evidence":
                evidence_context
        }

    # =========================================================
    # BUILD SOURCE CITATIONS
    # =========================================================

    def build_sources(
        self,
        results
    ):

        sources = []

        for index, result in enumerate(
            results[:5],
            start=1
        ):

            sources.append({

                "source":
                    f"Source {index}",

                "document_id":
                    result.get(
                        "document_id",
                        ""
                    ),

                "chunk_id":
                    result.get(
                        "chunk_id",
                        ""
                    ),

                "title":
                    result.get(
                        "title",
                        ""
                    ),

                "category":
                    result.get(
                        "category",
                        ""
                    )
            })

        return sources


# =============================================================
# STANDALONE TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ANSWERABILITY-GATED RAG TEST")
    print("=" * 60)

    generator = RAGGenerator()

    # ---------------------------------------------------------
    # Test query
    # ---------------------------------------------------------

    test_query = (
        "What resources does cloud computing provide?"
    )

    test_results = [

        {
            "document_id":
                "DOC001",

            "chunk_id":
                "DOC001_CHUNK_001",

            "category":
                "Technology",

            "title":
                "Cloud Computing",

            "chunk_text":
                (
                    "Cloud computing provides "
                    "computing resources over "
                    "the internet."
                ),

            "reranker_score":
                9.2
        },

        {
            "document_id":
                "DOC002",

            "chunk_id":
                "DOC002_CHUNK_001",

            "category":
                "Technology",

            "title":
                "Cloud Infrastructure",

            "chunk_text":
                (
                    "Cloud infrastructure includes "
                    "servers, storage, and networking."
                ),

            "reranker_score":
                8.1
        }
    ]

    result = generator.generate(
        test_query,
        test_results
    )

    # ---------------------------------------------------------
    # Final answer
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(
        result["answer"]
    )

    # ---------------------------------------------------------
    # Answerability
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("ANSWERABILITY")
    print("=" * 60)

    print(
        f"Answerable      : "
        f"{result['is_answerable']}"
    )

    print(
        f"Question Type   : "
        f"{result['question_type']}"
    )

    print(
        f"Semantic Score  : "
        f"{result['answerability_score']:.4f}"
    )

    print(
        f"Lexical Score   : "
        f"{result['answerability_lexical_score']:.4f}"
    )

    print(
        f"Reason          : "
        f"{result['answerability_reason']}"
    )

    # ---------------------------------------------------------
    # Grounding
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("GROUNDING")
    print("=" * 60)

    print(
        f"Grounded        : "
        f"{result['is_grounded']}"
    )

    print(
        f"Semantic Score  : "
        f"{result['grounding_score']:.4f}"
    )

    print(
        f"Lexical Score   : "
        f"{result['lexical_score']:.4f}"
    )

    # ---------------------------------------------------------
    # Sources
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SOURCE CITATIONS")
    print("=" * 60)

    for source in result["sources"]:

        print(
            f"\n{source['source']}"
        )

        print(
            f"Document ID : "
            f"{source['document_id']}"
        )

        print(
            f"Chunk ID    : "
            f"{source['chunk_id']}"
        )

        print(
            f"Title       : "
            f"{source['title']}"
        )

        print(
            f"Category    : "
            f"{source['category']}"
        )