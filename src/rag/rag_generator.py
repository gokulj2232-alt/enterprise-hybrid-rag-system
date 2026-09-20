from src.evidence_extractor import EvidenceExtractor
from src.answerability import AnswerabilityChecker
from src.grounding_validator import GroundingValidator
from src.llm import LLM


FALLBACK_ANSWER = (
    "I don't know based on the provided documents."
)


class RAGGenerator:

    def __init__(self):
        print("Initializing RAG Generator...")

        self.llm = LLM()

        self.evidence_extractor = EvidenceExtractor(
            max_sentences=8
        )

        self.answerability_checker = AnswerabilityChecker(
            semantic_threshold=0.55,
            lexical_threshold=0.15
        )

        self.grounding_validator = GroundingValidator(
            semantic_threshold=0.65,
            lexical_threshold=0.15
        )

        print("RAG Generator initialized successfully!")

    # =========================================================
    # BUILD EVIDENCE CONTEXT
    # =========================================================

    def build_evidence_context(self, evidence):

        context_parts = []

        for i, item in enumerate(
            evidence,
            start=1
        ):

            context_parts.append(
                f"[Evidence {i}]\n"
                f"Document ID: {item.get('document_id', '')}\n"
                f"Chunk ID: {item.get('chunk_id', '')}\n"
                f"Title: {item.get('title', '')}\n"
                f"Evidence: {item.get('sentence', '')}"
            )

        return "\n\n".join(context_parts)

    # =========================================================
    # BUILD LLM PROMPT
    # =========================================================

    def build_prompt(
        self,
        query,
        evidence_context
    ):

        return f"""You are a document question-answering system.

Answer the user's question using ONLY the document evidence.

DOCUMENT EVIDENCE:
{evidence_context}

USER QUESTION:
{query}

STRICT INSTRUCTIONS:
1. Answer directly.
2. Use only information explicitly stated in the evidence.
3. Do not use outside or pretrained knowledge.
4. Do not guess or invent information.
5. Do not add examples, facts, benefits, numbers, dates, or technologies not in the evidence.
6. For a definition question, provide the definition directly supported by the evidence.
7. Keep the answer short and clear.
8. If the evidence does not answer the question, output exactly:
I don't know based on the provided documents.

FINAL ANSWER:
"""

    # =========================================================
    # EXTRACTIVE DEFINITION ANSWER
    # =========================================================

    def extractive_definition_answer(
        self,
        query,
        evidence
    ):

        if not query or not evidence:
            return None

        query_lower = query.lower().strip()

        definition_words = [
            "what is ",
            "what are ",
            "define ",
            "meaning of ",
            "explain "
        ]

        if not any(
            query_lower.startswith(word)
            for word in definition_words
        ):
            return None

        # -----------------------------------------------------
        # Extract topic from question
        # -----------------------------------------------------

        topic = query_lower

        for word in definition_words:

            if topic.startswith(word):

                topic = topic[
                    len(word):
                ].strip()

                break

        topic = topic.rstrip("?.!")

        topic_words = [
            word
            for word in topic.split()
            if len(word) > 2
        ]

        # -----------------------------------------------------
        # Search evidence sentences
        # -----------------------------------------------------

        for item in evidence:

            sentence = item.get(
                "sentence",
                ""
            ).strip()

            if not sentence:
                continue

            if len(sentence.split()) < 6:
                continue

            sentence_lower = sentence.lower()

            # -------------------------------------------------
            # Topic matching
            # -------------------------------------------------

            topic_match = False

            if (
                topic
                and topic in sentence_lower
            ):

                topic_match = True

            elif topic_words:

                matched_words = sum(
                    1
                    for word in topic_words
                    if word in sentence_lower
                )

                if matched_words >= max(
                    1,
                    len(topic_words) // 2
                ):

                    topic_match = True

            if not topic_match:
                continue

            # -------------------------------------------------
            # Definition patterns
            # -------------------------------------------------
            #
            # IMPORTANT:
            # Added " are " for plural definitions.
            # -------------------------------------------------

            definition_patterns = [

                " is ",
                " are ",

                " refers to ",
                " refer to ",

                " means ",
                " mean ",

                " retrieves ",
                " retrieve ",

                " uses ",
                " use ",

                " involves ",
                " involve ",

                " allows ",
                " allow ",

                " provides ",
                " provide ",

                " captures ",
                " capture ",

                " finds ",
                " find "
            ]

            if any(
                pattern in sentence_lower
                for pattern in definition_patterns
            ):

                return sentence.rstrip()

        return None

    # =========================================================
    # SOURCES
    # =========================================================

    def _sources(self, results):

        return [

            {
                "source": f"Source {i + 1}",
                "document_id": item.get(
                    "document_id",
                    ""
                ),
                "chunk_id": item.get(
                    "chunk_id",
                    ""
                ),
                "title": item.get(
                    "title",
                    ""
                ),
                "category": item.get(
                    "category",
                    ""
                )
            }

            for i, item in enumerate(
                results[:5]
            )
        ]

    # =========================================================
    # GENERATE ANSWER
    # =========================================================

    def generate(
        self,
        query,
        results
    ):

        # -----------------------------------------------------
        # Extract evidence
        # -----------------------------------------------------

        evidence = (
            self.evidence_extractor.extract(
                query,
                results
            )
        )

        evidence_context = (
            self.build_evidence_context(
                evidence
            )
        )

        # -----------------------------------------------------
        # Answerability
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
            f"{answerability.get('question_type', '')}"
        )

        print(
            f"Semantic Score          : "
            f"{answerability.get('semantic_score', 0):.4f}"
        )

        print(
            f"Lexical Score           : "
            f"{answerability.get('lexical_score', 0):.4f}"
        )

        print(
            f"Required Evidence Found : "
            f"{answerability.get('required_evidence_found', False)}"
        )

        print(
            f"Answerable              : "
            f"{answerability.get('is_answerable', False)}"
        )

        print(
            f"Reason                  : "
            f"{answerability.get('reason', '')}"
        )

        # -----------------------------------------------------
        # Answerability failed
        # -----------------------------------------------------

        if not answerability.get(
            "is_answerable",
            False
        ):

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
                        ""
                    ),

                "answerability_reason":
                    answerability.get(
                        "reason",
                        ""
                    ),

                "sentence_results":
                    [],

                "sources":
                    self._sources(results),

                "context":
                    evidence_context,

                "evidence":
                    evidence_context
            }

        print("\nANSWERABILITY GATE: PASSED")

        # =====================================================
        # EXTRACTIVE DEFINITION ANSWER
        # =====================================================

        extractive_answer = (
            self.extractive_definition_answer(
                query,
                evidence
            )
        )

        if extractive_answer:

            print(
                "\nEXTRACTIVE DEFINITION ANSWER USED"
            )

            grounding = (
                self.grounding_validator.validate(
                    extractive_answer,
                    evidence_context
                )
            )

            final_answer = extractive_answer

            if not grounding.get(
                "is_grounded",
                False
            ):

                final_answer = FALLBACK_ANSWER

            return {

                "answer":
                    final_answer,

                "grounding_score":
                    grounding.get(
                        "semantic_score",
                        0.0
                    ),

                "lexical_score":
                    grounding.get(
                        "lexical_score",
                        0.0
                    ),

                "is_grounded":
                    grounding.get(
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
                        ""
                    ),

                "answerability_reason":
                    answerability.get(
                        "reason",
                        ""
                    ),

                "sentence_results":
                    grounding.get(
                        "sentence_results",
                        []
                    ),

                "sources":
                    self._sources(results),

                "context":
                    evidence_context,

                "evidence":
                    evidence_context
            }

        # =====================================================
        # LLM GENERATION
        # =====================================================

        prompt = self.build_prompt(
            query,
            evidence_context
        )

        generated_answer = (
            self.llm.generate(
                prompt
            ).strip()
        )

        # -----------------------------------------------------
        # Grounding validation
        # -----------------------------------------------------

        grounding = (
            self.grounding_validator.validate(
                generated_answer,
                evidence_context
            )
        )

        final_answer = generated_answer

        if not grounding.get(
            "is_grounded",
            False
        ):

            final_answer = FALLBACK_ANSWER

        return {

            "answer":
                final_answer,

            "grounding_score":
                grounding.get(
                    "semantic_score",
                    0.0
                ),

            "lexical_score":
                grounding.get(
                    "lexical_score",
                    0.0
                ),

            "is_grounded":
                grounding.get(
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
                    ""
                ),

            "answerability_reason":
                answerability.get(
                    "reason",
                    ""
                ),

            "sentence_results":
                grounding.get(
                    "sentence_results",
                    []
                ),

            "sources":
                self._sources(results),

            "context":
                evidence_context,

            "evidence":
                evidence_context
        }


if __name__ == "__main__":

    print(
        "RAG Generator module loaded successfully."
    )