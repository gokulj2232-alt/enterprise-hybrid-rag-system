import re
from collections import Counter


class EvidenceExtractor:
    """
    Extracts the most relevant evidence sentences from retrieved documents.

    Supports:
    - chunk_text
    - content
    - text

    Ranks individual sentences instead of simply taking the
    first sentences from the reranked documents.
    """

    def __init__(self, max_sentences=8):
        self.max_sentences = max_sentences

    # =========================================================
    # NORMALIZE TEXT
    # =========================================================

    def normalize_text(self, text):
        text = str(text).lower()
        text = text.replace("-", " ")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # =========================================================
    # SPLIT SENTENCES
    # =========================================================

    def split_sentences(self, text):
        if not text:
            return []

        text = str(text).strip()

        if not text:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    # =========================================================
    # GET DOCUMENT CONTENT
    # =========================================================

    def get_content(self, result):
        """
        Get clean document/chunk content.

        Priority:
        1. chunk_text
        2. content
        3. text

        If the text field contains metadata such as:
        Category: ... Title: ... Content: ...
        only the actual Content portion is returned.
        """

        # -----------------------------------------------------
        # 1. chunk_text
        # -----------------------------------------------------

        content = result.get("chunk_text")

        if content:
            return str(content).strip()

        # -----------------------------------------------------
        # 2. content
        # -----------------------------------------------------

        content = result.get("content")

        if content:
            return str(content).strip()

        # -----------------------------------------------------
        # 3. text
        # -----------------------------------------------------

        content = result.get("text", "")

        if not content:
            return ""

        content = str(content).strip()

        # -----------------------------------------------------
        # Remove metadata prefix if present
        # -----------------------------------------------------

        if "Content:" in content:
            content = content.split(
                "Content:",
                1
            )[1].strip()

        return content

    # =========================================================
    # QUERY KEYWORDS
    # =========================================================

    def get_query_keywords(self, query):
        normalized = self.normalize_text(query)

        stop_words = {
            "what",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "to",
            "for",
            "in",
            "on",
            "and",
            "or",
            "how",
            "why",
            "does",
            "do",
            "can",
            "explain",
            "define",
            "definition",
            "meaning"
        }

        return [
            word
            for word in normalized.split()
            if len(word) > 2
            and word not in stop_words
        ]

    # =========================================================
    # QUERY PHRASES
    # =========================================================

    def get_query_phrases(self, query):
        normalized = self.normalize_text(query)

        known_phrases = [
            "retrieval augmented generation",
            "large language model",
            "vector database",
            "semantic search",
            "natural language processing",
            "artificial intelligence",
            "machine learning",
            "question answering",
            "computer vision",
            "generative ai"
        ]

        return [
            phrase
            for phrase in known_phrases
            if phrase in normalized
        ]

    # =========================================================
    # DEFINITION QUESTION
    # =========================================================

    def is_definition_question(self, query):
        normalized = self.normalize_text(query)

        definition_patterns = [
            "what is",
            "what are",
            "define",
            "definition of",
            "meaning of",
            "explain"
        ]

        return any(
            normalized.startswith(pattern)
            for pattern in definition_patterns
        )

    # =========================================================
    # SCORE SENTENCE
    # =========================================================

    def score_sentence(
        self,
        query,
        sentence,
        title="",
        reranker_score=0.0
    ):

        normalized_sentence = self.normalize_text(
            sentence
        )

        normalized_title = self.normalize_text(
            title
        )

        query_keywords = self.get_query_keywords(
            query
        )

        query_phrases = self.get_query_phrases(
            query
        )

        score = 0.0

        # -----------------------------------------------------
        # Exact concept match
        # -----------------------------------------------------

        for phrase in query_phrases:

            if phrase in normalized_sentence:
                score += 5.0

            if phrase in normalized_title:
                score += 1.5

        # -----------------------------------------------------
        # Keyword matching
        # -----------------------------------------------------

        sentence_words = set(
            normalized_sentence.split()
        )

        matched_keywords = sum(
            1
            for word in query_keywords
            if word in sentence_words
        )

        score += matched_keywords * 0.8

        # -----------------------------------------------------
        # Partial keyword matching
        # -----------------------------------------------------

        for word in query_keywords:

            if word in normalized_sentence:
                score += 0.25

        # -----------------------------------------------------
        # Definition bonus
        # -----------------------------------------------------

        if self.is_definition_question(query):

            definition_patterns = [
                " is an approach ",
                " is a method ",
                " is a technique ",
                " is a system ",
                " is a process ",
                " is a framework ",
                " is a field ",
                " is a technology ",
                " refers to ",
                " means ",
                " combines ",
                " consists of ",
                " is used to ",
                " is the process of "
            ]

            if any(
                pattern in normalized_sentence
                for pattern in definition_patterns
            ):
                score += 3.0

        # -----------------------------------------------------
        # Explanation/context bonus
        # -----------------------------------------------------

        explanation_terms = [
            "when a user",
            "knowledge base",
            "retrieved documents",
            "language model",
            "typically",
            "includes",
            "useful when",
            "allows",
            "helps",
            "used when"
        ]

        explanation_matches = sum(
            1
            for term in explanation_terms
            if term in normalized_sentence
        )

        score += min(
            explanation_matches * 0.35,
            1.5
        )

        # -----------------------------------------------------
        # Sentence quality
        # -----------------------------------------------------

        word_count = len(
            sentence.split()
        )

        if 12 <= word_count <= 60:
            score += 0.8

        elif 8 <= word_count < 12:
            score += 0.3

        # -----------------------------------------------------
        # Penalize generic sentences
        # -----------------------------------------------------

        generic_patterns = [
            "important concept",
            "widely used in real world projects",
            "practical software systems",
            "may vary depending on",
            "application requirements",
            "expected system behavior"
        ]

        for pattern in generic_patterns:

            if pattern in normalized_sentence:
                score -= 1.5

        # -----------------------------------------------------
        # Small reranker contribution
        # -----------------------------------------------------

        try:

            score += min(
                max(
                    float(reranker_score),
                    0.0
                ) * 0.05,
                0.6
            )

        except (
            TypeError,
            ValueError
        ):
            pass

        return score

    # =========================================================
    # EXTRACT EVIDENCE
    # =========================================================

    def extract(self, query, results):

        if not query or not query.strip():
            return []

        if not results:
            return []

        candidates = []

        # -----------------------------------------------------
        # Process every retrieved result
        # -----------------------------------------------------

        for result_rank, result in enumerate(results):

            content = self.get_content(
                result
            )

            title = result.get(
                "title",
                ""
            )

            document_id = result.get(
                "document_id",
                ""
            )

            chunk_id = result.get(
                "chunk_id",
                ""
            )

            reranker_score = result.get(
                "reranker_score",
                0.0
            )

            if not content:
                continue

            sentences = self.split_sentences(
                content
            )

            # -------------------------------------------------
            # Score every sentence
            # -------------------------------------------------

            for sentence_rank, sentence in enumerate(
                sentences
            ):

                sentence = sentence.strip()

                if not sentence:
                    continue

                score = self.score_sentence(
                    query=query,
                    sentence=sentence,
                    title=title,
                    reranker_score=reranker_score
                )

                candidates.append({
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "sentence": sentence,
                    "_score": score,
                    "_result_rank": result_rank,
                    "_sentence_rank": sentence_rank
                })

        # -----------------------------------------------------
        # Sort by relevance
        # -----------------------------------------------------

        candidates.sort(
            key=lambda item: (
                item["_score"],
                -item["_result_rank"],
                -item["_sentence_rank"]
            ),
            reverse=True
        )

        # =====================================================
        # REMOVE DUPLICATES
        # =====================================================

        unique_evidence = []

        seen_sentences = set()

        seen_documents = Counter()

        for item in candidates:

            normalized = self.normalize_text(
                item["sentence"]
            )

            if normalized in seen_sentences:
                continue

            seen_sentences.add(
                normalized
            )

            document_id = item[
                "document_id"
            ]

            # Maximum 4 sentences from one document
            if seen_documents[
                document_id
            ] >= 4:
                continue

            seen_documents[
                document_id
            ] += 1

            unique_evidence.append(
                item
            )

            if len(unique_evidence) >= self.max_sentences:
                break

        # =====================================================
        # RETURN CLEAN EVIDENCE
        # =====================================================

        final_evidence = []

        for item in unique_evidence:

            final_evidence.append({
                "document_id": item[
                    "document_id"
                ],
                "chunk_id": item[
                    "chunk_id"
                ],
                "title": item[
                    "title"
                ],
                "sentence": item[
                    "sentence"
                ]
            })

        return final_evidence


# =============================================================
# STANDALONE TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EVIDENCE EXTRACTOR TEST")
    print("=" * 60)

    extractor = EvidenceExtractor(
        max_sentences=8
    )

    query = (
        "What is artificial intelligence?"
    )

    results = [

        {
            "document_id":
                "DOC001",

            "chunk_id":
                "DOC001_CHUNK_001",

            "title":
                "Artificial Intelligence",

            "text":
                (
                    "Artificial intelligence "
                    "is a field of technology. "
                    "It involves computer systems "
                    "performing tasks associated "
                    "with human intelligence."
                ),

            "reranker_score":
                8.5
        },

        {
            "document_id":
                "DOC002",

            "chunk_id":
                "DOC002_CHUNK_001",

            "title":
                "General Technology",

            "text":
                (
                    "Artificial intelligence "
                    "is an important concept "
                    "in modern software systems."
                ),

            "reranker_score":
                8.0
        }
    ]

    evidence = extractor.extract(
        query,
        results
    )

    for i, item in enumerate(
        evidence,
        start=1
    ):

        print(
            f"\nEvidence {i}"
        )

        print(
            f"Document ID: "
            f"{item['document_id']}"
        )

        print(
            f"Chunk ID: "
            f"{item['chunk_id']}"
        )

        print(
            f"Title: "
            f"{item['title']}"
        )

        print(
            f"Sentence: "
            f"{item['sentence']}"
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "EVIDENCE EXTRACTOR TEST COMPLETED"
    )

    print(
        "=" * 60
    )