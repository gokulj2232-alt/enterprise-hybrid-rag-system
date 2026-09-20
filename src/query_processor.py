import re


class QueryProcessor:

    # Generic intent words that should not become search keywords
    STOP_WORDS = {
        "what", "is", "are", "the", "a", "an", "of", "in", "on", "to",
        "for", "and", "or", "does", "do", "can", "could", "will", "how",
        "why", "when", "where", "who", "which", "was", "were", "be",
        "been", "being", "this", "that", "these", "those",

        # Question modifiers
        "many", "much", "long",

        # Process / intent words
        "work", "works",

        # Entity intent words
        "company", "companies",
        "organization", "organizations",
        "provider", "providers",
        "provide", "provides",

        # Number / availability intent
        "available",

        # Location intent
        "located",

        # Date intent
        "launched",

        # Reason / descriptive intent
        "useful"
    }

    ENTITY_WORDS = {
        "company",
        "companies",
        "organization",
        "organizations",
        "provider",
        "providers"
    }

    def normalize(self, query):
        query = str(query).strip()

        # Normalize spaces
        query = re.sub(r"\s+", " ", query)

        # Normalize multiple question marks
        query = re.sub(r"\?+$", "?", query)

        return query

    def detect_query_type(self, query):
        normalized = query.lower().strip()
        words = normalized.split()

        if not words:
            return "unknown"

        # WHO
        if words[0] == "who":
            return "person"

        # WHEN
        if words[0] == "when":
            return "date"

        # WHERE
        if words[0] == "where":
            return "location"

        # WHICH
        if words[0] == "which":
            if any(word in normalized for word in self.ENTITY_WORDS):
                return "entity"

            return "selection"

        # WHY
        if words[0] == "why":
            return "reason"

        # HOW MANY / HOW MUCH
        if len(words) >= 2:
            if words[0] == "how" and words[1] in ["many", "much"]:
                return "number"

            if words[0] == "how" and words[1] == "long":
                return "duration"

        # HOW
        if words[0] == "how":
            return "process"

        # WHAT
        if words[0] == "what":

            if any(
                phrase in normalized
                for phrase in [
                    "what is",
                    "what are",
                    "what does",
                    "what do",
                    "what means"
                ]
            ):
                return "definition"

            return "information"

        # YES / NO / FACT
        if words[0] in [
            "is",
            "are",
            "does",
            "do",
            "can",
            "could",
            "will"
        ]:
            return "fact"

        return "unknown"

    def extract_keywords(self, query, query_type=None):

        words = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query.lower()
        )

        keywords = []

        for word in words:

            # Remove generic intent words
            if word in self.STOP_WORDS:
                continue

            # Ignore very short words
            if len(word) <= 2:
                continue

            # Avoid duplicates
            if word not in keywords:
                keywords.append(word)

        return keywords

    def build_keyword_query(self, keywords):
        return " ".join(keywords)

    def add_expansion(self, expanded_queries, text):
        """
        Adds an expansion only if it is not already present.
        """

        text = self.normalize(text)

        if not text:
            return

        if text not in expanded_queries:
            expanded_queries.append(text)

    def expand_query(self, query, query_type):

        normalized = self.normalize(query)

        keywords = self.extract_keywords(
            normalized,
            query_type
        )

        keyword_query = self.build_keyword_query(keywords)

        expanded_queries = []

        # Always keep original query
        self.add_expansion(
            expanded_queries,
            normalized
        )

        # Add clean keyword query
        if keyword_query:
            self.add_expansion(
                expanded_queries,
                keyword_query
            )

        # --------------------------------------------------
        # DEFINITION
        # --------------------------------------------------

        if query_type == "definition":

            if keyword_query:

                self.add_expansion(
                    expanded_queries,
                    f"definition of {keyword_query}"
                )

                self.add_expansion(
                    expanded_queries,
                    f"meaning of {keyword_query}"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} concept"
                )

        # --------------------------------------------------
        # PROCESS
        # --------------------------------------------------

        elif query_type == "process":

            if keyword_query:

                self.add_expansion(
                    expanded_queries,
                    f"how {keyword_query} works"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} process"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} procedure"
                )

        # --------------------------------------------------
        # REASON
        # --------------------------------------------------

        elif query_type == "reason":

            if keyword_query:

                self.add_expansion(
                    expanded_queries,
                    f"reason for {keyword_query}"
                )

                self.add_expansion(
                    expanded_queries,
                    f"why {keyword_query}"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} causes"
                )

        # --------------------------------------------------
        # ENTITY
        # --------------------------------------------------

        elif query_type == "entity":

            if keyword_query:

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} companies"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} organizations"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} providers"
                )

        # --------------------------------------------------
        # DATE
        # --------------------------------------------------

        elif query_type == "date":

            if keyword_query:

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} date"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} year"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} timeline"
                )

        # --------------------------------------------------
        # LOCATION
        # --------------------------------------------------

        elif query_type == "location":

            if keyword_query:

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} location"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} place"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} address"
                )

        # --------------------------------------------------
        # NUMBER
        # --------------------------------------------------

        elif query_type == "number":

            if keyword_query:

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} number"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} count"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} quantity"
                )

        # --------------------------------------------------
        # DURATION
        # --------------------------------------------------

        elif query_type == "duration":

            if keyword_query:

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} duration"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} time"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} length"
                )

        # --------------------------------------------------
        # SELECTION
        # --------------------------------------------------

        elif query_type == "selection":

            if keyword_query:

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} options"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} choices"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} selection"
                )

        # --------------------------------------------------
        # FACT
        # --------------------------------------------------

        elif query_type == "fact":

            if keyword_query:

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} information"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} facts"
                )

        # --------------------------------------------------
        # INFORMATION
        # --------------------------------------------------

        elif query_type == "information":

            if keyword_query:

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} information"
                )

                self.add_expansion(
                    expanded_queries,
                    f"{keyword_query} details"
                )

        # Return maximum 5 queries
        return expanded_queries[:5]

    def choose_strategy(self, query_type):

        if query_type in [
            "entity",
            "date",
            "number",
            "location"
        ]:
            return "hybrid"

        if query_type in [
            "definition",
            "information",
            "process",
            "reason",
            "duration"
        ]:
            return "semantic"

        if query_type in [
            "fact",
            "selection"
        ]:
            return "hybrid"

        return "hybrid"

    def process(self, query):

        normalized_query = self.normalize(query)

        query_type = self.detect_query_type(
            normalized_query
        )

        keywords = self.extract_keywords(
            normalized_query,
            query_type
        )

        expanded_queries = self.expand_query(
            normalized_query,
            query_type
        )

        retrieval_strategy = self.choose_strategy(
            query_type
        )

        return {
            "original_query": query,
            "normalized_query": normalized_query,
            "query_type": query_type,
            "keywords": keywords,
            "expanded_queries": expanded_queries,
            "retrieval_strategy": retrieval_strategy
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ADVANCED QUERY PROCESSOR TEST")
    print("=" * 60)

    processor = QueryProcessor()

    test_queries = [

        "What is cloud computing???",

        "Which companies provide cloud services?",

        "How does cloud computing work?",

        "Where is the company located?",

        "When was the system launched?",

        "How many servers are available?",

        "Why is cloud computing useful?"
    ]

    for query in test_queries:

        result = processor.process(query)

        print("\n" + "-" * 60)

        print(
            f"Original Query : "
            f"{result['original_query']}"
        )

        print(
            f"Normalized     : "
            f"{result['normalized_query']}"
        )

        print(
            f"Query Type     : "
            f"{result['query_type']}"
        )

        print(
            f"Keywords       : "
            f"{result['keywords']}"
        )

        print(
            f"Strategy       : "
            f"{result['retrieval_strategy']}"
        )

        print("\nExpanded Queries:")

        for expanded in result["expanded_queries"]:

            print(f"  - {expanded}")

    print("\n" + "=" * 60)

    print(
        "QUERY PROCESSOR TEST COMPLETED"
    )

    print("=" * 60)