from collections import deque


class ConversationMemory:
    """
    Stores recent user questions and answers for multi-turn RAG conversations.
    """

    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)

    def add_turn(self, question, answer):
        self.history.append({
            "question": question,
            "answer": answer
        })

    def get_history(self):
        return list(self.history)

    def get_context(self):
        if not self.history:
            return ""

        context = []

        for i, turn in enumerate(self.history, start=1):
            context.append(
                f"Turn {i}:\n"
                f"User: {turn['question']}\n"
                f"Assistant: {turn['answer']}"
            )

        return "\n\n".join(context)

    def clear(self):
        self.history.clear()

    def is_empty(self):
        return len(self.history) == 0

    def __len__(self):
        return len(self.history)


if __name__ == "__main__":
    print("=" * 60)
    print("CONVERSATION MEMORY TEST")
    print("=" * 60)

    memory = ConversationMemory(max_turns=3)

    memory.add_turn(
        "What is cloud computing?",
        "Cloud computing provides computing resources over the internet."
    )

    memory.add_turn(
        "What are its benefits?",
        "Benefits include flexible access to computing resources."
    )

    print("\nStored Turns:", len(memory))

    print("\nConversation Context:")
    print("-" * 60)
    print(memory.get_context())

    print("\nMemory Test Completed!")