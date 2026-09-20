class ConversationMemory:

    def __init__(self, max_messages=10):
        self.max_messages = max_messages
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({
            "role": role,
            "content": content
        })

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self):
        return self.messages.copy()

    def get_context(self):
        if not self.messages:
            return ""

        context = []

        for message in self.messages:
            context.append(
                f"{message['role'].capitalize()}: "
                f"{message['content']}"
            )

        return "\n".join(context)

    def clear(self):
        self.messages = []