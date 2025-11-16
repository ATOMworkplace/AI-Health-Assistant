class SafetyFilter:
    KEYWORDS = [
        "suicide", "end my life", "kill myself",
        "worthless", "hopeless", "self harm", "cut myself"
    ]

    def is_unsafe(self, text):
        text = text.lower()

        for word in self.KEYWORDS:
            if word in text:
                return True
        return False
