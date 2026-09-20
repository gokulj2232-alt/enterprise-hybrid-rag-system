import re


def clean_text(text):
    if not text:
        return ""

    # Remove UTF-8 BOM
    text = text.replace("\ufeff", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text