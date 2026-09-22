"""String utilities."""

import re


def clean_text(text: str) -> str:
    """Clean text by removing extra spaces and newlines.

    Args:
        text (str): Text to clean.

    Returns:
        str: Cleaned text.
    """
    return re.sub(r"\n+", " ", text.strip())
