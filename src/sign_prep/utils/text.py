"""Text normalization utilities."""

import re

import ftfy


def normalize_text(text: str) -> str:
    """Keep semantic content; fix mojibake and whitespace only."""
    text = ftfy.fix_text(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
