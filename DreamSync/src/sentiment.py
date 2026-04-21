from __future__ import annotations

import re

from rules import NEGATIVE_WORDS, POSITIVE_WORDS


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def analyze_sentiment(text: str) -> tuple[str, float]:
    tokens = tokenize(text)
    if not tokens:
        return "neutral", 0.0

    positive_count = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative_count = sum(1 for token in tokens if token in NEGATIVE_WORDS)
    raw_score = positive_count - negative_count
    normalized = raw_score / max(1, positive_count + negative_count)

    if normalized >= 0.25:
        label = "positive"
    elif normalized <= -0.25:
        label = "negative"
    else:
        label = "neutral"

    return label, round(normalized, 3)

