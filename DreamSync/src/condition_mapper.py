from __future__ import annotations

import re

from rules import (
    ACTION_CLASSES,
    CHARACTER_CLASSES,
    KEYWORD_RULES,
    LOCATION_CLASSES,
    MOOD_CLASSES,
)


def count_keyword(text: str, keyword: str) -> int:
    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
    return len(re.findall(pattern, text))


def score_label(scene_text: str, category: str, labels: list[str]) -> str:
    lowered = scene_text.lower()
    label_scores = {label: 0 for label in labels if label != "none"}

    for label, keywords in KEYWORD_RULES[category].items():
        label_scores[label] = sum(count_keyword(lowered, keyword) for keyword in keywords)

    best_label, best_score = max(label_scores.items(), key=lambda item: item[1], default=("none", 0))
    return best_label if best_score > 0 else "none"


def infer_conditions(scene_text: str, sentiment: str) -> dict[str, str]:
    mood = score_label(scene_text, "mood", MOOD_CLASSES)
    if mood == "none" and sentiment == "positive":
        mood = "warm"
    elif mood == "none" and sentiment == "negative":
        mood = "rainy"

    return {
        "character": score_label(scene_text, "character", CHARACTER_CLASSES),
        "action": score_label(scene_text, "action", ACTION_CLASSES),
        "location": score_label(scene_text, "location", LOCATION_CLASSES),
        "mood": mood,
    }
