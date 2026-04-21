from __future__ import annotations

import re


def split_story_into_scenes(story: str, max_scenes: int = 3) -> list[str]:
    normalized = re.sub(r"\s+", " ", story).strip()
    if not normalized:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", story) if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs[:max_scenes]

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if not sentences:
        return [normalized]

    scenes = []
    group_size = max(1, len(sentences) // max_scenes + (1 if len(sentences) % max_scenes else 0))
    for index in range(0, len(sentences), group_size):
        scenes.append(" ".join(sentences[index:index + group_size]))

    return scenes[:max_scenes]

