from __future__ import annotations

import re

from rules import MOOD_TO_MUSIC_STYLE


def clean_scene(scene_text: str) -> str:
    return re.sub(r"\s+", " ", scene_text).strip().rstrip(".!?")


def build_image_prompt(scene_text: str, character: str, action: str, location: str, mood: str) -> str:
    scene = clean_scene(scene_text)
    labels = ", ".join(
        label.replace("_", " ")
        for label in [character, action, location, mood]
        if label != "none"
    )
    label_phrase = labels or "open dream scene"
    return f"{scene}. Dreamlike children's story illustration, {label_phrase}, soft colors, gentle atmosphere."


def build_music_prompt(scene_text: str, mood: str) -> str:
    scene = clean_scene(scene_text)
    style = MOOD_TO_MUSIC_STYLE.get(mood, MOOD_TO_MUSIC_STYLE["none"])
    return (
        f"{style}. Background score for this scene: {scene}. "
        "Instrumental only, no vocals, seamless loop feeling, suitable for a children's dream story."
    )

