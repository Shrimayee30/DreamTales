from __future__ import annotations

import re
from dataclasses import dataclass


MOOD_TO_MUSIC = {
    "warm": "warm acoustic lullaby, soft piano, gentle strings, cozy and hopeful",
    "calm": "calm ambient lullaby, soft pads, celesta, slow tempo, peaceful",
    "night": "moonlit ambient music, quiet harp, airy synths, soft reverb",
    "sunny": "bright whimsical music, light marimba, ukulele, playful rhythm",
    "rainy": "gentle rainy day piano, soft bells, muted strings, reflective",
    "tense": "subtle suspense underscore, low strings, restrained pulse",
    "magical": "magical fantasy score, shimmering bells, harp glissando, wonder",
}


@dataclass(frozen=True)
class MusicPrompt:
    scene: str
    mood: str
    prompt: str


def normalize_mood(mood: str | None) -> str:
    clean = (mood or "calm").strip().lower()
    return re.sub(r"[^a-z_ -]", "", clean) or "calm"


def build_music_prompt(scene: str, mood: str | None = None) -> MusicPrompt:
    clean_scene = re.sub(r"\s+", " ", scene).strip().rstrip(".!?")
    clean_mood = normalize_mood(mood)
    music_style = MOOD_TO_MUSIC.get(
        clean_mood,
        f"{clean_mood} cinematic background music, instrumental, soft dynamics",
    )

    prompt = (
        f"{music_style}. Background score for this scene: {clean_scene}. "
        "Instrumental only, no vocals, seamless loop feeling, suitable for a children's dream story."
    )
    return MusicPrompt(scene=clean_scene, mood=clean_mood, prompt=prompt)
