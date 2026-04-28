from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "DreamAudio" / "outputs"
DEFAULT_VOICE = "Karen"
DEFAULT_RATE = 145
DEFAULT_LEAD_IN_MS = 500
DEFAULT_TRAIL_MS = 700


def split_story_into_scenes(story: str, max_scenes: int = 3) -> list[str]:
    normalized = re.sub(r"\s+", " ", story).strip()
    if not normalized:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    return sentences[:max_scenes]


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "dream"


def build_say_command(
    text: str,
    output_path: Path,
    voice: str = DEFAULT_VOICE,
    rate: int = DEFAULT_RATE,
    lead_in_ms: int = DEFAULT_LEAD_IN_MS,
    trail_ms: int = DEFAULT_TRAIL_MS,
) -> list[str]:
    spoken_text = f"[[slnc {lead_in_ms}]] {text.strip()} [[slnc {trail_ms}]]"
    return [
        "say",
        "-v",
        voice,
        "-r",
        str(rate),
        "-o",
        str(output_path),
        spoken_text,
    ]


def export_scene_narrations(
    story: str,
    title: str = "DreamTales Story",
    voice: str = DEFAULT_VOICE,
    rate: int = DEFAULT_RATE,
    max_scenes: int = 3,
) -> dict[str, object]:
    scenes = split_story_into_scenes(story, max_scenes=max_scenes)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    story_slug = sanitize_slug(title)
    story_dir = OUTPUT_DIR / story_slug
    story_dir.mkdir(parents=True, exist_ok=True)

    manifest_scenes: list[dict[str, object]] = []
    for index, scene_text in enumerate(scenes, start=1):
        aiff_path = story_dir / f"scene_{index:02d}.aiff"
        wav_path = story_dir / f"scene_{index:02d}.wav"
        command = build_say_command(scene_text, aiff_path, voice=voice, rate=rate)
        subprocess.run(command, check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@22050", str(aiff_path), str(wav_path)], check=True)
        manifest_scenes.append(
            {
                "scene_number": index,
                "text": scene_text,
                "audio_path": str(wav_path),
                "source_aiff_path": str(aiff_path),
                "voice": voice,
                "rate": rate,
            }
        )

    manifest = {
        "title": title,
        "voice": voice,
        "rate": rate,
        "scene_count": len(manifest_scenes),
        "scenes": manifest_scenes,
    }

    manifest_path = story_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export calm scene-by-scene narration audio for a DreamTales story.")
    parser.add_argument("--story", required=True, help="Full story text to narrate.")
    parser.add_argument("--title", default="DreamTales Story", help="Story title used for the output folder name.")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="macOS say voice name. Default is Samantha.")
    parser.add_argument("--rate", type=int, default=DEFAULT_RATE, help="Speech rate in words per minute-ish for macOS say.")
    parser.add_argument("--max-scenes", type=int, default=3, help="Maximum number of scene sentences to export.")
    args = parser.parse_args()

    manifest = export_scene_narrations(
        story=args.story,
        title=args.title,
        voice=args.voice,
        rate=args.rate,
        max_scenes=args.max_scenes,
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
