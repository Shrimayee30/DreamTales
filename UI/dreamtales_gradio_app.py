import random
import re
import sys
import time
import json
import math
import base64
import csv
import importlib.util
import struct
import wave
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Generator

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DREAMAUDIO_ROOT = PROJECT_ROOT / "DreamAudio"
DREAMVISION_ROOT = PROJECT_ROOT / "DreamVision"
DREAMVISION_2_ROOT = PROJECT_ROOT / "DreamVision 2.0"
DREAMSYNC_SRC = PROJECT_ROOT / "DreamSync" / "src"
SAMPLES_DIR = DREAMVISION_ROOT / "outputs" / "samples"
CHECKPOINT_PATH = DREAMVISION_ROOT / "outputs" / "checkpoints" / "conditional_generator_epoch_010.pt"
UI_OUTPUT_DIR = SAMPLES_DIR / "ui_generated"
LAKE_GAN_MODEL_PATH = DREAMVISION_2_ROOT / "models" / "lake_background_generator.torchscript.pt"
LAKE_GAN_HELPER_PATH = DREAMVISION_2_ROOT / "src" / "lake_background_gan.py"
NARRATION_HELPER_PATH = DREAMAUDIO_ROOT / "narration.py"
FEEDBACK_CSV_PATH = PROJECT_ROOT / "UI" / "dreamtales_feedback.csv"

if str(DREAMVISION_ROOT) not in sys.path:
    sys.path.append(str(DREAMVISION_ROOT))
if str(DREAMSYNC_SRC) not in sys.path:
    sys.path.append(str(DREAMSYNC_SRC))

from pipeline import analyze_story


CHARACTER_CLASSES = ["none", "mother_child", "friends", "animal_pair"]
ACTION_CLASSES = ["none", "walking", "holding_hands", "playing", "sleeping", "shopping"]
LOCATION_CLASSES = ["none", "store", "park", "forest", "bedroom", "street", "home"]
MOOD_CLASSES = ["none", "warm", "calm", "night", "sunny", "rainy"]

SCENE_PRESETS = {
    "Moonlit Memory": (
        "A glowing cloud drifts above a sleepy town and gathers the kindest thoughts from the night. "
        "It follows two best friends through a quiet park where lanterns sway in the breeze. "
        "At the end of the dream, silver stars settle over the trees and the whole sky feels calm."
    ),
    "Cozy Family Dream": (
        "A mother and child walk hand in hand toward a little neighborhood shop with golden lights in the windows. "
        "They laugh softly as the cloud paints warm colors across the evening. "
        "When they return home, the dream wraps them in a peaceful glow."
    ),
    "Forest Lullaby": (
        "The cloud glides into a moonlit forest where fireflies drift between quiet branches. "
        "Small animals curl up together while the leaves whisper like a lullaby. "
        "Everything slows down until the whole dream becomes still and gentle."
    ),
}

TINYSTORIES_SAMPLE_STORIES = [
    {
        "title": "The Brown Kayak",
        "keywords": ["kayak", "kayaking", "lake", "water", "boat", "splash", "sun"],
        "story": (
            "Once upon a time, in a big lake, there was a brown kayak that loved to roll and splash in the warm water. "
            "One sunny day, a little boy named Tim came to play, and they laughed as the kayak bobbed gently across the lake. "
            "When Tim went home, the brown kayak waited happily in the water, knowing they would play again soon."
        ),
    },
    {
        "title": "The Little Red Ball",
        "keywords": ["ball", "play", "park", "friend", "bounce"],
        "story": (
            "A little red ball rolled into the park and bounced beside a small girl named Mia. "
            "Mia chased the ball over the soft grass, laughing every time it jumped away. "
            "At sunset, she carried the ball home and promised they would play again tomorrow."
        ),
    },
    {
        "title": "The Sleepy Moon",
        "keywords": ["moon", "night", "sleep", "stars", "dream"],
        "story": (
            "The sleepy moon climbed into the sky and sprinkled silver light over a quiet town. "
            "A tiny rabbit looked up from the garden and wished the stars good night. "
            "Soon the rabbit curled under a leaf and dreamed of glowing clouds."
        ),
    },
]

TINYSTORIES_DATASET_PATHS = [
    PROJECT_ROOT / "DreamCore" / "data" / "tinystories.jsonl",
    PROJECT_ROOT / "DreamCore" / "data" / "tinystories.txt",
    PROJECT_ROOT / "data" / "tinystories.jsonl",
    PROJECT_ROOT / "data" / "tinystories.txt",
]

_TINYSTORIES_CACHE = None

SAMPLE_IMAGE_MAP = {
    ("friends", "holding_hands", "park", "calm"): SAMPLES_DIR / "generated_friends_holding_hands_park_calm.png",
    ("mother_child", "walking", "store", "warm"): SAMPLES_DIR / "generated_mother_child_walking_store_warm.png",
    ("none", "none", "forest", "night"): SAMPLES_DIR / "generated_none_none_forest_night.png",
}

KEYWORD_RULES = {
    "character": {
        "mother_child": ["mother", "child", "parent", "daughter", "son", "family"],
        "friends": ["friends", "friend", "together", "companions"],
        "animal_pair": ["animals", "animal", "foxes", "birds", "rabbits", "wolves", "deer"],
    },
    "action": {
        "walking": ["walk", "walking", "wander", "stroll", "glide"],
        "holding_hands": ["holding hands", "hand in hand", "together"],
        "playing": ["play", "playing", "dance", "laugh", "chase"],
        "sleeping": ["sleep", "sleeping", "rest", "dream", "lullaby"],
        "shopping": ["shop", "shopping", "store", "market"],
    },
    "location": {
        "store": ["store", "market", "shop"],
        "park": ["park", "garden", "meadow"],
        "forest": ["forest", "woods", "trees"],
        "bedroom": ["bedroom", "bed", "pillow"],
        "street": ["street", "road", "town"],
        "home": ["home", "house", "kitchen"],
    },
    "mood": {
        "warm": ["warm", "golden", "glow", "cozy"],
        "calm": ["calm", "peaceful", "gentle", "quiet", "still"],
        "night": ["night", "moonlit", "moon", "stars", "midnight"],
        "sunny": ["sunny", "bright", "daylight", "sun"],
        "rainy": ["rain", "rainy", "storm", "mist"],
    },
}

_MODEL_BUNDLE = None
_MODEL_LOAD_ERROR = None
_LAKE_GAN_BUNDLE = None
_LAKE_GAN_LOAD_ERROR = None
_NARRATION_MODULE = None
_NARRATION_LOAD_ERROR = None
_LOFI_LOOP_DATA_URI = None
CLOUD_GREETING = "What do you wanna dream about today?"


@dataclass
class ScenePlan:
    title: str
    text: str
    character: str
    action: str
    location: str
    mood: str


def split_story_into_scenes(story: str, max_scenes: int = 3) -> list[str]:
    normalized = re.sub(r"\s+", " ", story).strip()
    if not normalized:
        return [SCENE_PRESETS["Moonlit Memory"]]

    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if not chunks:
        return [normalized]

    scenes = []
    group_size = max(1, len(chunks) // max_scenes + (1 if len(chunks) % max_scenes else 0))
    for index in range(0, len(chunks), group_size):
        scenes.append(" ".join(chunks[index:index + group_size]))

    return scenes[:max_scenes]


def score_label(scene_text: str, category: str, labels: list[str]) -> str:
    lowered = scene_text.lower()
    label_scores = {label: 0 for label in labels if label != "none"}

    for label, keywords in KEYWORD_RULES[category].items():
        label_scores[label] = sum(lowered.count(keyword) for keyword in keywords)

    best_label, best_score = max(label_scores.items(), key=lambda item: item[1], default=("none", 0))
    return best_label if best_score > 0 else "none"


def infer_scene_plan(scene_text: str, scene_number: int) -> ScenePlan:
    clean_text = scene_text.strip()
    preview = clean_text.split(".")[0][:46].strip() or f"Dream pulse {scene_number}"
    title = f"Scene {scene_number}: {preview}"

    return ScenePlan(
        title=title,
        text=clean_text,
        character=score_label(clean_text, "character", CHARACTER_CLASSES),
        action=score_label(clean_text, "action", ACTION_CLASSES),
        location=score_label(clean_text, "location", LOCATION_CLASSES),
        mood=score_label(clean_text, "mood", MOOD_CLASSES),
    )


def format_tag(label: str) -> str:
    return label.replace("_", " ").title()


def build_scene_markup(plans: list[ScenePlan]) -> str:
    cards = []
    for plan in plans:
        cards.append(
            f"""
            <article class="scene-panel">
                <div class="scene-eyebrow">{plan.title}</div>
                <div class="scene-copy">{plan.text}</div>
                <div class="scene-tags">
                    <span>{format_tag(plan.character)}</span>
                    <span>{format_tag(plan.action)}</span>
                    <span>{format_tag(plan.location)}</span>
                    <span>{format_tag(plan.mood)}</span>
                </div>
            </article>
            """
        )

    return "<section class='scene-grid'>" + "".join(cards) + "</section>"


def split_story_by_sentence(story: str, max_scenes: int = 3) -> list[str]:
    normalized = re.sub(r"\s+", " ", story).strip()
    if not normalized:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    return (sentences or [normalized])[:max_scenes]


def score_story_for_prompt(prompt: str, story_record: dict[str, object]) -> int:
    prompt_words = set(re.findall(r"[a-z]+", prompt.lower()))
    keywords = set(story_record["keywords"])
    title_words = set(re.findall(r"[a-z]+", str(story_record["title"]).lower()))
    story_words = set(re.findall(r"[a-z]+", str(story_record["story"]).lower()))
    return 4 * len(prompt_words & keywords) + 2 * len(prompt_words & title_words) + len(prompt_words & story_words)


def make_story_record(story: str, index: int) -> dict[str, object]:
    story = re.sub(r"\s+", " ", story).strip()
    first_sentence = split_story_by_sentence(story, max_scenes=1)
    title = first_sentence[0][:42].rstrip(".!?") if first_sentence else f"TinyStories Story {index}"
    keywords = sorted(set(re.findall(r"[a-z]+", story.lower())))
    return {"title": title or f"TinyStories Story {index}", "keywords": keywords, "story": story}


def load_local_tinystories(limit: int = 5000) -> list[dict[str, object]]:
    global _TINYSTORIES_CACHE
    if _TINYSTORIES_CACHE is not None:
        return _TINYSTORIES_CACHE

    records = []
    for dataset_path in TINYSTORIES_DATASET_PATHS:
        if not dataset_path.exists():
            continue

        if dataset_path.suffix == ".jsonl":
            with dataset_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if len(records) >= limit:
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    story = row.get("story") or row.get("text") or row.get("content")
                    if story:
                        records.append(make_story_record(str(story), len(records) + 1))
        else:
            raw_text = dataset_path.read_text(encoding="utf-8")
            for story in re.split(r"\n\s*\n", raw_text):
                if len(records) >= limit:
                    break
                if story.strip():
                    records.append(make_story_record(story, len(records) + 1))

        if records:
            break

    _TINYSTORIES_CACHE = records or TINYSTORIES_SAMPLE_STORIES
    return _TINYSTORIES_CACHE


def select_tinystories_story(prompt: str) -> dict[str, object]:
    prompt = prompt.strip()
    story_records = load_local_tinystories()
    if not prompt:
        return story_records[0]
    return max(story_records, key=lambda story_record: score_story_for_prompt(prompt, story_record))


def build_story_scene_markup(story_record: dict[str, object], visible_scene_count: int | None = None) -> str:
    story = str(story_record["story"])
    scenes = split_story_by_sentence(story)
    visible_scene_count = len(scenes) if visible_scene_count is None else visible_scene_count
    scene_cards = []

    for index, scene_text in enumerate(scenes[:visible_scene_count], start=1):
        scene_cards.append(
            f"""
            <article class="story-scene-card">
                <div class="blank-image-window">
                    <span>Scene {index} image placeholder</span>
                </div>
                <div class="scene-eyebrow">Scene {index}</div>
                <div class="scene-copy">{escape(scene_text)}</div>
            </article>
            """
        )

    return f"""
    <section class="selected-story-card">
        <div class="metric-label">Selected TinyStories Story</div>
        <h2>{escape(str(story_record["title"]))}</h2>
        <p>{escape(story)}</p>
    </section>
    <section class="storyboard-grid">
        {"".join(scene_cards)}
    </section>
    """


def build_story_metrics(plans: list[ScenePlan], story: str) -> str:
    unique_moods = sorted({format_tag(plan.mood) for plan in plans if plan.mood != "none"}) or ["Open"]
    unique_locations = sorted({format_tag(plan.location) for plan in plans if plan.location != "none"}) or ["Anywhere"]
    word_count = len(story.split())

    return f"""
    <section class="metrics-bar">
        <div class="metric">
            <div class="metric-label">Story Pulse</div>
            <div class="metric-value">{word_count} words</div>
        </div>
        <div class="metric">
            <div class="metric-label">Mood Palette</div>
            <div class="metric-value">{", ".join(unique_moods)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Worlds Touched</div>
            <div class="metric-value">{", ".join(unique_locations)}</div>
        </div>
    </section>
    """


def format_status(message: str) -> str:
    return f"<div class='status-card'>{message}</div>"


def format_cloud_speech(message: str, muted: bool = False) -> str:
    mute_note = "Narration muted. " if muted else ""
    return f"""
    <div class="cloud-speech">
        <div class="cloud-speech-text">{escape(mute_note + message)}</div>
    </div>
    """


def image_to_data_uri(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def file_to_data_uri(file_path: Path) -> str:
    mime_type = "audio/wav" if file_path.suffix.lower() == ".wav" else "application/octet-stream"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def get_wav_duration_seconds(file_path: Path) -> float:
    with wave.open(str(file_path), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
    if frame_rate <= 0:
        return 0.0
    return frame_count / frame_rate


def build_audio_markup(audio_data_uri: str | None = None) -> str:
    if not audio_data_uri:
        return ""
    return f"""
    <div class="narration-audio-shell" aria-hidden="true">
        <audio class="dream-narration-audio" data-role="narration" autoplay playsinline preload="auto" src="{audio_data_uri}"></audio>
    </div>
    """


def synthesize_lofi_loop_data_uri(duration_seconds: float = 16.0, sample_rate: int = 22050) -> str:
    global _LOFI_LOOP_DATA_URI
    if _LOFI_LOOP_DATA_URI is not None:
        return _LOFI_LOOP_DATA_URI

    total_samples = int(duration_seconds * sample_rate)
    chord_progression = [
        (261.63, 329.63, 392.00),
        (220.00, 261.63, 329.63),
        (196.00, 246.94, 392.00),
        (174.61, 220.00, 293.66),
    ]
    melody_notes = [523.25, 493.88, 440.00, 392.00, 440.00, 392.00, 349.23, 329.63]
    frames = bytearray()

    for sample_index in range(total_samples):
        t = sample_index / sample_rate
        section = int(t // 4) % len(chord_progression)
        section_t = t % 4
        beat_t = t % 1
        half_beat_t = t % 0.5
        chord = chord_progression[section]

        pad = 0.0
        for freq in chord:
            pad += 0.10 * math.sin(2 * math.pi * freq * t)
            pad += 0.05 * math.sin(2 * math.pi * (freq / 2) * t)
        pad *= 0.42 + 0.18 * math.sin(2 * math.pi * 0.12 * t)

        melody_index = int(t * 2) % len(melody_notes)
        melody_env = math.exp(-4.8 * half_beat_t)
        melody = 0.12 * melody_env * math.sin(2 * math.pi * melody_notes[melody_index] * t)

        kick_env = math.exp(-10.0 * beat_t)
        kick = 0.20 * kick_env * math.sin(2 * math.pi * 62.0 * t)

        hat_phase = (t * 4) % 1
        hat_noise = math.sin(2 * math.pi * 3100.0 * t) * math.sin(2 * math.pi * 3700.0 * t)
        hat = 0.016 * math.exp(-22.0 * hat_phase) * hat_noise

        wow = 0.012 * math.sin(2 * math.pi * 0.23 * t)
        sample_value = pad + melody + kick + hat + wow
        sample_value = max(-0.95, min(0.95, sample_value))
        frames.extend(struct.pack("<h", int(sample_value * 32767)))

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    _LOFI_LOOP_DATA_URI = f"data:audio/wav;base64,{encoded}"
    return _LOFI_LOOP_DATA_URI


def build_background_music_markup() -> str:
    return f"""
    <div class="background-music-shell" aria-hidden="true">
        <audio id="dream-bg-music" loop preload="auto" src="{synthesize_lofi_loop_data_uri()}"></audio>
    </div>
    """


def build_movie_screen_markup(
    scene_text: str | None = None,
    scene_number: int | None = None,
    total_scenes: int | None = None,
    image_data_uri: str | None = None,
) -> str:
    if scene_text is None or scene_number is None or total_scenes is None:
        return ""

    image_markup = (
        f'<img class="scene-generated-image" src="{image_data_uri}" alt="Generated lake background for scene {scene_number}">'
        if image_data_uri
        else f'<div class="movie-placeholder">Scene {scene_number}/{total_scenes} image appears here</div>'
    )

    return f"""
    <section class="movie-stage">
        <div class="dream-projector-light" aria-hidden="true"></div>
        <div class="movie-screen">
            {image_markup}
            <div class="scene-number-badge">Scene {scene_number}/{total_scenes}</div>
        </div>
    </section>
    """


def build_feedback_markup() -> str:
    return """
    <section class="feedback-stage">
        <div class="feedback-card">
            <div class="feedback-kicker">Dream complete</div>
            <div class="feedback-title">How was your dream?</div>
            <div class="feedback-hint">Tap a star rating below to help the cloud learn.</div>
        </div>
    </section>
    """


def write_feedback_row(rating: str | int | None, feedback_context: dict | None) -> str:
    if rating in (None, ""):
        return ""

    context = feedback_context or {}
    FEEDBACK_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = FEEDBACK_CSV_PATH.exists()
    numeric_rating = str(rating).count("★") or rating

    with FEEDBACK_CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "rating", "prompt", "story_title", "story"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "rating": numeric_rating,
                "prompt": context.get("prompt", ""),
                "story_title": context.get("story_title", ""),
                "story": context.get("story", ""),
            }
        )

    return f"<div class='feedback-saved'>Thank you. Saved {numeric_rating}/5 stars.</div>"


def save_feedback_star(rating: int, feedback_context: dict | None) -> str:
    return write_feedback_row(rating, feedback_context)


def build_condition_vector(character: str, action: str, location: str, mood: str, torch, device):
    character_vec = torch.zeros(len(CHARACTER_CLASSES), device=device)
    action_vec = torch.zeros(len(ACTION_CLASSES), device=device)
    location_vec = torch.zeros(len(LOCATION_CLASSES), device=device)
    mood_vec = torch.zeros(len(MOOD_CLASSES), device=device)

    character_vec[CHARACTER_CLASSES.index(character)] = 1.0
    action_vec[ACTION_CLASSES.index(action)] = 1.0
    location_vec[LOCATION_CLASSES.index(location)] = 1.0
    mood_vec[MOOD_CLASSES.index(mood)] = 1.0

    return torch.cat([character_vec, action_vec, location_vec, mood_vec], dim=0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


def load_lake_helper_module():
    spec = importlib.util.spec_from_file_location("lake_background_gan", LAKE_GAN_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {LAKE_GAN_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_narration_module():
    global _NARRATION_MODULE, _NARRATION_LOAD_ERROR

    if _NARRATION_MODULE is not None or _NARRATION_LOAD_ERROR is not None:
        return _NARRATION_MODULE

    try:
        spec = importlib.util.spec_from_file_location("dreamaudio_narration", NARRATION_HELPER_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {NARRATION_HELPER_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _NARRATION_MODULE = module
    except Exception as exc:
        _NARRATION_LOAD_ERROR = str(exc)

    return _NARRATION_MODULE


def load_lake_gan_bundle():
    global _LAKE_GAN_BUNDLE, _LAKE_GAN_LOAD_ERROR

    if _LAKE_GAN_BUNDLE is not None or _LAKE_GAN_LOAD_ERROR is not None:
        return _LAKE_GAN_BUNDLE

    try:
        import torch

        if not LAKE_GAN_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing exported lake GAN at {LAKE_GAN_MODEL_PATH}")

        lake_helper = load_lake_helper_module()
        device = torch.device("cpu")
        generator = torch.jit.load(str(LAKE_GAN_MODEL_PATH), map_location=device)
        generator.eval()

        _LAKE_GAN_BUNDLE = {
            "device": device,
            "generator": generator,
            "helper": lake_helper,
            "torch": torch,
        }
    except Exception as exc:
        _LAKE_GAN_LOAD_ERROR = str(exc)

    return _LAKE_GAN_BUNDLE


def generate_lake_scene_images(scene_count: int) -> list[str]:
    bundle = load_lake_gan_bundle()
    if bundle is None:
        return []

    torch = bundle["torch"]
    helper = bundle["helper"]
    generator = bundle["generator"]
    device = bundle["device"]

    base_seed = random.randint(1, 1_000_000_000)
    image_data_uris = []
    for scene_index in range(scene_count):
        torch.manual_seed(base_seed + scene_index)
        noise = torch.randn(1, helper.LATENT_DIM, 1, 1, device=device)
        with torch.no_grad():
            output = generator(noise).squeeze(0)
        image = helper.tensor_to_pil(output)
        image = helper.storybook_cleanup(image)
        image_data_uris.append(image_to_data_uri(image))

    return image_data_uris


def generate_scene_narration_audio(story: str, title: str) -> list[dict[str, object]]:
    narration_module = load_narration_module()
    if narration_module is None:
        return []

    try:
        manifest = narration_module.export_scene_narrations(
            story=story,
            title=title,
            voice=narration_module.DEFAULT_VOICE,
            rate=narration_module.DEFAULT_RATE,
            max_scenes=3,
        )
    except Exception as exc:
        global _NARRATION_LOAD_ERROR
        _NARRATION_LOAD_ERROR = str(exc)
        return []

    audio_records = []
    for scene_record in manifest.get("scenes", []):
        audio_path = Path(str(scene_record["audio_path"]))
        if audio_path.exists():
            audio_records.append(
                {
                    "audio_data_uri": file_to_data_uri(audio_path),
                    "duration_seconds": get_wav_duration_seconds(audio_path),
                    "audio_path": str(audio_path),
                }
            )
    return audio_records


def load_model_bundle():
    global _MODEL_BUNDLE, _MODEL_LOAD_ERROR

    if _MODEL_BUNDLE is not None or _MODEL_LOAD_ERROR is not None:
        return _MODEL_BUNDLE

    try:
        import numpy as np
        import torch
        from PIL import Image

        from src.config import CONDITION_DIM, LATENT_DIM, NGF, NUM_CHANNELS
        from src.model import ConditionalGenerator
        from src.utils import get_device

        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Missing checkpoint at {CHECKPOINT_PATH}")

        device = get_device()
        generator = ConditionalGenerator(
            latent_dim=LATENT_DIM,
            condition_dim=CONDITION_DIM,
            ngf=NGF,
            num_channels=NUM_CHANNELS,
        ).to(device)

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        generator.load_state_dict(checkpoint["model_state_dict"])
        generator.eval()

        UI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _MODEL_BUNDLE = {
            "Image": Image,
            "device": device,
            "generator": generator,
            "numpy": np,
            "torch": torch,
            "latent_dim": LATENT_DIM,
        }
    except Exception as exc:
        _MODEL_LOAD_ERROR = str(exc)

    return _MODEL_BUNDLE


def save_generated_scene(plan: ScenePlan, scene_number: int, seed: int) -> Path | None:
    sample_path = SAMPLE_IMAGE_MAP.get((plan.character, plan.action, plan.location, plan.mood))
    if sample_path and sample_path.exists():
        return sample_path

    bundle = load_model_bundle()
    if bundle is None:
        return None

    torch = bundle["torch"]
    np = bundle["numpy"]
    image_class = bundle["Image"]
    device = bundle["device"]
    generator = bundle["generator"]
    latent_dim = bundle["latent_dim"]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    with torch.no_grad():
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        condition = build_condition_vector(plan.character, plan.action, plan.location, plan.mood, torch, device)
        output = generator(noise, condition).detach().cpu().squeeze(0)

    output = ((output + 1) / 2).clamp(0, 1)
    pixels = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    slug = f"scene_{scene_number}_{plan.character}_{plan.action}_{plan.location}_{plan.mood}".replace(" ", "_")
    save_path = UI_OUTPUT_DIR / f"{slug}.png"
    image_class.fromarray(pixels).save(save_path)
    return save_path


def create_gallery_item(plan: ScenePlan, image_path: Path | None) -> tuple[str | None, str]:
    caption = f"{plan.title} | {format_tag(plan.character)} | {format_tag(plan.action)} | {format_tag(plan.location)} | {format_tag(plan.mood)}"
    return (str(image_path) if image_path else None, caption)


def default_story() -> str:
    return SCENE_PRESETS["Moonlit Memory"]


def run_dreamvision(story: str, preset_name: str) -> Generator[tuple[str, str, list[tuple[str | None, str]]], None, None]:
    selected_story = story.strip() or SCENE_PRESETS.get(preset_name, default_story())
    sync_plan = analyze_story(selected_story)
    plans = sync_plan.scenes
    metrics = build_story_metrics(plans, selected_story)

    yield (
        format_status("DreamSync is breaking the story into scenes and mapping the emotional beats."),
        metrics + build_scene_markup(plans),
        [],
    )
    time.sleep(0.5)

    gallery_items = []
    session_seed = random.randint(1, 999_999)
    for index, plan in enumerate(plans, start=1):
        image_path = save_generated_scene(plan, index, session_seed + index)
        gallery_items.append(create_gallery_item(plan, image_path))
        status = (
            f"Rendering scene {index}/{len(plans)} with "
            f"{format_tag(plan.character)}, {format_tag(plan.action)}, "
            f"{format_tag(plan.location)}, and {format_tag(plan.mood)}."
        )
        yield (format_status(status), metrics + build_scene_markup(plans), gallery_items)
        time.sleep(0.4)

    final_note = "DreamVision 2.0 complete."
    if _MODEL_LOAD_ERROR:
        final_note += f" Using sample and fallback visuals because live model loading was unavailable: {_MODEL_LOAD_ERROR}"

    yield (format_status(final_note), metrics + build_scene_markup(plans), gallery_items)


def use_preset_story(preset_name: str) -> str:
    return SCENE_PRESETS.get(preset_name, default_story())


def run_cloud_intro():
    return (
        format_cloud_speech(CLOUD_GREETING),
        gr.update(visible=True),
        "",
        gr.update(visible=False),
        "",
        {},
        "",
    )


def run_tinystories_scene_planner(prompt: str):
    selected_story = select_tinystories_story(prompt)
    scenes = split_story_by_sentence(str(selected_story["story"]))
    feedback_context = {
        "prompt": prompt,
        "story_title": str(selected_story["title"]),
        "story": str(selected_story["story"]),
    }
    yield (
        format_cloud_speech("Making your dream..."),
        gr.update(visible=False),
        build_movie_screen_markup(),
        gr.update(visible=False),
        "",
        feedback_context,
        "",
    )
    generation_started = time.time()
    scene_image_uris = generate_lake_scene_images(len(scenes))
    scene_audio_records = generate_scene_narration_audio(str(selected_story["story"]), str(selected_story["title"]))
    elapsed = time.time() - generation_started
    time.sleep(max(0, 10 - elapsed))

    yield (
        format_cloud_speech("Here we go!"),
        gr.update(visible=False),
        build_movie_screen_markup(),
        gr.update(visible=False),
        "",
        feedback_context,
        "",
    )
    time.sleep(1)

    for index, sentence in enumerate(scenes, start=1):
        scene_image_uri = scene_image_uris[index - 1] if index - 1 < len(scene_image_uris) else None
        scene_audio_record = scene_audio_records[index - 1] if index - 1 < len(scene_audio_records) else {}
        scene_audio_uri = str(scene_audio_record.get("audio_data_uri", "")) or None
        scene_duration_seconds = float(scene_audio_record.get("duration_seconds", 4.0) or 4.0)
        yield (
            format_cloud_speech(sentence),
            gr.update(visible=False),
            build_movie_screen_markup(sentence, index, len(scenes), image_data_uri=scene_image_uri),
            gr.update(visible=False),
            "",
            feedback_context,
            build_audio_markup(scene_audio_uri),
        )
        time.sleep(max(scene_duration_seconds + 0.35, 4.0))

    yield (
        format_cloud_speech("And that was the dream."),
        gr.update(visible=False),
        build_feedback_markup(),
        gr.update(visible=True),
        "",
        feedback_context,
        "",
    )


CUSTOM_CSS = """
.gradio-container {
    background:
        radial-gradient(circle at 18% 18%, rgba(117, 164, 255, 0.22), transparent 28%),
        radial-gradient(circle at 82% 18%, rgba(255, 213, 139, 0.16), transparent 22%),
        radial-gradient(circle at 50% 115%, rgba(255, 255, 255, 0.12), transparent 34%),
        linear-gradient(180deg, #160055 0%, #1b0667 48%, #08001f 100%);
    color: #f4f6ff;
    font-family: "Avenir Next", "Trebuchet MS", sans-serif;
}

.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        radial-gradient(circle, rgba(255, 255, 255, 0.5) 0 1px, transparent 1.8px),
        radial-gradient(circle, rgba(255, 236, 186, 0.45) 0 1px, transparent 1.7px);
    background-size: 86px 86px, 132px 132px;
    background-position: 0 0, 32px 44px;
    opacity: 0.2;
}

.shell {
    max-width: 1120px;
    margin: 0 auto;
    padding: 12px 18px 4px;
}

.brand-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 9px;
    width: fit-content;
    margin: 0 auto;
    padding: 6px 14px;
    border-radius: 999px;
    background: transparent;
    border: 0;
    box-shadow: none;
}

.brand-cloud {
    position: relative;
    width: 30px;
    height: 17px;
    border-radius: 999px;
    background: #fffaf2;
    box-shadow:
        -8px 4px 0 -3px #fffaf2,
        9px 3px 0 -3px #fffaf2,
        0 -6px 0 -4px #fffaf2,
        0 0 13px rgba(255, 255, 255, 0.5);
}

.brand-title {
    color: #fffaf2;
    font-family: "Avenir Next", "Trebuchet MS", sans-serif;
    font-size: 1.08rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-shadow: 0 4px 18px rgba(0, 0, 0, 0.24);
}

.background-music-shell {
    width: 0;
    height: 0;
    overflow: hidden;
    opacity: 0;
    pointer-events: none;
}

.dream-settings-toggle {
    position: fixed;
    top: 22px;
    right: 24px;
    z-index: 1200;
    width: 46px;
    height: 46px;
    border: 0;
    border-radius: 999px;
    background: rgba(255, 250, 242, 0.14);
    color: #fffaf2;
    font-size: 1.18rem;
    line-height: 1;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
    backdrop-filter: blur(14px);
    cursor: pointer;
}

.dream-settings-panel {
    position: fixed;
    top: 76px;
    right: 24px;
    z-index: 1199;
    width: 132px;
    padding: 16px 14px 18px;
    border-radius: 26px;
    background: rgba(14, 6, 64, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
    backdrop-filter: blur(18px);
    opacity: 0;
    pointer-events: none;
    transform: translateY(-6px) scale(0.98);
    transition: opacity 180ms ease, transform 180ms ease;
}

.dream-settings-panel.is-open {
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0) scale(1);
}

.dream-settings-title {
    color: rgba(255, 250, 242, 0.92);
    font-size: 0.84rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 14px;
}

.dream-slider-stack {
    display: flex;
    justify-content: center;
    gap: 18px;
}

.dream-slider-column {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
}

.dream-slider-label {
    color: rgba(255, 250, 242, 0.72);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dream-volume-slider {
    appearance: slider-vertical;
    -webkit-appearance: slider-vertical;
    writing-mode: bt-lr;
    width: 28px;
    height: 120px;
    accent-color: #8fb6ff;
    background: transparent;
}

.panel {
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 20px;
    backdrop-filter: blur(14px);
}

.dream-stage {
    position: relative;
    max-width: 940px;
    min-height: calc(100vh - 74px);
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    gap: 8px;
    padding: 0 18px 8px;
}

.cloud-speech {
    width: min(780px, 94vw);
    min-height: 76px;
    padding: 0 10px;
    background: transparent;
    border: 0;
    box-shadow: none;
    text-align: center;
}

.cloud-speech-text {
    color: #fffaf0;
    font-size: clamp(1.15rem, 2.2vw, 1.8rem);
    line-height: 1.3;
    text-shadow: 0 4px 24px rgba(0, 0, 0, 0.38);
}

.cloud-row {
    align-items: center !important;
    justify-content: center;
    gap: 10px;
}

.cloud-row > .form {
    flex: 0 0 auto !important;
}

#cloud-touch {
    position: relative;
    width: 210px !important;
    min-width: 210px !important;
    max-width: 210px !important;
    height: 112px !important;
    margin: 0 auto;
    padding: 0 !important;
    flex: 0 0 auto !important;
    border: 0 !important;
    border-radius: 52% 48% 46% 54% / 58% 58% 42% 42% !important;
    background: #fff !important;
    color: #21152c !important;
    font-size: 1.85rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.2em;
    box-shadow:
        -46px 16px 0 -14px #fff,
        44px 14px 0 -12px #fff,
        0 -28px 0 -13px #fff,
        0 0 24px 10px rgba(255, 255, 255, 0.66),
        0 24px 48px rgba(0, 0, 0, 0.2) !important;
    transform: translateZ(0);
    transition: transform 180ms ease, filter 180ms ease;
}

#cloud-touch::before,
#cloud-touch::after {
    content: "";
    position: absolute;
    width: 30px;
    height: 20px;
    border-radius: 999px;
    background: rgba(236, 83, 141, 0.55);
    filter: blur(8px);
    top: 58px;
    z-index: 1;
}

#cloud-touch::before {
    left: 48px;
}

#cloud-touch::after {
    right: 48px;
}

#cloud-touch:hover {
    filter: brightness(1.02);
    transform: translateY(-4px) scale(1.01);
}

#cloud-touch:active {
    transform: translateY(1px) scale(0.99);
}

.mute-panel {
    min-width: 54px !important;
    max-width: 64px !important;
    padding: 0;
    border-radius: 999px;
    background: transparent;
    border: 0;
    margin-left: -4px;
}

.mute-panel .wrap {
    display: flex;
    justify-content: center;
}

.mute-panel label {
    font-size: 0 !important;
}

.mute-panel label span {
    font-size: 1.45rem !important;
}

.mute-panel input {
    display: none !important;
}

.mute-panel label:has(input:checked) span::after {
    content: " off";
    font-size: 0;
}
.cloud-input-panel {
    width: min(700px, 90vw);
    padding: 6px 8px 6px 18px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid rgba(255, 255, 255, 0.8);
    box-shadow: 0 16px 52px rgba(0, 0, 0, 0.18);
    align-items: center !important;
    gap: 8px;
}

.cloud-input-panel label {
    color: #362852 !important;
}

.cloud-input-panel > .form {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.cloud-input-panel .form,
.cloud-input-panel .block,
.cloud-input-panel .wrap,
.cloud-input-panel .input-container,
.cloud-input-panel .form > div {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.cloud-input-panel textarea {
    min-height: 46px !important;
    height: 46px !important;
    color: #160055 !important;
    border-radius: 999px !important;
    padding: 10px 12px !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    font-size: 1.05rem !important;
    line-height: 1.5 !important;
    resize: none !important;
    overflow: hidden !important;
    outline: none !important;
}

.storyboard-shell {
    max-width: 920px;
    margin: 0 auto;
    padding: 0 18px 14px;
}

.movie-stage {
    position: relative;
    display: flex;
    justify-content: center;
    margin-top: 0;
    isolation: isolate;
}

.dream-projector-light {
    position: absolute;
    top: -116px;
    left: 50%;
    width: min(760px, 90vw);
    height: 190px;
    transform: translateX(-50%);
    pointer-events: none;
    z-index: 0;
    opacity: 0.58;
    filter: blur(12px);
    background:
        linear-gradient(104deg, transparent 5%, rgba(255, 132, 187, 0.24) 24%, transparent 43%),
        linear-gradient(92deg, transparent 15%, rgba(255, 226, 142, 0.2) 39%, transparent 58%),
        linear-gradient(78deg, transparent 32%, rgba(128, 220, 255, 0.24) 56%, transparent 78%),
        radial-gradient(ellipse at 50% 0%, rgba(255, 255, 255, 0.28), transparent 64%);
    clip-path: polygon(42% 0%, 58% 0%, 100% 100%, 0% 100%);
    mix-blend-mode: screen;
}

.movie-screen {
    position: relative;
    z-index: 1;
    width: min(680px, 88vw);
    aspect-ratio: 16 / 8.4;
    display: grid;
    place-items: center;
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.025)),
        repeating-linear-gradient(45deg, rgba(255, 255, 255, 0.035) 0 10px, transparent 10px 20px);
    border: 1px dashed rgba(255, 255, 255, 0.26);
    box-shadow: 0 22px 90px rgba(0, 0, 0, 0.28);
    overflow: hidden;
}

.movie-placeholder {
    color: rgba(255, 250, 242, 0.56);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.78rem;
}

.scene-generated-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    image-rendering: auto;
    filter: saturate(1.04) contrast(1.02);
}

.scene-number-badge {
    position: absolute;
    left: 16px;
    bottom: 14px;
    padding: 7px 11px;
    border-radius: 999px;
    background: rgba(6, 0, 30, 0.48);
    color: rgba(255, 250, 242, 0.82);
    font-size: 0.76rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    backdrop-filter: blur(10px);
}

.feedback-stage {
    display: flex;
    justify-content: center;
    margin-top: 16px;
}

.feedback-card {
    width: min(560px, 88vw);
    padding: 34px 28px;
    text-align: center;
    border-radius: 34px;
    background:
        radial-gradient(circle at 50% 0%, rgba(255, 255, 255, 0.16), transparent 48%),
        rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.14);
    box-shadow: 0 24px 90px rgba(0, 0, 0, 0.22);
    backdrop-filter: blur(16px);
}

.feedback-kicker {
    color: rgba(255, 215, 141, 0.86);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    font-size: 0.78rem;
    margin-bottom: 10px;
}

.feedback-title {
    color: #fffaf2;
    font-size: clamp(1.5rem, 3vw, 2.2rem);
    margin-bottom: 8px;
}

.feedback-hint,
.feedback-saved {
    color: rgba(255, 250, 242, 0.72);
    font-size: 0.98rem;
}

.feedback-saved {
    text-align: center;
    margin-top: 10px;
}

.narration-audio-shell {
    width: 0;
    height: 0;
    overflow: hidden;
    opacity: 0;
    pointer-events: none;
}

.narration-audio-shell audio {
    width: 0;
    height: 0;
}

.star-feedback-row {
    width: min(560px, 88vw);
    margin: 2px auto 0;
    justify-content: center;
    gap: 10px;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.star-feedback-row > div {
    flex-grow: 0 !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.star-feedback-button,
.star-feedback-button button {
    min-width: 0 !important;
}

.star-feedback-button {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.star-feedback-button button {
    width: 52px !important;
    min-width: 52px !important;
    max-width: 52px !important;
    height: 52px !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 999px !important;
    background: transparent !important;
    box-shadow: none !important;
    color: rgba(255, 250, 242, 0.66) !important;
    font-size: 2.1rem !important;
    line-height: 1 !important;
    transition: transform 140ms ease, color 140ms ease, text-shadow 140ms ease !important;
}

.star-feedback-button button:hover,
.star-feedback-button button:focus-visible {
    color: #ffd98e !important;
    transform: translateY(-3px) scale(1.08);
    text-shadow: 0 0 16px rgba(255, 217, 142, 0.34);
}

.star-feedback-button button.star-lit {
    color: #ffd98e !important;
    text-shadow: 0 0 16px rgba(255, 217, 142, 0.34);
    transform: translateY(-3px) scale(1.08);
}

.star-feedback-button:nth-child(1) button { animation: star-float 420ms ease 0ms both; }
.star-feedback-button:nth-child(2) button { animation: star-float 420ms ease 40ms both; }
.star-feedback-button:nth-child(3) button { animation: star-float 420ms ease 80ms both; }
.star-feedback-button:nth-child(4) button { animation: star-float 420ms ease 120ms both; }
.star-feedback-button:nth-child(5) button { animation: star-float 420ms ease 160ms both; }

@keyframes star-float {
    from {
        opacity: 0;
        transform: translateY(10px) scale(0.92);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.status-card {
    padding: 14px 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(121, 160, 255, 0.2), rgba(255, 208, 122, 0.16));
    border: 1px solid rgba(255, 255, 255, 0.08);
    font-size: 1rem;
}

.metrics-bar {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 16px;
}

.metric {
    padding: 16px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: #9fb8ff;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 1.1rem;
    color: #fff6df;
}

.scene-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
}

.scene-panel {
    padding: 18px;
    border-radius: 20px;
    background: linear-gradient(180deg, rgba(14, 23, 48, 0.88), rgba(8, 12, 24, 0.92));
    border: 1px solid rgba(255, 255, 255, 0.07);
}

.scene-eyebrow {
    font-size: 0.95rem;
    font-weight: 700;
    color: #fff6df;
    margin-bottom: 10px;
}

.scene-copy {
    color: #d8def8;
    line-height: 1.65;
    min-height: 120px;
}

.scene-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.scene-tags span {
    padding: 8px 10px;
    border-radius: 999px;
    background: rgba(138, 180, 255, 0.12);
    border: 1px solid rgba(138, 180, 255, 0.2);
    color: #dce7ff;
    font-size: 0.85rem;
}

.selected-story-card {
    padding: 20px;
    margin-bottom: 16px;
    border-radius: 22px;
    background: linear-gradient(145deg, rgba(255, 246, 223, 0.1), rgba(138, 180, 255, 0.08));
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.selected-story-card h2 {
    margin: 6px 0 10px;
    color: #fff6df;
    font-size: 1.55rem;
}

.selected-story-card p {
    color: #d8def8;
    line-height: 1.7;
    margin: 0;
}

.storyboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 16px;
}

.story-scene-card {
    padding: 16px;
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(14, 23, 48, 0.9), rgba(8, 12, 24, 0.94));
    border: 1px solid rgba(255, 255, 255, 0.07);
}

.blank-image-window {
    min-height: 180px;
    display: grid;
    place-items: center;
    margin-bottom: 14px;
    border-radius: 18px;
    background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.02)),
        repeating-linear-gradient(45deg, rgba(255, 255, 255, 0.035) 0 10px, transparent 10px 20px);
    border: 1px dashed rgba(216, 222, 248, 0.28);
    color: rgba(216, 222, 248, 0.58);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.76rem;
}

#dream-button {
    width: 46px !important;
    min-width: 46px !important;
    max-width: 46px !important;
    min-height: 46px;
    height: 46px !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 999px;
    border: 0;
    background: rgba(22, 0, 85, 0.92);
    color: #fffaf2;
    font-weight: 800;
    font-size: 1.25rem;
    letter-spacing: 0;
}

#dream-button:hover {
    background: rgba(42, 20, 112, 0.96);
    filter: none;
}

@media (max-width: 900px) {
    .metrics-bar {
        grid-template-columns: 1fr;
    }

    .shell {
        padding: 18px 12px 36px;
    }

    .cloud-row {
        flex-direction: column;
    }

    .mute-panel {
        max-width: 180px !important;
    }

    .dream-settings-toggle {
        top: 18px;
        right: 18px;
    }

    .dream-settings-panel {
        top: 70px;
        right: 18px;
    }
}
"""

APP_HEAD = """
<script>
(() => {
  const applyDreamAudioVolumes = () => {
    const narrationSlider = document.getElementById('dream-narration-volume');
    const musicSlider = document.getElementById('dream-music-volume');
    const narrationVolume = Number(narrationSlider?.value || 1);
    const musicVolume = Number(musicSlider?.value || 0.22);
    const bgMusic = document.getElementById('dream-bg-music');
    if (bgMusic) bgMusic.volume = musicVolume;
    document.querySelectorAll('audio[data-role="narration"]').forEach((audio) => {
      audio.volume = narrationVolume;
    });
  };

  const startDreamMusic = () => {
    const bgMusic = document.getElementById('dream-bg-music');
    if (!bgMusic) return;
    applyDreamAudioVolumes();
    bgMusic.play().catch(() => {});
  };

  const stopDreamMusic = () => {
    const bgMusic = document.getElementById('dream-bg-music');
    if (!bgMusic) return;
    bgMusic.pause();
    bgMusic.currentTime = 0;
  };

  const bindDreamStars = () => {
    const row = document.querySelector('.star-feedback-row');
    if (!row || row.dataset.starBound === 'true') return;
    const buttons = [1, 2, 3, 4, 5]
      .map((rating) => document.getElementById(`star-button-${rating}`)?.querySelector('button'))
      .filter(Boolean);
    if (buttons.length !== 5) return;
    row.dataset.starBound = 'true';
    let selectedRating = 0;
    const paint = (count) => {
      buttons.forEach((button, index) => {
        button.classList.toggle('star-lit', index < count);
      });
    };
    buttons.forEach((button, index) => {
      const rating = index + 1;
      button.addEventListener('mouseenter', () => paint(rating));
      button.addEventListener('focus', () => paint(rating));
      button.addEventListener('click', () => {
        selectedRating = rating;
        paint(selectedRating);
      });
    });
    row.addEventListener('mouseleave', () => paint(selectedRating));
    row.addEventListener('focusout', () => setTimeout(() => paint(selectedRating), 0));
  };

  const bindDreamSettings = () => {
    const toggle = document.getElementById('dream-settings-toggle');
    const panel = document.getElementById('dream-settings-panel');
    const narrationSlider = document.getElementById('dream-narration-volume');
    const musicSlider = document.getElementById('dream-music-volume');
    if (!toggle || !panel || !narrationSlider || !musicSlider || toggle.dataset.audioBound === 'true') return;
    toggle.dataset.audioBound = 'true';
    toggle.addEventListener('click', () => {
      panel.classList.toggle('is-open');
    });
    document.addEventListener('click', (event) => {
      if (!panel.contains(event.target) && !toggle.contains(event.target)) {
        panel.classList.remove('is-open');
      }
    });
    narrationSlider.addEventListener('input', applyDreamAudioVolumes);
    musicSlider.addEventListener('input', applyDreamAudioVolumes);
    applyDreamAudioVolumes();
  };

  const bindDreamTriggers = () => {
    if (document.body.dataset.dreamTriggersBound === 'true') return;
    document.body.dataset.dreamTriggersBound = 'true';
    document.addEventListener('click', (event) => {
      if (event.target.closest('#dream-button')) {
        startDreamMusic();
      }
      if (event.target.closest('#cloud-touch')) {
        stopDreamMusic();
      }
    });
    document.addEventListener('keydown', (event) => {
      const target = event.target;
      if (event.key === 'Enter' && target && target.closest('#story-input')) {
        startDreamMusic();
      }
    });
  };

  const bindAllDreamUi = () => {
    bindDreamStars();
    bindDreamSettings();
    bindDreamTriggers();
    applyDreamAudioVolumes();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bindAllDreamUi);
  } else {
    bindAllDreamUi();
  }
  new MutationObserver(() => bindAllDreamUi()).observe(document.body, { childList: true, subtree: true });
})();
</script>
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(), head=APP_HEAD) as demo:
    gr.HTML(
        """
        <button id="dream-settings-toggle" class="dream-settings-toggle" type="button" aria-label="Audio settings">⚙</button>
        <section id="dream-settings-panel" class="dream-settings-panel" aria-label="Audio settings">
            <div class="dream-settings-title">Settings</div>
            <div class="dream-slider-stack">
                <div class="dream-slider-column">
                    <label class="dream-slider-label" for="dream-narration-volume">Voice</label>
                    <input id="dream-narration-volume" class="dream-volume-slider" type="range" min="0" max="1" step="0.01" value="1">
                </div>
                <div class="dream-slider-column">
                    <label class="dream-slider-label" for="dream-music-volume">Music</label>
                    <input id="dream-music-volume" class="dream-volume-slider" type="range" min="0" max="1" step="0.01" value="0.22">
                </div>
            </div>
        </section>
        """
    )
    gr.HTML(build_background_music_markup())
    gr.HTML(
        """
        <div class="shell">
            <div class="brand-bar" aria-label="DreamTales">
                <span class="brand-cloud"></span>
                <span class="brand-title">DreamTales</span>
            </div>
        </div>
        """
    )

    with gr.Column(elem_classes=["dream-stage"]):
        feedback_context_state = gr.State({})
        cloud_speech = gr.HTML(format_cloud_speech("Touch the cloud to start."))
        narration_audio = gr.HTML("")
        with gr.Row(elem_classes=["cloud-row"]):
            cloud_btn = gr.Button("• ᴗ •", elem_id="cloud-touch")
        with gr.Row(elem_classes=["cloud-input-panel"], visible=False) as input_panel:
            story_input = gr.Textbox(
                value="Tell me a story about kayaking",
                show_label=False,
                placeholder="Tell me a story about kayaking",
                lines=1,
                scale=6,
                elem_id="story-input",
            )
            dream_btn = gr.Button("↵", elem_id="dream-button", scale=0)
        with gr.Column(elem_classes=["storyboard-shell"]):
            scene_summary = gr.HTML("")
            with gr.Row(visible=False, elem_classes=["star-feedback-row"]) as feedback_rating:
                star_1 = gr.Button("★", elem_id="star-button-1", elem_classes=["star-feedback-button"], min_width=52)
                star_2 = gr.Button("★", elem_id="star-button-2", elem_classes=["star-feedback-button"], min_width=52)
                star_3 = gr.Button("★", elem_id="star-button-3", elem_classes=["star-feedback-button"], min_width=52)
                star_4 = gr.Button("★", elem_id="star-button-4", elem_classes=["star-feedback-button"], min_width=52)
                star_5 = gr.Button("★", elem_id="star-button-5", elem_classes=["star-feedback-button"], min_width=52)
            feedback_saved = gr.HTML("")

    cloud_btn.click(
        fn=run_cloud_intro,
        inputs=None,
        outputs=[cloud_speech, input_panel, scene_summary, feedback_rating, feedback_saved, feedback_context_state, narration_audio],
        show_progress="hidden",
    )
    dream_btn.click(
        fn=run_tinystories_scene_planner,
        inputs=story_input,
        outputs=[cloud_speech, input_panel, scene_summary, feedback_rating, feedback_saved, feedback_context_state, narration_audio],
        show_progress="hidden",
    )
    story_input.submit(
        fn=run_tinystories_scene_planner,
        inputs=story_input,
        outputs=[cloud_speech, input_panel, scene_summary, feedback_rating, feedback_saved, feedback_context_state, narration_audio],
        show_progress="hidden",
    )
    star_1.click(fn=lambda ctx: save_feedback_star(1, ctx), inputs=feedback_context_state, outputs=feedback_saved, show_progress="hidden")
    star_2.click(fn=lambda ctx: save_feedback_star(2, ctx), inputs=feedback_context_state, outputs=feedback_saved, show_progress="hidden")
    star_3.click(fn=lambda ctx: save_feedback_star(3, ctx), inputs=feedback_context_state, outputs=feedback_saved, show_progress="hidden")
    star_4.click(fn=lambda ctx: save_feedback_star(4, ctx), inputs=feedback_context_state, outputs=feedback_saved, show_progress="hidden")
    star_5.click(fn=lambda ctx: save_feedback_star(5, ctx), inputs=feedback_context_state, outputs=feedback_saved, show_progress="hidden")


if __name__ == "__main__":
    demo.launch()
