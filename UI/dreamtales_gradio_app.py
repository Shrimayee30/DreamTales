import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DREAMVISION_ROOT = PROJECT_ROOT / "DreamVision"
DREAMSYNC_SRC = PROJECT_ROOT / "DreamSync" / "src"
SAMPLES_DIR = DREAMVISION_ROOT / "outputs" / "samples"
CHECKPOINT_PATH = DREAMVISION_ROOT / "outputs" / "checkpoints" / "conditional_generator_epoch_010.pt"
UI_OUTPUT_DIR = SAMPLES_DIR / "ui_generated"

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


CUSTOM_CSS = """
.gradio-container {
    background:
        radial-gradient(circle at 15% 20%, rgba(251, 202, 121, 0.18), transparent 24%),
        radial-gradient(circle at 85% 15%, rgba(128, 170, 255, 0.22), transparent 22%),
        linear-gradient(180deg, #07101f 0%, #0d1733 48%, #040814 100%);
    color: #f4f6ff;
    font-family: "Avenir Next", "Trebuchet MS", sans-serif;
}

.shell {
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 18px 42px;
}

.hero {
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 28px;
    background: linear-gradient(145deg, rgba(18, 30, 61, 0.9), rgba(9, 16, 31, 0.88));
    box-shadow: 0 28px 80px rgba(0, 0, 0, 0.28);
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.24em;
    color: #8ab4ff;
    font-size: 0.82rem;
    margin-bottom: 14px;
}

.title {
    font-size: clamp(2.6rem, 6vw, 4.8rem);
    line-height: 0.95;
    font-weight: 800;
    margin-bottom: 14px;
}

.subtitle {
    max-width: 680px;
    color: #d8def8;
    font-size: 1.05rem;
    line-height: 1.7;
}

.panel {
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 20px;
    backdrop-filter: blur(14px);
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

#dream-button {
    min-height: 54px;
    border-radius: 999px;
    border: 0;
    background: linear-gradient(90deg, #ffd27c 0%, #f89b4e 35%, #82a9ff 100%);
    color: #08101f;
    font-weight: 800;
    letter-spacing: 0.06em;
}

#dream-button:hover {
    filter: brightness(1.03);
}

.gradio-textbox textarea {
    min-height: 220px !important;
}

@media (max-width: 900px) {
    .metrics-bar {
        grid-template-columns: 1fr;
    }

    .shell {
        padding: 18px 12px 36px;
    }
}
"""


with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
    gr.HTML(
        """
        <div class="shell">
            <section class="hero">
                <div class="eyebrow">Dream Studio</div>
                <div class="title">DreamVision 2.0</div>
                <div class="subtitle">
                    Turn a bedtime story into a three-scene visual journey powered by the DreamVision generator.
                    The studio reads the story, extracts emotional cues, and renders a small sequence of dream frames.
                </div>
            </section>
        </div>
        """
    )

    with gr.Row(elem_classes=["shell"]):
        with gr.Column(scale=5, elem_classes=["panel"]):
            preset = gr.Dropdown(
                choices=list(SCENE_PRESETS.keys()),
                value="Moonlit Memory",
                label="Launch Pad",
            )
            story_input = gr.Textbox(
                value=default_story(),
                label="Dream Prompt",
                placeholder="Describe a soft, magical dream world...",
            )
            with gr.Row():
                load_preset_btn = gr.Button("Load Preset")
                dream_btn = gr.Button("Render Dream Sequence", elem_id="dream-button")
        with gr.Column(scale=7):
            status = gr.HTML("<div class='status-card'>DreamVision 2.0 is waiting for a story.</div>")
            scene_summary = gr.HTML("")
            gallery = gr.Gallery(
                label="Dream Frames",
                show_label=True,
                columns=3,
                height="auto",
                object_fit="cover",
            )

    load_preset_btn.click(use_preset_story, inputs=preset, outputs=story_input)
    dream_btn.click(
        fn=run_dreamvision,
        inputs=[story_input, preset],
        outputs=[status, scene_summary, gallery],
    )


if __name__ == "__main__":
    demo.launch()
