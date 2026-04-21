from pathlib import Path

import torch

from src.config import (
    ACTION_CLASSES,
    CHARACTER_CLASSES,
    CHECKPOINT_DIR,
    CONDITION_DIM,
    IMAGE_SIZE,
    LATENT_DIM,
    LOCATION_CLASSES,
    MOOD_CLASSES,
    NGF,
    NUM_CHANNELS,
    SAMPLES_DIR,
)
from src.model import ConditionalGenerator
from src.utils import (
    ensure_project_dirs,
    get_device,
    save_image_grid,
)


def build_condition_vector(
    character: str,
    action: str,
    location: str,
    mood: str,
    device: torch.device,
) -> torch.Tensor:
    character_vec = torch.zeros(len(CHARACTER_CLASSES), device=device)
    action_vec = torch.zeros(len(ACTION_CLASSES), device=device)
    location_vec = torch.zeros(len(LOCATION_CLASSES), device=device)
    mood_vec = torch.zeros(len(MOOD_CLASSES), device=device)

    character_vec[CHARACTER_CLASSES.index(character)] = 1.0
    action_vec[ACTION_CLASSES.index(action)] = 1.0
    location_vec[LOCATION_CLASSES.index(location)] = 1.0
    mood_vec[MOOD_CLASSES.index(mood)] = 1.0

    condition = torch.cat([character_vec, action_vec, location_vec, mood_vec], dim=0)
    return condition.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


def load_generator(checkpoint_path: Path, device: torch.device) -> ConditionalGenerator:
    generator = ConditionalGenerator(
        latent_dim=LATENT_DIM,
        condition_dim=CONDITION_DIM,
        ngf=NGF,
        num_channels=NUM_CHANNELS,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["model_state_dict"])
    generator.eval()
    return generator


def main() -> None:
    device = get_device()
    ensure_project_dirs([SAMPLES_DIR])

    checkpoint_path = CHECKPOINT_DIR / "conditional_generator_epoch_010.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    generator = load_generator(checkpoint_path, device)

    scenes = [
        ("mother_child", "walking", "store", "warm"),
        ("friends", "holding_hands", "park", "calm"),
        ("none", "none", "forest", "night"),
    ]

    all_images = []

    with torch.no_grad():
        for character, action, location, mood in scenes:
            noise = torch.randn(4, LATENT_DIM, 1, 1, device=device)
            condition = build_condition_vector(
                character=character,
                action=action,
                location=location,
                mood=mood,
                device=device,
            )
            condition = condition.repeat(4, 1, 1, 1)

            fake_images = generator(noise, condition)
            all_images.append(fake_images)

            safe_name = f"{character}_{action}_{location}_{mood}".replace(" ", "_")
            save_image_grid(
                fake_images,
                SAMPLES_DIR / f"generated_{safe_name}.png",
                nrow=2,
            )

    combined = torch.cat(all_images, dim=0)
    save_image_grid(
        combined,
        SAMPLES_DIR / "generated_scene_grid.png",
        nrow=4,
    )

    print("Saved generated scene samples to outputs/samples/")


if __name__ == "__main__":
    main()