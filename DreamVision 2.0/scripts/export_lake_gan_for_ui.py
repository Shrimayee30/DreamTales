from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DREAMVISION_2_ROOT = Path(__file__).resolve().parents[1]
MODEL_SRC_PATH = DREAMVISION_2_ROOT / "src" / "lake_background_gan.py"
DEFAULT_CHECKPOINT_DIR = DREAMVISION_2_ROOT / "outputs" / "lake_background_story_256" / "checkpoints"


def find_latest_checkpoint(checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> Path:
    checkpoints = sorted(checkpoint_dir.glob("lake_story_epoch_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No lake GAN checkpoints found in {checkpoint_dir}")
    return checkpoints[-1]


def load_lake_module():
    spec = importlib.util.spec_from_file_location("lake_background_gan", MODEL_SRC_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {MODEL_SRC_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the trained lake GAN generator for DreamTales UI inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Training checkpoint to export. Defaults to the latest lake_story_epoch_*.pt checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=DREAMVISION_2_ROOT / "models")
    parser.add_argument("--sample-dir", type=Path, default=DREAMVISION_2_ROOT / "outputs" / "lake_background_story_256" / "ui_export_test")
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260427)
    args = parser.parse_args()

    lake = load_lake_module()
    checkpoint_path = args.checkpoint or find_latest_checkpoint()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.sample_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    generator = lake.load_generator_from_checkpoint(checkpoint_path, device=device)

    generator_state_path = args.output_dir / "lake_background_generator.pt"
    torchscript_path = args.output_dir / "lake_background_generator.torchscript.pt"

    torch.save(
        {
            "model_name": "DreamTales lake background generator",
            "latent_dim": lake.LATENT_DIM,
            "ngf": lake.NGF,
            "num_channels": lake.NUM_CHANNELS,
            "image_size": 256,
            "generator_state_dict": generator.state_dict(),
            "source_checkpoint": str(checkpoint_path),
            "fresh_generation_note": "This file stores learned generator weights. Each UI scene should sample fresh random noise.",
        },
        generator_state_path,
    )

    example_noise = torch.randn(1, lake.LATENT_DIM, 1, 1, device=device)
    traced = torch.jit.trace(generator, example_noise)
    traced.save(torchscript_path)

    sample_paths = []
    images = lake.generate_lake_backgrounds_for_story(
        generator=generator,
        scene_count=args.sample_count,
        base_seed=args.seed,
        device=device,
        cleanup=True,
    )
    for index, image in enumerate(images):
        sample_path = args.sample_dir / f"lake_ui_export_sample_{index + 1:02d}.png"
        image.save(sample_path)
        sample_paths.append(sample_path)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Saved generator state: {generator_state_path}")
    print(f"Saved TorchScript model: {torchscript_path}")
    print("Fresh generation: call the model once per scene with new random noise or a new seed.")
    print("Saved original generated test samples:")
    for sample_path in sample_paths:
        print(f"  {sample_path}")


if __name__ == "__main__":
    main()
