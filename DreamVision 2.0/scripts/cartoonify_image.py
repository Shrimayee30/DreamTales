from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from PIL import Image


DREAMVISION_2_ROOT = Path(__file__).resolve().parents[1]
MODEL_SRC_PATH = DREAMVISION_2_ROOT / "src" / "lake_background_gan.py"


def load_lake_module():
    spec = importlib.util.spec_from_file_location("lake_background_gan", MODEL_SRC_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {MODEL_SRC_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Cartoonify a generated DreamTales background image.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or args.input.with_name(f"{args.input.stem}_cartoon.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lake = load_lake_module()
    image = Image.open(args.input).convert("RGB")
    cartoon = lake.cartoonify_image(image)
    cartoon.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
