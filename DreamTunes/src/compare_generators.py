from __future__ import annotations

from pathlib import Path

from config import DreamTunesConfig, parse_args
from generate_audioldm import generate_with_audioldm
from generate_musicgen import generate_with_musicgen
from utils import ensure_dir, set_seed, write_metadata


def main() -> None:
    args = parse_args()
    config = DreamTunesConfig.from_json(args.config)

    if args.scene is not None:
        config.scene = args.scene
    if args.provider is not None:
        config.provider = args.provider
    if args.duration_seconds is not None:
        config.duration_seconds = args.duration_seconds

    set_seed(config.seed)
    ensure_dir(config.output_dir)
    metadata_dir = ensure_dir(config.metadata_dir)

    results = []
    if config.provider in {"musicgen", "both"}:
        print("Generating with MusicGen...")
        results.append(generate_with_musicgen(config))

    if config.provider in {"audioldm", "both"}:
        print("Generating with AudioLDM...")
        results.append(generate_with_audioldm(config))

    metadata_path = Path(metadata_dir) / "latest_generation.json"
    write_metadata(metadata_path, {"config": config.to_dict(), "results": results})

    print(f"Saved metadata to: {metadata_path}")
    for result in results:
        print(f"{result['provider']}: {result['output_path']}")


if __name__ == "__main__":
    main()

