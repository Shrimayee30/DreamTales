import shutil
from pathlib import Path

import pandas as pd

from src.config import (
    DANBOORU_IMAGES_DIR,
    FILTERED_METADATA_DIR,
    SCENE_IMAGES_DIR,
)
from src.utils import ensure_project_dirs


VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def find_image_by_id(image_dir: Path, image_id: str) -> Path | None:
    for ext in VALID_EXTENSIONS:
        candidate = image_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    ensure_project_dirs([SCENE_IMAGES_DIR])

    filtered_csv = FILTERED_METADATA_DIR / "filtered_scene_metadata.csv"
    if not filtered_csv.exists():
        raise FileNotFoundError(
            f"Filtered metadata not found at: {filtered_csv}"
        )

    df = pd.read_csv(filtered_csv)

    if "id" not in df.columns:
        raise ValueError("Expected an 'id' column in filtered_scene_metadata.csv")

    copied_count = 0
    missing_count = 0

    for _, row in df.iterrows():
        image_id = str(row["id"])
        source_path = find_image_by_id(DANBOORU_IMAGES_DIR, image_id)

        if source_path is None:
            missing_count += 1
            continue

        destination_path = SCENE_IMAGES_DIR / source_path.name
        shutil.copy2(source_path, destination_path)
        copied_count += 1

    print(f"Total metadata rows: {len(df)}")
    print(f"Copied images: {copied_count}")
    print(f"Missing images: {missing_count}")
    print(f"Scene image folder ready at: {SCENE_IMAGES_DIR}")


if __name__ == "__main__":
    main()