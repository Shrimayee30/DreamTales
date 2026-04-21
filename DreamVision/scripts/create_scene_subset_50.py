import shutil
from pathlib import Path

import pandas as pd

from src.config import (
    DANBOORU_IMAGES_DIR,
    FILTERED_METADATA_DIR,
    LABELS_DIR,
    SCENE_SUBSET_DIR,
    SCENE_SUBSET_LABELS_PATH,
)
from src.utils import ensure_project_dirs


VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
TARGET_COUNT = 50


def find_image_by_id(image_dir: Path, image_id: str) -> Path | None:
    for ext in VALID_EXTENSIONS:
        candidate = image_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    ensure_project_dirs([SCENE_SUBSET_DIR, LABELS_DIR])

    filtered_csv = FILTERED_METADATA_DIR / "filtered_scene_metadata.csv"
    if not filtered_csv.exists():
        raise FileNotFoundError(f"Filtered metadata not found: {filtered_csv}")

    df = pd.read_csv(filtered_csv)

    if "id" not in df.columns:
        raise ValueError("Expected an 'id' column in filtered_scene_metadata.csv")

    selected_rows = []
    copied_count = 0

    for _, row in df.iterrows():
        if copied_count >= TARGET_COUNT:
            break

        image_id = str(row["id"])
        source_path = find_image_by_id(DANBOORU_IMAGES_DIR, image_id)

        if source_path is None:
            continue

        destination_path = SCENE_SUBSET_DIR / source_path.name
        shutil.copy2(source_path, destination_path)

        selected_rows.append({
            "filename": source_path.name,
            "character": "none",
            "action": "none",
            "location": "none",
            "mood": "none",
        })
        copied_count += 1

    labels_df = pd.DataFrame(selected_rows)
    labels_df.to_csv(SCENE_SUBSET_LABELS_PATH, index=False)

    print(f"Copied images: {copied_count}")
    print(f"Subset folder: {SCENE_SUBSET_DIR}")
    print(f"Label sheet created: {SCENE_SUBSET_LABELS_PATH}")


if __name__ == "__main__":
    main()