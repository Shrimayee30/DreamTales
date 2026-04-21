from pathlib import Path

import pandas as pd

from src.config import SCENE_IMAGES_DIR, SCENE_LABELS_PATH, LABELS_DIR
from src.utils import ensure_project_dirs


def main() -> None:
    ensure_project_dirs([LABELS_DIR])

    image_paths = sorted(
        [
            path for path in SCENE_IMAGES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )

    if not image_paths:
        raise ValueError(f"No images found in {SCENE_IMAGES_DIR}")

    df = pd.DataFrame({
        "filename": [path.name for path in image_paths],
        "character": ["none"] * len(image_paths),
        "action": ["none"] * len(image_paths),
        "location": ["none"] * len(image_paths),
        "mood": ["none"] * len(image_paths),
    })

    if SCENE_LABELS_PATH.exists():
        existing_df = pd.read_csv(SCENE_LABELS_PATH)
        df = df.merge(existing_df, on="filename", how="left", suffixes=("_new", ""))

        for col in ["character", "action", "location", "mood"]:
            df[col] = df[col].fillna(df[f"{col}_new"])
            df = df.drop(columns=[f"{col}_new"])

    df.to_csv(SCENE_LABELS_PATH, index=False)

    print(f"Created/updated label sheet: {SCENE_LABELS_PATH}")
    print(f"Total images listed: {len(df)}")


if __name__ == "__main__":
    main()