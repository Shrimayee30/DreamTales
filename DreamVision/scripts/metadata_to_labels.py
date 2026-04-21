import pandas as pd
from pathlib import Path

metadata_path = Path("data/processed/labels/scene_subset_50_metadata.csv")
labels_path = Path("data/processed/labels/scene_subset_50_labels.csv")

df = pd.read_csv(metadata_path)

labels_df = pd.DataFrame({
    "filename": df["filename"],
    "character": "none",
    "action": "none",
    "location": "none",
    "mood": "none",
})

labels_df.to_csv(labels_path, index=False)

print(f"Created: {labels_path}")
print(f"Rows: {len(labels_df)}")