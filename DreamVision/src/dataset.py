from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import (
    ACTION_CLASSES,
    CHARACTER_CLASSES,
    LOCATION_CLASSES,
    MOOD_CLASSES,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ConditionalSceneDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        labels_csv: Path,
        image_size: int = 64,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.labels_df = pd.read_csv(labels_csv)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.character_to_idx = {name: i for i, name in enumerate(CHARACTER_CLASSES)}
        self.action_to_idx = {name: i for i, name in enumerate(ACTION_CLASSES)}
        self.location_to_idx = {name: i for i, name in enumerate(LOCATION_CLASSES)}
        self.mood_to_idx = {name: i for i, name in enumerate(MOOD_CLASSES)}

        self.rows = []
        for _, row in self.labels_df.iterrows():
            image_path = self.image_dir / row["filename"]
            if image_path.exists() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                self.rows.append(row)

        if not self.rows:
            raise ValueError("No valid labeled images found.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.image_dir / row["filename"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor([
            self.character_to_idx[row["character"]],
            self.action_to_idx[row["action"]],
            self.location_to_idx[row["location"]],
            self.mood_to_idx[row["mood"]],
        ], dtype=torch.long)

        return image, label


def build_conditional_dataloader(
    image_dir: Path,
    labels_csv: Path,
    image_size: int = 64,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ConditionalSceneDataset(
        image_dir=image_dir,
        labels_csv=labels_csv,
        image_size=image_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )