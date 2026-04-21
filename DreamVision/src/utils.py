import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

from src.config import (
    ACTION_CLASSES,
    CHARACTER_CLASSES,
    LOCATION_CLASSES,
    MOOD_CLASSES,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_project_dirs(paths: list[Path]) -> None:
    for path in paths:
        ensure_dir(path)


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor + 1) / 2


def save_image_grid(images: torch.Tensor, save_path: Path, nrow: int = 4) -> None:
    images = denormalize_image(images.detach().cpu()).clamp(0, 1)
    grid = make_grid(images, nrow=nrow)
    grid = grid.permute(1, 2, 0).numpy()
    grid = (grid * 255).astype(np.uint8)
    Image.fromarray(grid).save(save_path)


def one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()


def encode_condition_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    labels shape: [B, 4]
    fields: character, action, location, mood

    returns: [B, total_condition_dim]
    """
    character = one_hot(labels[:, 0], len(CHARACTER_CLASSES))
    action = one_hot(labels[:, 1], len(ACTION_CLASSES))
    location = one_hot(labels[:, 2], len(LOCATION_CLASSES))
    mood = one_hot(labels[:, 3], len(MOOD_CLASSES))

    return torch.cat([character, action, location, mood], dim=1)


def reshape_condition_for_generator(condition: torch.Tensor) -> torch.Tensor:
    return condition.unsqueeze(-1).unsqueeze(-1)