from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import utils


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 256, ngf: int = 64, num_channels: int = 3) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
        )
        self.upsampler = nn.Sequential(
            UpBlock(ngf * 16, ngf * 8),
            UpBlock(ngf * 8, ngf * 4),
            UpBlock(ngf * 4, ngf * 2),
            UpBlock(ngf * 2, ngf),
            UpBlock(ngf, ngf // 2),
            UpBlock(ngf // 2, ngf // 4),
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ngf // 4, num_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.to_rgb(self.upsampler(self.project(noise)))


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return (images.clamp(-1, 1) + 1) / 2


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = denormalize(image).mul(255).byte()
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def story_cleanup(image: Image.Image) -> Image.Image:
    """Make GAN outputs calmer and more storybook-friendly without hiding the lake layout."""
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Color(image).enhance(1.18)
    image = ImageEnhance.Brightness(image).enhance(1.06)
    image = ImageEnhance.Contrast(image).enhance(0.88)
    image = image.filter(ImageFilter.SMOOTH_MORE)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.4, percent=85, threshold=4))
    return image


def make_contact_sheet(paths: list[Path], output_path: Path, columns: int = 6) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        return
    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * height
        sheet.paste(image, (x, y))
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export final lake story candidates from the trained GAN.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("DreamVision 2.0/outputs/lake_background_story_256/checkpoints/lake_story_epoch_180.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DreamVision 2.0/outputs/lake_background_story_256/final_story_exports"),
    )
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--ngf", type=int, default=64)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    clean_dir = args.output_dir / "storybook_clean"
    raw_dir.mkdir(exist_ok=True)
    clean_dir.mkdir(exist_ok=True)

    device = torch.device("cpu")
    # This is a local checkpoint created by our notebook; weights_only=False is
    # needed because the saved config contains Path objects in addition to tensors.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generator = Generator(latent_dim=args.latent_dim, ngf=args.ngf, num_channels=3).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    torch.manual_seed(args.seed)
    noise = torch.randn(args.count, args.latent_dim, 1, 1, device=device)
    with torch.no_grad():
        samples = generator(noise).cpu()

    raw_paths: list[Path] = []
    clean_paths: list[Path] = []
    for index, sample in enumerate(samples, start=1):
        raw_image = tensor_to_pil(sample)
        clean_image = story_cleanup(raw_image)

        raw_path = raw_dir / f"lake_raw_{index:02d}.png"
        clean_path = clean_dir / f"lake_story_clean_{index:02d}.png"
        raw_image.save(raw_path)
        clean_image.save(clean_path)
        raw_paths.append(raw_path)
        clean_paths.append(clean_path)

    utils.save_image(denormalize(samples), args.output_dir / "raw_grid.png", nrow=6)
    make_contact_sheet(clean_paths, args.output_dir / "storybook_clean_grid.png", columns=6)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Saved {len(raw_paths)} raw candidates to {raw_dir}")
    print(f"Saved {len(clean_paths)} storybook candidates to {clean_dir}")
    print(f"Contact sheets: {args.output_dir / 'raw_grid.png'}")
    print(f"Contact sheets: {args.output_dir / 'storybook_clean_grid.png'}")


if __name__ == "__main__":
    main()
