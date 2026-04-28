from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


LATENT_DIM = 256
NGF = 64
NUM_CHANNELS = 3


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


class LakeBackgroundGenerator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, ngf: int = NGF, num_channels: int = NUM_CHANNELS) -> None:
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


def load_generator_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device | str = "cpu",
    latent_dim: int = LATENT_DIM,
    ngf: int = NGF,
    num_channels: int = NUM_CHANNELS,
) -> LakeBackgroundGenerator:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator = LakeBackgroundGenerator(latent_dim=latent_dim, ngf=ngf, num_channels=num_channels).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return (images.clamp(-1, 1) + 1) / 2


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = denormalize(image.detach().cpu()).mul(255).byte()
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def storybook_cleanup(image: Image.Image) -> Image.Image:
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Color(image).enhance(1.18)
    image = ImageEnhance.Brightness(image).enhance(1.06)
    image = ImageEnhance.Contrast(image).enhance(0.88)
    image = image.filter(ImageFilter.SMOOTH_MORE)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.4, percent=85, threshold=4))
    return image


def cartoonify_image(image: Image.Image) -> Image.Image:
    """Local cartoon-style cleanup for GAN backgrounds before UI display."""
    base = storybook_cleanup(image.convert("RGB"))
    detailed = base.filter(ImageFilter.UnsharpMask(radius=1.1, percent=135, threshold=3))

    # Light posterization gives a painted/cartoon feel without washing out detail.
    posterized = ImageOps.posterize(detailed, bits=5)
    posterized = ImageEnhance.Color(posterized).enhance(1.08)
    posterized = ImageEnhance.Contrast(posterized).enhance(1.02)

    grayscale = ImageOps.grayscale(detailed).filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.invert(grayscale).filter(ImageFilter.GaussianBlur(radius=0.35))
    edges = ImageEnhance.Contrast(edges).enhance(1.35)
    edges_rgb = Image.merge("RGB", (edges, edges, edges))

    cartoon = ImageChops.multiply(posterized, edges_rgb)
    cartoon = Image.blend(detailed, cartoon, alpha=0.38)
    cartoon = cartoon.filter(ImageFilter.UnsharpMask(radius=0.8, percent=90, threshold=3))
    return ImageEnhance.Sharpness(cartoon).enhance(1.08)


def clean_sharp_image(image: Image.Image) -> Image.Image:
    """Sharper storybook cleanup when full cartoon edges make GAN noise too visible."""
    cleaned = storybook_cleanup(image.convert("RGB"))
    cleaned = ImageEnhance.Color(cleaned).enhance(1.08)
    cleaned = ImageEnhance.Contrast(cleaned).enhance(1.04)
    cleaned = cleaned.filter(ImageFilter.UnsharpMask(radius=1.2, percent=125, threshold=4))
    return ImageEnhance.Sharpness(cleaned).enhance(1.12)


def generate_lake_background(
    generator: nn.Module,
    seed: int,
    device: torch.device | str = "cpu",
    latent_dim: int = LATENT_DIM,
    cleanup: bool = True,
    cartoonify: bool = False,
    clean_sharp: bool = False,
) -> Image.Image:
    torch.manual_seed(seed)
    noise = torch.randn(1, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        sample = generator(noise).squeeze(0)
    image = tensor_to_pil(sample)
    if clean_sharp:
        return clean_sharp_image(image)
    if cartoonify:
        return cartoonify_image(image)
    return storybook_cleanup(image) if cleanup else image


def generate_lake_backgrounds_for_story(
    generator: nn.Module,
    scene_count: int,
    base_seed: int | None = None,
    device: torch.device | str = "cpu",
    latent_dim: int = LATENT_DIM,
    cleanup: bool = True,
    cartoonify: bool = False,
    clean_sharp: bool = False,
) -> list[Image.Image]:
    if base_seed is None:
        base_seed = torch.seed() % 1_000_000_000
    return [
        generate_lake_background(
            generator=generator,
            seed=base_seed + scene_index,
            device=device,
            latent_dim=latent_dim,
            cleanup=cleanup,
            cartoonify=cartoonify,
            clean_sharp=clean_sharp,
        )
        for scene_index in range(scene_count)
    ]
