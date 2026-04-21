import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        condition_dim: int = 23,
        ngf: int = 64,
        num_channels: int = 3,
    ) -> None:
        super().__init__()

        input_dim = latent_dim + condition_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, condition], dim=1)
        return self.model(x)


class ConditionalDiscriminator(nn.Module):
    def __init__(
        self,
        condition_dim: int = 23,
        ndf: int = 64,
        num_channels: int = 3,
        image_size: int = 64,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.condition_dim = condition_dim

        self.model = nn.Sequential(
            nn.Conv2d(num_channels + condition_dim, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch_size = image.size(0)
        condition_map = condition.expand(batch_size, self.condition_dim, self.image_size, self.image_size)
        x = torch.cat([image, condition_map], dim=1)
        return self.model(x).view(-1)


def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__

    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)