import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    BETA1,
    CHECKPOINT_DIR,
    CONDITION_DIM,
    IMAGE_SIZE,
    LATENT_DIM,
    LEARNING_RATE,
    LOGS_DIR,
    NDF,
    NGF,
    NUM_CHANNELS,
    NUM_EPOCHS,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SAMPLE_INTERVAL,
    SAMPLES_DIR,
    SCENE_IMAGES_DIR,
    SCENE_LABELS_PATH,
    SEED,
    TRAINING_HISTORY_PATH,
)
from src.dataset import build_conditional_dataloader
from src.model import ConditionalDiscriminator, ConditionalGenerator, weights_init
from src.utils import (
    encode_condition_labels,
    ensure_project_dirs,
    get_device,
    reshape_condition_for_generator,
    save_image_grid,
    set_seed,
)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )


def main() -> None:
    set_seed(SEED)

    ensure_project_dirs([
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUTS_DIR,
        CHECKPOINT_DIR,
        SAMPLES_DIR,
        LOGS_DIR,
    ])

    device = get_device()
    print(f"Using device: {device}")

    dataloader = build_conditional_dataloader(
        image_dir=SCENE_IMAGES_DIR,
        labels_csv=SCENE_LABELS_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print(f"Labeled batches per epoch: {len(dataloader)}")

    generator = ConditionalGenerator(
        latent_dim=LATENT_DIM,
        condition_dim=CONDITION_DIM,
        ngf=NGF,
        num_channels=NUM_CHANNELS,
    ).to(device)

    discriminator = ConditionalDiscriminator(
        condition_dim=CONDITION_DIM,
        ndf=NDF,
        num_channels=NUM_CHANNELS,
        image_size=IMAGE_SIZE,
    ).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)
    history = []

    print("Starting conditional GAN training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        g_running_loss = 0.0
        d_running_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")

        for real_images, labels in progress_bar:
            real_images = real_images.to(device)
            labels = labels.to(device)

            batch_size = real_images.size(0)

            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            condition = encode_condition_labels(labels).to(device)
            condition_g = reshape_condition_for_generator(condition)
            condition_d = condition.unsqueeze(-1).unsqueeze(-1)

            # -------------------------
            # Train Discriminator
            # -------------------------
            discriminator.zero_grad()

            real_output = discriminator(real_images, condition_d)
            d_loss_real = criterion(real_output, real_targets)

            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = generator(noise, condition_g)

            fake_output = discriminator(fake_images.detach(), condition_d)
            d_loss_fake = criterion(fake_output, fake_targets)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # -------------------------
            # Train Generator
            # -------------------------
            generator.zero_grad()

            gen_output = discriminator(fake_images, condition_d)
            g_loss = criterion(gen_output, real_targets)

            g_loss.backward()
            optimizer_g.step()

            g_running_loss += g_loss.item()
            d_running_loss += d_loss.item()

            progress_bar.set_postfix({
                "D_loss": f"{d_loss.item():.4f}",
                "G_loss": f"{g_loss.item():.4f}",
            })

        avg_d_loss = d_running_loss / len(dataloader)
        avg_g_loss = g_running_loss / len(dataloader)
        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "d_loss": avg_d_loss,
            "g_loss": avg_g_loss,
            "epoch_time_sec": epoch_time,
        })

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"D_loss: {avg_d_loss:.4f} "
            f"G_loss: {avg_g_loss:.4f} "
            f"Time: {epoch_time:.2f}s"
        )

        if epoch % SAMPLE_INTERVAL == 0:
            with torch.no_grad():
                sample_count = min(16, batch_size)
                fixed_condition = condition[:sample_count]
                fixed_condition_g = reshape_condition_for_generator(fixed_condition)
                fake_samples = generator(fixed_noise[:sample_count], fixed_condition_g)

            save_image_grid(
                fake_samples,
                SAMPLES_DIR / f"conditional_epoch_{epoch:03d}.png",
                nrow=4,
            )

        save_checkpoint(
            generator,
            optimizer_g,
            epoch,
            CHECKPOINT_DIR / f"conditional_generator_epoch_{epoch:03d}.pt",
        )
        save_checkpoint(
            discriminator,
            optimizer_d,
            epoch,
            CHECKPOINT_DIR / f"conditional_discriminator_epoch_{epoch:03d}.pt",
        )

        pd.DataFrame(history).to_csv(TRAINING_HISTORY_PATH, index=False)

    print("Conditional training complete.")
    print(f"Samples saved in: {SAMPLES_DIR}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"Training history saved in: {TRAINING_HISTORY_PATH}")


if __name__ == "__main__":
    main()