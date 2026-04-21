# DreamVision

DreamVision is the visual generation module for DreamTales. It trains a conditional GAN that generates small dream-scene images from structured labels such as character, action, location, and mood.

DreamVision is designed to work with the story scenes produced by `DreamCore` and the interactive experience in `UI/dreamtales_gradio_app.py`.

## What This Does

DreamVision:

1. Loads a labeled image subset from `DreamVision/data/raw/scene_subset_50/`.
2. Reads scene labels from `DreamVision/data/processed/labels/scene_subset_50_labels.csv`.
3. Encodes labels into a condition vector.
4. Trains a conditional generator and discriminator.
5. Saves checkpoints, generated samples, and training history under `DreamVision/outputs/`.
6. Provides a sample-generation script that loads a trained generator and creates scene grids.

## Folder Layout

```text
DreamVision/
  data/
    raw/
      danbooru_images/
      danbooru_metadata/
      scene_images/
      scene_subset_50/
    processed/
      labels/
        scene_subset_50_labels.csv
      metadata/
        filtered_scene_metadata.csv
  outputs/
    checkpoints/
    logs/
      training_history.csv
    samples/
  scripts/
    generate_samples.py
    validate_labels.py
    normalize_labels.py
    metadata_to_labels.py
    create_scene_subset_50.py
  src/
    config.py
    dataset.py
    model.py
    train.py
    utils.py
  requirements.txt
  README.md
```

## Installation and Setup

From the project root, create or activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the core packages used by DreamVision:

```bash
pip install torch torchvision pandas pillow tqdm numpy
```

If you are also running the full DreamTales project, install the root/module requirements as needed:

```bash
pip install -r DreamCore/requirements.txt
pip install -r DreamTunes/requirements.txt
pip install gradio jupyter requests
```

## Dataset Information

DreamVision currently uses a small labeled scene subset:

```text
DreamVision/data/raw/scene_subset_50/
DreamVision/data/processed/labels/scene_subset_50_labels.csv
```

Each label row should include:

```text
filename,character,action,location,mood
```

The current model config expects the following label values.

### Character Labels

- `none`
- `mother_child`
- `friends`
- `animal_pair`

### Action Labels

- `none`
- `walking`
- `holding_hands`
- `playing`
- `sleeping`
- `shopping`

### Location Labels

- `none`
- `store`
- `park`
- `forest`
- `bedroom`
- `street`
- `home`

### Mood Labels

- `none`
- `warm`
- `calm`
- `night`
- `sunny`
- `rainy`

These labels are one-hot encoded and concatenated into a single condition vector.

## Important Label Note

Before training, make sure the values in `scene_subset_50_labels.csv` match the classes in `DreamVision/src/config.py`.

For example, labels such as `happy`, `good`, `waterfall`, `bridge`, or `city` are not currently listed in the config. Either normalize those labels to the supported values or update `DreamVision/src/config.py` to include the additional classes before training.

Validate labels with:

```bash
python DreamVision/scripts/validate_labels.py
```

## Train DreamVision

From the project root:

```bash
python DreamVision/src/train.py
```

Training uses the settings in `DreamVision/src/config.py`:

- Image size: `64x64`
- Batch size: `16`
- Epochs: `30`
- Latent dimension: `100`
- Learning rate: `0.0002`
- Seed: `42`

During training, DreamVision saves:

- generator checkpoints
- discriminator checkpoints
- generated sample grids
- training loss history

## Generate Samples

After a checkpoint exists, run:

```bash
python DreamVision/scripts/generate_samples.py
```

By default, this script loads:

```text
DreamVision/outputs/checkpoints/conditional_generator_epoch_010.pt
```

It writes generated images to:

```text
DreamVision/outputs/samples/
```

The script currently generates sample grids for scenes such as:

- `mother_child`, `walking`, `store`, `warm`
- `friends`, `holding_hands`, `park`, `calm`
- `none`, `none`, `forest`, `night`

## Model Overview

DreamVision uses a conditional DCGAN-style architecture:

- `ConditionalGenerator`: combines random noise with a condition vector and upsamples into a `64x64` RGB image.
- `ConditionalDiscriminator`: receives both the image and condition map, then predicts whether the image is real or generated.

The condition vector is built from:

```text
character + action + location + mood
```

## Outputs

```text
DreamVision/outputs/checkpoints/
DreamVision/outputs/samples/
DreamVision/outputs/logs/training_history.csv
```

`training_history.csv` stores epoch-level discriminator loss, generator loss, and epoch time.

## Scripts

Useful scripts include:

- `DreamVision/scripts/validate_labels.py`: checks whether label CSV values match the configured classes.
- `DreamVision/scripts/generate_samples.py`: loads a trained generator and creates sample image grids.
- `DreamVision/scripts/normalize_labels.py`: utility for label cleanup.
- `DreamVision/scripts/metadata_to_labels.py`: creates a starter label CSV from metadata.

Some older data-preparation scripts may reference constants that are not currently defined in `DreamVision/src/config.py`. Review and update those scripts before using them for a fresh dataset pipeline.

## Troubleshooting

### No valid labeled images found

Check that every `filename` in the label CSV exists in:

```text
DreamVision/data/raw/scene_subset_50/
```

Also confirm the files use supported extensions:

```text
.jpg, .jpeg, .png, .webp
```

### Invalid label values

Run:

```bash
python DreamVision/scripts/validate_labels.py
```

Then normalize unsupported values or update the label class lists in `DreamVision/src/config.py`.

### Missing checkpoint

If `generate_samples.py` cannot find `conditional_generator_epoch_010.pt`, train the model first or edit the checkpoint path in the script.

## Relationship to DreamTales

DreamVision provides the image-generation layer for DreamTales:

- `DreamCore` writes the story.
- `DreamVision` generates or samples matching scene visuals.
- `DreamTunes` experiments with background music for the same scene.
- `UI` brings the pieces together in a Gradio interface.
