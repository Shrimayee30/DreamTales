# DreamTales

DreamTales is a creative AI storytelling project that turns dream-like prompts into short stories and visual scene concepts. The project is organized into three main parts:

- `DreamCore`: fine-tunes a causal language model on TinyStories-style text for dream story generation.
- `DreamVision`: trains a conditional GAN to generate small scene images from labels such as character, action, location, and mood.
- `UI`: provides a Gradio interface that connects story text, scene planning, and generated/sample visuals.

## Project Structure

```text
DreamTales/
  DreamCore/
    configs/              # TinyLlama and Mistral training configs
    src/                  # Text dataset, training, and generation code
    outputs/              # Fine-tuned adapters and checkpoints
    requirements.txt
    README.md
  DreamVision/
    data/
      raw/                # Raw image data and metadata
      processed/          # Processed labels and metadata
    scripts/              # Data preparation and sample generation scripts
    src/                  # Conditional GAN dataset, model, training, and utilities
    outputs/              # Checkpoints, generated samples, and training logs
    requirements.txt
    README.md
  UI/
    dreamtales_gradio_app.py
  LICENSE
  README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd DreamTales
```

### 2. Create a Python environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Install the DreamCore dependencies:

```bash
pip install -r DreamCore/requirements.txt
```

Install the additional packages used by DreamVision and the Gradio UI:

```bash
pip install pandas pillow torchvision tqdm gradio requests jupyter
```

If you are training on Linux with CUDA and want 4-bit QLoRA support for DreamCore, also install `bitsandbytes`.

## How to Run the Project

### Run the Gradio UI

From the project root:

```bash
python UI/dreamtales_gradio_app.py
```

The app loads scene presets, maps story text into scene labels, and uses DreamVision sample outputs or the trained conditional generator when the checkpoint is available at:

```text
DreamVision/outputs/checkpoints/conditional_generator_epoch_010.pt
```

### Train DreamCore

Run a quick smoke fine-tune:

```bash
python DreamCore/src/train.py --config DreamCore/configs/tinyllama_smoke.json
```

Run the full TinyLlama LoRA fine-tune:

```bash
python DreamCore/src/train.py --config DreamCore/configs/tinyllama_qlora.json
```

Generate a story after training:

```bash
python DreamCore/src/generate.py \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path DreamCore/outputs/tinyllama-tinystories/final_adapter \
  --prompt "a shy moon who wants to sing to the stars"
```

### Train DreamVision

DreamVision expects labeled scene images at:

```text
DreamVision/data/raw/scene_subset_50/
DreamVision/data/processed/labels/scene_subset_50_labels.csv
```

Train the conditional GAN:

```bash
python DreamVision/src/train.py
```

Generate sample image grids from a saved checkpoint:

```bash
python DreamVision/scripts/generate_samples.py
```

## How to Run the Notebook

The project notebook is located at:

```text
DreamVision/scripts/Untitled.ipynb
```

To open it locally:

```bash
jupyter notebook DreamVision/scripts/Untitled.ipynb
```

Or, with JupyterLab:

```bash
jupyter lab DreamVision/scripts/Untitled.ipynb
```

Run the notebook cells from top to bottom. The notebook includes data-download and setup logic for image metadata and should be run from the project root or checked carefully for path assumptions before execution.

## Dataset Information

### DreamCore Text Dataset

DreamCore uses the Hugging Face dataset:

- `roneneldan/TinyStories`

The training code downloads the dataset automatically through the `datasets` library. Each story is converted into a prompt-and-story training pair so the model learns to generate short, dreamlike narratives from custom prompts.

### DreamVision Image Dataset

DreamVision uses a small labeled scene-image dataset stored in the repository under:

```text
DreamVision/data/raw/
DreamVision/data/processed/
```

Important files and folders include:

- `DreamVision/data/raw/scene_subset_50/`: 50-image scene subset used for conditional GAN training.
- `DreamVision/data/processed/labels/scene_subset_50_labels.csv`: labels for the 50-image subset.
- `DreamVision/data/raw/danbooru_metadata/metadata.csv`: raw metadata used during filtering.
- `DreamVision/data/processed/metadata/filtered_scene_metadata.csv`: filtered metadata.

DreamVision labels each image using four condition groups:

- Character: `none`, `mother_child`, `friends`, `animal_pair`
- Action: `none`, `walking`, `holding_hands`, `playing`, `sleeping`, `shopping`
- Location: `none`, `store`, `park`, `forest`, `bedroom`, `street`, `home`
- Mood: `none`, `warm`, `calm`, `night`, `sunny`, `rainy`

These labels are encoded as condition vectors for the conditional GAN.

## Outputs

Generated files are written to:

- `DreamCore/outputs/`: language-model adapters, checkpoints, and training configs.
- `DreamVision/outputs/checkpoints/`: conditional GAN generator and discriminator checkpoints.
- `DreamVision/outputs/samples/`: generated scene samples and preview grids.
- `DreamVision/outputs/logs/training_history.csv`: GAN training history.

## Author

Shrimayee Deshpande  
Check out my profile: `github.com/Shrimayee30`

## License

This project is licensed under the terms included in `LICENSE`.
