# DreamTales

DreamTales is a creative AI storytelling project that turns dream-like prompts into short stories and visual scene concepts. The project is organized into three main parts:

- `DreamCore`: fine-tunes a causal language model on TinyStories-style text for dream story generation.
- `DreamVision`: trains a conditional GAN to generate small scene images from labels such as character, action, location, and mood.
- `DreamTunes`: experiments with MusicGen and AudioLDM to generate background music from scene descriptions.
- `DreamSync`: breaks stories into scenes, analyzes sentiment, and creates sync plans for text, image, and audio generation.
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
  DreamTunes/
    configs/              # Text-to-audio experiment configs
    src/                  # MusicGen and AudioLDM generation code
    outputs/              # Generated audio and metadata
    DATASETS.md
    requirements.txt
    README.md
  DreamSync/
    configs/              # Scene sync experiment configs
    src/                  # Scene splitting, sentiment, prompt, and timing logic
    outputs/              # Generated sync plans
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

Install the DreamTunes dependencies:

```bash
pip install -r DreamTunes/requirements.txt
```

Install the DreamSync dependencies:

```bash
pip install -r DreamSync/requirements.txt
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

### Generate DreamTunes Background Music

Run both text-to-audio generators:

```bash
python DreamTunes/src/compare_generators.py --config DreamTunes/configs/dreamtunes_smoke.json
```

Run only MusicGen:

```bash
python DreamTunes/src/compare_generators.py --provider musicgen
```

Run only AudioLDM:

```bash
python DreamTunes/src/compare_generators.py --provider audioldm
```

### Create a DreamSync Scene Plan

Break a story into synced scene instructions:

```bash
python DreamSync/src/sync_story.py
```

DreamSync outputs scene text, sentiment, DreamVision labels, DreamTunes prompts, and timing metadata to:

```text
DreamSync/outputs/plans/
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

### DreamTunes Audio Dataset

DreamTunes currently uses pretrained text-to-audio models, so no local audio dataset is required for the first experiment. The module compares generated background music from scene descriptions using:

- `facebook/musicgen-small`
- `cvssp/audioldm-s-full-v2`

Candidate datasets for the next stage are documented in `DreamTunes/DATASETS.md`. The recommended next step is to evaluate with DreamTales-specific scene prompts first, then add MusicCaps if a formal text-to-music benchmark is needed.

## Outputs

Generated files are written to:

- `DreamCore/outputs/`: language-model adapters, checkpoints, and training configs.
- `DreamVision/outputs/checkpoints/`: conditional GAN generator and discriminator checkpoints.
- `DreamVision/outputs/samples/`: generated scene samples and preview grids.
- `DreamVision/outputs/logs/training_history.csv`: GAN training history.
- `DreamTunes/outputs/audio/`: generated background music files.
- `DreamTunes/outputs/metadata/`: generation prompts, settings, and output metadata.
- `DreamSync/outputs/plans/`: scene-level sync plans for UI orchestration.

## Author

Shrimayee Deshpande  
Check out my profile: `github.com/Shrimayee30`

## License

This project is licensed under the terms included in `LICENSE`.
