# DreamCore

DreamCore fine-tunes a causal language model to write short dreamlike stories from the TinyStories dataset.

It is set up for parameter-efficient fine-tuning with LoRA so the same training code can target:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `mistralai/Mistral-7B-v0.1`

`TinyLlama` is the practical default. `Mistral 7B` is supported, but it needs substantially more memory.

## What This Does

DreamCore:

1. Downloads `roneneldan/TinyStories` from Hugging Face.
2. Converts each story into a simple prompt-and-story training pair.
3. Fine-tunes the base model with LoRA adapters.
4. Saves the adapter, tokenizer, and training config into `DreamCore/outputs/`.
5. Loads the fine-tuned adapter later to generate new stories from custom prompts.

## Folder Layout

```text
DreamCore/
  configs/
    tinyllama_qlora.json
    tinyllama_smoke.json
    mistral_lora.json
  outputs/
    .gitkeep
  requirements.txt
  README.md
  src/
    __init__.py
    config.py
    data.py
    generate.py
    train.py
```

## Install

Create or reuse a Python environment, then install dependencies:

```bash
pip install -r DreamCore/requirements.txt
```

If you are training on Linux with CUDA and want 4-bit QLoRA, also install `bitsandbytes`.

## Train TinyLlama

```bash
python DreamCore/src/train.py --config DreamCore/configs/tinyllama_qlora.json
```

## Run A Short Smoke Fine-Tune

```bash
python DreamCore/src/train.py --config DreamCore/configs/tinyllama_smoke.json
```

## Train Mistral 7B

```bash
python DreamCore/src/train.py --config DreamCore/configs/mistral_lora.json
```

## Generate A Story

After training finishes, point generation to the base model and adapter output folder:

```bash
python DreamCore/src/generate.py \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path DreamCore/outputs/tinyllama-tinystories/final_adapter \
  --prompt "a shy moon who wants to sing to the stars"
```

## Notes

- TinyStories is a text-only dataset, so DreamCore trains story generation rather than image generation.
- On Apple Silicon, LoRA fine-tuning is more realistic with `TinyLlama` than `Mistral 7B`.
- If memory is tight, reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps`.
- The configs use a synthetic prompt built from each story so the model learns prompt-to-story behavior rather than plain next-token continuation only.
