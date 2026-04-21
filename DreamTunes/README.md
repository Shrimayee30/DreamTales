# DreamTunes

DreamTunes experiments with text-to-audio models for generating background music from DreamTales scene descriptions. It is designed as the third model module in the project, alongside `DreamCore` for stories and `DreamVision` for scene images.

The first version compares two pretrained generators:

- MusicGen through Hugging Face Transformers
- AudioLDM through Hugging Face Diffusers

## Why Start With Pretrained Models?

Training text-to-music models requires a large captioned audio dataset and substantial compute. For the first experiment, DreamTunes uses pretrained models so we can quickly compare whether MusicGen or AudioLDM better matches DreamTales scenes.

No dataset is required for this first pass. See `DATASETS.md` for the dataset options we should decide on before adding evaluation or fine-tuning.

## Folder Layout

```text
DreamTunes/
  configs/
    dreamtunes_smoke.json
  outputs/
    audio/
    metadata/
  src/
    config.py
    compare_generators.py
    generate_audioldm.py
    generate_musicgen.py
    prompts.py
    utils.py
  DATASETS.md
  README.md
  requirements.txt
```

## Install

From the project root:

```bash
pip install -r DreamTunes/requirements.txt
```

The first generation run downloads model weights from Hugging Face, so it can take a while and requires internet access.

## Generate Background Music

Run both generators using the smoke config:

```bash
python DreamTunes/src/compare_generators.py --config DreamTunes/configs/dreamtunes_smoke.json
```

Generate with only MusicGen:

```bash
python DreamTunes/src/compare_generators.py \
  --config DreamTunes/configs/dreamtunes_smoke.json \
  --provider musicgen
```

Generate with only AudioLDM:

```bash
python DreamTunes/src/compare_generators.py \
  --config DreamTunes/configs/dreamtunes_smoke.json \
  --provider audioldm
```

Override the scene from the command line:

```bash
python DreamTunes/src/compare_generators.py \
  --provider musicgen \
  --scene "A warm family dream in a glowing neighborhood shop."
```

Generated audio is saved to:

```text
DreamTunes/outputs/audio/
```

Generation metadata is saved to:

```text
DreamTunes/outputs/metadata/latest_generation.json
```

## Model Notes

### MusicGen

Default checkpoint:

```text
facebook/musicgen-small
```

MusicGen is the stronger first candidate for instrumental background music because it is specifically built for text-to-music generation.

### AudioLDM

Default checkpoint:

```text
cvssp/audioldm-s-full-v2
```

AudioLDM can generate text-conditioned audio, including music and sound effects. It may be useful if DreamTunes should eventually create ambient soundscapes as well as music.

## Decision Points

Before expanding DreamTunes, decide:

1. Should DreamTunes generate only background music, or also ambient sound effects?
2. Should evaluation use external datasets such as MusicCaps, or a small DreamTales-specific scene prompt set?
3. Should we keep both generators, or choose one after listening tests?

