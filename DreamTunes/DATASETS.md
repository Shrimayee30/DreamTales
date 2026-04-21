# DreamTunes Dataset Notes

DreamTunes is currently set up as a pretrained model comparison module. No dataset is required for the first experiment because both generators are text-conditioned models loaded from pretrained checkpoints.

## Current Experiment

No local dataset is used yet.

Inputs are scene descriptions from DreamTales, transformed into music prompts with:

- scene text
- mood
- instrumental/background-score constraints
- child-friendly dream-story tone

## Candidate Datasets for Evaluation or Fine-Tuning

These are the dataset directions to choose from before we add download scripts or training code:

### MusicCaps

MusicCaps is a captioned music dataset commonly used for text-to-music evaluation. It is the best first choice for prompt-quality evaluation because it pairs music clips with natural-language music captions.

Recommended use in DreamTunes:

- evaluate prompt-to-music alignment
- build a small prompt benchmark
- compare MusicGen vs. AudioLDM output quality

Decision needed: use it only for evaluation, or build a fine-tuning path later.

### AudioCaps

AudioCaps is a captioned audio dataset. It includes broader sound events, not only music.

Recommended use in DreamTunes:

- evaluate AudioLDM on scene ambience and soundscape generation
- test prompts that mix music with environmental audio

Decision needed: whether DreamTunes should generate only background music, or also ambient sound effects.

### DreamTales Scene Prompts

The project can also create its own small evaluation set from DreamCore/DreamVision scene descriptions.

Recommended use in DreamTunes:

- create 20-50 project-specific prompts
- label each prompt with mood, energy, instrumentation, and scene context
- use human listening notes to choose the better generator

Decision needed: whether to prioritize project-specific qualitative evaluation before external datasets.

## Current Recommendation

Start without training data:

1. Generate a small set of DreamTales scene prompts.
2. Compare `facebook/musicgen-small` and `cvssp/audioldm-s-full-v2`.
3. Keep notes on alignment, mood, loopability, and noise.
4. Add MusicCaps only if we need a more formal evaluation benchmark.

