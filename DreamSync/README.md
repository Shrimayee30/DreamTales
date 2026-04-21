# DreamSync

DreamSync is the orchestration layer for DreamTales. It turns a generated or user-written story into scene-level instructions that can be shared by the UI, DreamVision, and DreamTunes.

DreamSync is responsible for:

- breaking a story into scenes
- running lightweight sentiment analysis
- mapping scenes to DreamVision labels
- building image prompts
- building music prompts
- creating a timing plan for syncing text, image, and audio outputs

## Folder Layout

```text
DreamSync/
  configs/
    dreamsync_smoke.json
  outputs/
    plans/
  src/
    condition_mapper.py
    pipeline.py
    prompt_builder.py
    rules.py
    scene_splitter.py
    schema.py
    sentiment.py
    sync_story.py
  README.md
  requirements.txt
```

## Install

DreamSync currently uses only the Python standard library:

```bash
pip install -r DreamSync/requirements.txt
```

The current sentiment analyzer is intentionally heuristic, so it does not require downloading an NLP model. This keeps the UI fast and avoids adding another heavyweight dependency before the pipeline shape is stable.

## Run a Smoke Sync

From the project root:

```bash
python DreamSync/src/sync_story.py
```

Write a sync plan for a custom story:

```bash
python DreamSync/src/sync_story.py \
  --story "A warm cloud carries two friends through a quiet moonlit park." \
  --output DreamSync/outputs/plans/custom_sync_plan.json
```

The output is a JSON file containing:

- scene text
- sentiment label and score
- DreamVision labels
- image prompt
- DreamTunes music prompt
- start time and duration

## Python API

```python
import sys

sys.path.append("DreamSync/src")

from pipeline import analyze_story

plan = analyze_story("A calm moonlit story with friends in a park.")
print(plan.to_dict())
```

## Sync Plan Shape

Each scene includes:

```text
scene_number
title
text
sentiment
sentiment_score
character
action
location
mood
image_prompt
music_prompt
start_seconds
duration_seconds
```

The UI can use the same scene plan to:

- show scene cards
- request images from DreamVision
- request background music from DreamTunes
- schedule text, image, and audio playback together

## Current Design Choice

DreamSync starts with simple rules instead of a large sentiment model. This is enough for the current DreamTales mood labels and keeps the module easy to inspect.

Later upgrades could include:

- Hugging Face sentiment models
- emotion classification
- beat detection for audio timing
- subtitles or karaoke-style word timing
- scene transition scoring
