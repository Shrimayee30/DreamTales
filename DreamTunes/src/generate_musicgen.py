from __future__ import annotations

from pathlib import Path

import scipy.io.wavfile
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from config import DreamTunesConfig
from prompts import build_music_prompt
from utils import get_device, resolve_torch_dtype


def generate_with_musicgen(config: DreamTunesConfig, scene: str | None = None) -> dict[str, str | int | float]:
    device = get_device()
    dtype = resolve_torch_dtype(device)
    music_prompt = build_music_prompt(scene or config.scene, config.mood)

    processor = AutoProcessor.from_pretrained(config.musicgen_model)
    model = MusicgenForConditionalGeneration.from_pretrained(
        config.musicgen_model,
        torch_dtype=dtype,
    ).to(device)

    inputs = processor(
        text=[music_prompt.prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    audio_values = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=config.musicgen_max_new_tokens,
        temperature=config.musicgen_temperature,
        top_k=config.musicgen_top_k,
    )

    sampling_rate = model.config.audio_encoder.sampling_rate
    waveform = audio_values[0, 0].detach().cpu().float().numpy()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "musicgen_scene.wav"
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=waveform)

    return {
        "provider": "musicgen",
        "model": config.musicgen_model,
        "prompt": music_prompt.prompt,
        "output_path": str(output_path),
        "sampling_rate": sampling_rate,
        "duration_seconds": len(waveform) / sampling_rate,
    }

