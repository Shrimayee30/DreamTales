from __future__ import annotations

from pathlib import Path

import scipy.io.wavfile
import torch
from diffusers import AudioLDMPipeline

from config import DreamTunesConfig
from prompts import build_music_prompt
from utils import get_device, resolve_torch_dtype


def generate_with_audioldm(config: DreamTunesConfig, scene: str | None = None) -> dict[str, str | int | float]:
    device = get_device()
    dtype = resolve_torch_dtype(device)
    music_prompt = build_music_prompt(scene or config.scene, config.mood)

    pipe = AudioLDMPipeline.from_pretrained(
        config.audioldm_model,
        torch_dtype=dtype,
    ).to(device)

    result = pipe(
        music_prompt.prompt,
        negative_prompt=config.audioldm_negative_prompt,
        num_inference_steps=config.audioldm_num_inference_steps,
        audio_length_in_s=config.duration_seconds,
        guidance_scale=config.audioldm_guidance_scale,
    )

    waveform = result.audios[0]
    sampling_rate = pipe.mel.get_sample_rate()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "audioldm_scene.wav"
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=waveform)

    return {
        "provider": "audioldm",
        "model": config.audioldm_model,
        "prompt": music_prompt.prompt,
        "output_path": str(output_path),
        "sampling_rate": sampling_rate,
        "duration_seconds": len(waveform) / sampling_rate,
    }

