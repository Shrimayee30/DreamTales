import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal


ProviderName = Literal["musicgen", "audioldm", "both"]


@dataclass
class DreamTunesConfig:
    provider: ProviderName = "both"
    scene: str = "A calm moonlit forest where friends walk under glowing stars."
    mood: str = "calm"
    duration_seconds: float = 8.0
    seed: int = 42
    output_dir: str = "DreamTunes/outputs/audio"
    metadata_dir: str = "DreamTunes/outputs/metadata"

    musicgen_model: str = "facebook/musicgen-small"
    musicgen_max_new_tokens: int = 256
    musicgen_temperature: float = 1.0
    musicgen_top_k: int = 250

    audioldm_model: str = "cvssp/audioldm-s-full-v2"
    audioldm_num_inference_steps: int = 25
    audioldm_guidance_scale: float = 2.5
    audioldm_negative_prompt: str = "low quality, distorted, noisy, harsh vocals"

    @classmethod
    def from_json(cls, path: str | Path) -> "DreamTunesConfig":
        data = json.loads(Path(path).read_text())
        valid_names = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_names}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate background music for a DreamTales scene.")
    parser.add_argument("--config", type=str, default="DreamTunes/configs/dreamtunes_smoke.json")
    parser.add_argument("--scene", type=str, default=None, help="Override the scene description from config.")
    parser.add_argument("--provider", choices=["musicgen", "audioldm", "both"], default=None)
    parser.add_argument("--duration-seconds", type=float, default=None)
    return parser.parse_args()

