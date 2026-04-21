import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class TrainingConfig:
    model_name: str
    dataset_name: str = "roneneldan/TinyStories"
    dataset_text_field: str = "text"
    prompt_mode: str = "synthetic_instruction"
    output_dir: str = "DreamCore/outputs/run"
    max_seq_length: int = 512
    max_train_samples: int | None = None
    max_eval_samples: int | None = 2000
    num_train_epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    logging_steps: int = 10
    save_steps: int = 250
    eval_steps: int = 250
    save_total_limit: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    gradient_checkpointing: bool = True
    use_bf16: bool = False
    use_4bit: bool = False
    seed: int = 42
    trust_remote_code: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        data = json.loads(Path(path).read_text())
        valid_names = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_names}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama or Mistral on TinyStories with LoRA.")
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON config file.")
    return parser.parse_args()
