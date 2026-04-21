from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ScenePlan:
    scene_number: int
    title: str
    text: str
    sentiment: str
    sentiment_score: float
    character: str
    action: str
    location: str
    mood: str
    image_prompt: str
    music_prompt: str
    start_seconds: float
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SyncPlan:
    story: str
    word_count: int
    scene_count: int
    overall_sentiment: str
    overall_mood: str
    total_duration_seconds: float
    scenes: list[ScenePlan]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scenes"] = [scene.to_dict() for scene in self.scenes]
        return data

