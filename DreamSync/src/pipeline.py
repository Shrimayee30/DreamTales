from __future__ import annotations

from schema import ScenePlan, SyncPlan
from condition_mapper import infer_conditions
from prompt_builder import build_image_prompt, build_music_prompt
from scene_splitter import split_story_into_scenes
from sentiment import analyze_sentiment


def build_scene_title(scene_text: str, scene_number: int) -> str:
    preview = scene_text.strip().split(".")[0][:46].strip()
    return f"Scene {scene_number}: {preview or f'Dream beat {scene_number}'}"


def estimate_scene_duration(scene_text: str, min_seconds: float = 6.0, words_per_second: float = 2.4) -> float:
    word_count = len(scene_text.split())
    return round(max(min_seconds, word_count / words_per_second), 2)


def choose_overall_mood(scenes: list[ScenePlan]) -> str:
    moods = [scene.mood for scene in scenes if scene.mood != "none"]
    if not moods:
        return "none"
    return max(set(moods), key=moods.count)


def choose_overall_sentiment(scene_scores: list[float]) -> str:
    if not scene_scores:
        return "neutral"
    average = sum(scene_scores) / len(scene_scores)
    if average >= 0.25:
        return "positive"
    if average <= -0.25:
        return "negative"
    return "neutral"


def analyze_story(story: str, max_scenes: int = 3) -> SyncPlan:
    clean_story = " ".join(story.split())
    scene_texts = split_story_into_scenes(story, max_scenes=max_scenes)
    scene_plans = []
    cursor = 0.0

    for index, scene_text in enumerate(scene_texts, start=1):
        sentiment, sentiment_score = analyze_sentiment(scene_text)
        conditions = infer_conditions(scene_text, sentiment)
        duration = estimate_scene_duration(scene_text)

        plan = ScenePlan(
            scene_number=index,
            title=build_scene_title(scene_text, index),
            text=scene_text,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            character=conditions["character"],
            action=conditions["action"],
            location=conditions["location"],
            mood=conditions["mood"],
            image_prompt=build_image_prompt(scene_text, **conditions),
            music_prompt=build_music_prompt(scene_text, conditions["mood"]),
            start_seconds=round(cursor, 2),
            duration_seconds=duration,
        )
        scene_plans.append(plan)
        cursor += duration

    return SyncPlan(
        story=clean_story,
        word_count=len(clean_story.split()),
        scene_count=len(scene_plans),
        overall_sentiment=choose_overall_sentiment([scene.sentiment_score for scene in scene_plans]),
        overall_mood=choose_overall_mood(scene_plans),
        total_duration_seconds=round(cursor, 2),
        scenes=scene_plans,
    )


def analyze_story_to_dict(story: str, max_scenes: int = 3) -> dict:
    return analyze_story(story, max_scenes=max_scenes).to_dict()

