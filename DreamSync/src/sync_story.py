from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import analyze_story


DEFAULT_STORY = (
    "A glowing cloud drifts above a sleepy town and gathers the kindest thoughts from the night. "
    "It follows two best friends through a quiet park where lanterns sway in the breeze. "
    "At the end of the dream, silver stars settle over the trees and the whole sky feels calm."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Break a DreamTales story into synced scene plans.")
    parser.add_argument("--story", type=str, default=DEFAULT_STORY)
    parser.add_argument("--max-scenes", type=int, default=3)
    parser.add_argument("--output", type=str, default="DreamSync/outputs/plans/latest_sync_plan.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = analyze_story(args.story, max_scenes=args.max_scenes)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan.to_dict(), indent=2))

    print(f"Scenes: {plan.scene_count}")
    print(f"Overall sentiment: {plan.overall_sentiment}")
    print(f"Overall mood: {plan.overall_mood}")
    print(f"Saved sync plan to: {output_path}")


if __name__ == "__main__":
    main()

