import pandas as pd

from src.config import (
    ACTION_CLASSES,
    CHARACTER_CLASSES,
    LOCATION_CLASSES,
    MOOD_CLASSES,
    SCENE_LABELS_PATH,
)


def validate_column(values, valid_set, column_name):
    invalid = sorted(set(values) - set(valid_set))
    if invalid:
        raise ValueError(f"Invalid values in '{column_name}': {invalid}")


def main():
    df = pd.read_csv(SCENE_LABELS_PATH, encoding="latin1")

    required_columns = ["filename", "character", "action", "location", "mood"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    validate_column(df["character"].dropna().tolist(), CHARACTER_CLASSES, "character")
    validate_column(df["action"].dropna().tolist(), ACTION_CLASSES, "action")
    validate_column(df["location"].dropna().tolist(), LOCATION_CLASSES, "location")
    validate_column(df["mood"].dropna().tolist(), MOOD_CLASSES, "mood")

    print("Labels are valid.")
    print(f"Total labeled rows: {len(df)}")


if __name__ == "__main__":
    main()