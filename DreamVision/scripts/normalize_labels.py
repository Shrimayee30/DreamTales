import pandas as pd

from src.config import SCENE_LABELS_PATH

CHARACTER_MAP = {
    "2 friends": "friends",
    "sisters": "friends",
    "man and woman": "friends",
    "villagers": "friends",
    "soldiers": "friends",
    "3 soldiers": "friends",
    "girl and bird": "friends",
    "girl and owl": "friends",
    "man and dog": "animal_pair",
    "2 fish": "animal_pair",
    "boy": "none",
    "girl": "none",
    "angel": "none",
    "angels": "none",
    "girl ninja": "none",
    "solder": "none",
}

ACTION_MAP = {
    "cleaning": "none",
    "climbing stairs": "walking",
    "eating": "none",
    "experimenting": "none",
    "fighting": "none",
    "fishing": "none",
    "flying": "none",
    "looking": "none",
    "playing music": "playing",
    "playing with eggs": "playing",
    "posing": "none",
    "reading": "none",
    "running": "walking",
    "serving": "shopping",
    "serving_coffee": "shopping",
    "sitting": "none",
    "sketching": "none",
    "skydiving": "none",
    "standing": "none",
    "swimming": "none",
    "umbrella": "none",
    "waiving": "none",
    "watching": "none",
}


def main():
    df = pd.read_csv(SCENE_LABELS_PATH, encoding="latin1")

    df["character"] = df["character"].replace(CHARACTER_MAP)
    df["action"] = df["action"].replace(ACTION_MAP)

    df.to_csv(SCENE_LABELS_PATH, index=False, encoding="utf-8")
    print("Character and action labels normalized. CSV resaved as UTF-8.")


if __name__ == "__main__":
    main()