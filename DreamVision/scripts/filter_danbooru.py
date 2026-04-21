import re
from pathlib import Path

import pandas as pd

from src.config import (
    DANBOORU_METADATA_DIR,
    FILTERED_METADATA_DIR,
    EXCLUDE_TAG_KEYWORDS,
    INCLUDE_TAG_KEYWORDS,
    MIN_TAG_MATCHES,
)
from src.utils import ensure_project_dirs


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def count_keyword_matches(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if normalize_text(kw) in text)


def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    return any(normalize_text(kw) in text for kw in keywords)


def build_search_text(row: pd.Series) -> str:
    fields = [
        row.get("tag_string", ""),
        row.get("general_tags", ""),
        row.get("caption", ""),
        row.get("description", ""),
    ]
    return normalize_text(" ".join(str(x) for x in fields))


def filter_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["search_text"] = df.apply(build_search_text, axis=1)
    df["include_matches"] = df["search_text"].apply(
        lambda x: count_keyword_matches(x, INCLUDE_TAG_KEYWORDS)
    )
    df["has_excluded"] = df["search_text"].apply(
        lambda x: contains_any_keyword(x, EXCLUDE_TAG_KEYWORDS)
    )

    filtered = df[
        (df["include_matches"] >= MIN_TAG_MATCHES) &
        (~df["has_excluded"])
    ].copy()

    return filtered.sort_values(by="include_matches", ascending=False)


def main() -> None:
    ensure_project_dirs([FILTERED_METADATA_DIR])

    input_csv = DANBOORU_METADATA_DIR / "metadata.csv"
    output_csv = FILTERED_METADATA_DIR / "filtered_scene_metadata.csv"

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Expected metadata file at: {input_csv}\n"
            "Place your Danbooru metadata CSV there first."
        )

    df = pd.read_csv(input_csv)
    filtered_df = filter_metadata(df)

    filtered_df.to_csv(output_csv, index=False)

    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered_df)}")
    print(f"Saved filtered metadata to: {output_csv}")


if __name__ == "__main__":
    main()