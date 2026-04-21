from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Any

import requests

# ---------- Config ----------
TARGET_COUNT = 50
OUTPUT_DIR = Path("data/raw/scene_subset_50")
LABELS_CSV = Path("data/processed/labels/scene_subset_50_labels.csv")
METADATA_CSV = Path("data/processed/labels/scene_subset_50_metadata.csv")

# Keep the tag query simple for better unauthenticated compatibility.
# You can try: "rating:safe scenery", "rating:safe outdoors", "rating:safe indoors"
BASE_TAGS = "rating:safe scenery"

# Danbooru API endpoint
POSTS_URL = "https://danbooru.donmai.us/posts.json"

# Be polite to the API
SLEEP_BETWEEN_REQUESTS_SEC = 1.0
TIMEOUT_SEC = 30

HEADERS = {
    "User-Agent": "DreamVisionDatasetBuilder/1.0 (research project; contact: local-script)"
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)


def fetch_posts(tags: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Fetch a random batch of Danbooru posts.
    """
    params = {
        "tags": tags,
        "limit": limit,
        "random": "true",
    }

    response = requests.get(
        POSTS_URL,
        params=params,
        headers=HEADERS,
        timeout=TIMEOUT_SEC,
    )
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list):
        raise ValueError("Unexpected API response format.")

    return data


def choose_image_url(post: dict[str, Any]) -> str | None:
    """
    Prefer the original file URL; fall back to large/preview if needed.
    """
    return (
        post.get("file_url")
        or post.get("large_file_url")
        or post.get("preview_file_url")
    )


def safe_extension_from_url(url: str) -> str:
    lower = url.lower()
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        if ext in lower:
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def download_file(url: str, destination: Path) -> None:
    with requests.get(url, headers=HEADERS, timeout=TIMEOUT_SEC, stream=True) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def build_search_text(post: dict[str, Any]) -> str:
    tag_string = str(post.get("tag_string", ""))
    rating = str(post.get("rating", ""))
    return f"{tag_string} rating:{rating}".strip()


def is_scene_leaning(post: dict[str, Any]) -> bool:
    """
    Lightweight filter to reduce close-up portrait bias.
    This is intentionally simple and conservative.
    """
    text = build_search_text(post).lower()

    positive_terms = [
        "scenery", "outdoors", "indoors", "forest", "park", "street",
        "road", "store", "shop", "room", "bedroom", "kitchen", "home",
        "sky", "cloud", "building", "full_body", "walking", "shopping",
        "playing", "sleeping"
    ]
    negative_terms = [
        "close-up", "close_up", "portrait", "headshot", "comic", "speech_bubble",
        "text", "monochrome"
    ]

    pos_hits = sum(term in text for term in positive_terms)
    neg_hit = any(term in text for term in negative_terms)

    return pos_hits >= 1 and not neg_hit


def main() -> None:
    ensure_dirs()

    downloaded_ids: set[int] = set()
    label_rows: list[dict[str, str]] = []
    metadata_rows: list[dict[str, str]] = []

    attempts = 0
    max_attempts = 40

    print(f"Downloading up to {TARGET_COUNT} Danbooru images...")
    print(f"Query: {BASE_TAGS}")

    while len(label_rows) < TARGET_COUNT and attempts < max_attempts:
        attempts += 1

        try:
            posts = fetch_posts(BASE_TAGS, limit=20)
        except Exception as e:
            print(f"[Attempt {attempts}] API error: {e}")
            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
            continue

        random.shuffle(posts)

        for post in posts:
            post_id = post.get("id")
            if not isinstance(post_id, int):
                continue

            if post_id in downloaded_ids:
                continue

            url = choose_image_url(post)
            if not url:
                continue

            if not is_scene_leaning(post):
                continue

            ext = safe_extension_from_url(url)
            filename = f"{post_id}{ext}"
            destination = OUTPUT_DIR / filename

            try:
                download_file(url, destination)
            except Exception as e:
                print(f"  Skipped post {post_id}: download failed ({e})")
                continue

            downloaded_ids.add(post_id)

            label_rows.append(
                {
                    "filename": filename,
                    "character": "none",
                    "action": "none",
                    "location": "none",
                    "mood": "none",
                }
            )

            metadata_rows.append(
                {
                    "id": str(post_id),
                    "filename": filename,
                    "rating": str(post.get("rating", "")),
                    "tag_string": str(post.get("tag_string", "")),
                    "file_url": url,
                }
            )

            print(f"  Downloaded {len(label_rows)}/{TARGET_COUNT}: {filename}")

            if len(label_rows) >= TARGET_COUNT:
                break

        time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "character", "action", "location", "mood"],
        )
        writer.writeheader()
        writer.writerows(label_rows)

    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "filename", "rating", "tag_string", "file_url"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print("\nDone.")
    print(f"Images downloaded: {len(label_rows)}")
    print(f"Images folder: {OUTPUT_DIR}")
    print(f"Label CSV: {LABELS_CSV}")
    print(f"Metadata CSV: {METADATA_CSV}")

    if len(label_rows) < TARGET_COUNT:
        print(
            "Note: fewer than 50 images were downloaded. "
            "Try loosening BASE_TAGS or increasing max_attempts."
        )


if __name__ == "__main__":
    main()