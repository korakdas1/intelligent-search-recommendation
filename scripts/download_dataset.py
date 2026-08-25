"""Download a single Amazon Reviews 2023 category from Hugging Face.

Does not load the Hugging Face ``datasets`` scripts API.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
META_TEMPLATE = "raw/meta_categories/meta_{category}.jsonl"
REVIEW_TEMPLATE = "raw/review_categories/{category}.jsonl"

logger = logging.getLogger("download_dataset")


def _copy_if_needed(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        logger.info("Already present: %s (%s bytes)", dest, dest.stat().st_size)
        return
    shutil.copy2(src, dest)
    logger.info("Wrote %s (%s bytes)", dest, dest.stat().st_size)


def download_category(
    category: str,
    dest_dir: Path,
    cache_dir: Path,
) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "metadata": META_TEMPLATE.format(category=category),
        "reviews": REVIEW_TEMPLATE.format(category=category),
    }
    local: dict[str, Path] = {}
    for kind, filename in files.items():
        logger.info("Downloading %s (%s)", kind, filename)
        cached = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        dest = dest_dir / Path(filename).name
        _copy_if_needed(Path(cached), dest)
        local[kind] = dest
    return local


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download one Amazon Reviews 2023 category (reviews + metadata)."
    )
    parser.add_argument("--category", default="All_Beauty")
    parser.add_argument(
        "--dest-dir",
        default="data/raw/amazon_reviews_2023/all_beauty",
        help="Directory for copied JSONL files",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/.hf_cache",
        help="Hugging Face hub cache (gitignored)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = download_category(args.category, Path(args.dest_dir), Path(args.cache_dir))
    for kind, path in paths.items():
        print(f"{kind}: {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
