#!/usr/bin/env python3
"""Normalize Amazon Reviews 2023 JSONL into Parquet products and interactions."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.data.prepare import prepare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="All_Beauty")
    parser.add_argument("--raw-dir", default="data/raw/amazon_reviews_2023/all_beauty")
    parser.add_argument("--processed-dir", default="data/processed/amazon_reviews_2023/all_beauty")
    parser.add_argument("--sample-dir", default="data/samples/amazon_reviews_2023/all_beauty")
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("prepare_dataset").info("Normalizing %s", args.category)
    manifest = prepare(
        category=args.category,
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        sample_dir=Path(args.sample_dir),
        sample_seed=args.sample_seed,
    )
    print(json.dumps(manifest["processed"], indent=2))
    print("rejects.products", manifest["rejects"]["products"])
    print("rejects.interactions", manifest["rejects"]["interactions"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
