#!/usr/bin/env python3
"""Profile normalized products/interactions and write JSON stats."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from app.data.profile_report import build_profile, decide_cf_viability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed/amazon_reviews_2023/all_beauty")
    parser.add_argument(
        "--output",
        default="data/processed/amazon_reviews_2023/all_beauty/profile.json",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    processed = Path(args.processed_dir)
    products = pd.read_parquet(processed / "products.parquet")
    interactions = pd.read_parquet(processed / "interactions.parquet")
    profile = build_profile(products, interactions)
    profile["d020"] = decide_cf_viability(profile)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2, default=str) + "\n", encoding="utf-8")
    public = Path("docs/dataset_profile_stats.json")
    public.write_text(json.dumps(profile, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(profile["d020"], indent=2))
    print("users", profile["interactions"]["users"])
    print("items", profile["interactions"]["items"])
    print("rows", profile["interactions"]["interaction_rows"])
    print("pairs", profile["interactions"]["unique_user_item_pairs"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
