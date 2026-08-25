#!/usr/bin/env python3
"""Build recsys-eval-v1: 2-core leave-last-product-out split.

Does not download Amazon data. Uses the persisted PostgreSQL catalog.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.recommendations.constants import EVALUATION_VERSION
from app.recommendations.evaluation import build_eval_users


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recsys-eval-v1 split artifacts")
    parser.add_argument("--output-dir", default=None, help="Default: artifacts/evaluation/recsys-eval-v1")
    parser.add_argument("--no-two-core", action="store_true", help="Use all users with ≥2 distinct products")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    output = Path(args.output_dir or Path(settings.artifacts_root) / "evaluation" / EVALUATION_VERSION)
    split_path = output / "split.parquet"
    if split_path.exists() and not args.force:
        print(f"refusing to overwrite {split_path}; pass --force")
        return 2
    output.mkdir(parents=True, exist_ok=True)
    factory = get_session_factory()
    with factory() as session:
        eval_users, stats = build_eval_users(session, use_two_core=not args.no_two_core)
    rows = []
    for user in eval_users:
        for product_id in user.train_product_ids:
            rows.append(
                {
                    "user_id": user.user_id,
                    "product_id": product_id,
                    "split": "train",
                    "hidden_product_id": user.hidden_product_id,
                    "hidden_occurred_at": user.hidden_occurred_at,
                }
            )
        rows.append(
            {
                "user_id": user.user_id,
                "product_id": user.hidden_product_id,
                "split": "test",
                "hidden_product_id": user.hidden_product_id,
                "hidden_occurred_at": user.hidden_occurred_at,
            }
        )
    frame = pd.DataFrame(rows)
    tmp = output / ".split.parquet.tmp"
    frame.to_parquet(tmp, index=False)
    tmp.replace(split_path)
    manifest = {
        "evaluation_version": EVALUATION_VERSION,
        "label_class": "observed",
        "population": "iterative_2_core" if not args.no_two_core else "ge2_distinct_products",
        "split": "leave_last_distinct_product_by_last_occurred_at",
        "canonical_product_id": "parent_asin / products.product_id",
        "seen_filter": "train_products_only",
        "seed": args.seed,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "train_interaction_id_count": None,
        **stats,
        "split_rows": int(len(frame)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
