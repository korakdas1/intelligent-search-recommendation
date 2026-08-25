#!/usr/bin/env python3
"""Evaluate bpr-mf-v1 once on the frozen recsys-eval-v1 external test.

Does not tune. Does not overwrite existing popularity/content metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from app.core.config import get_settings
from app.ranking.metrics import macro_average
from app.recommendations.cf_artifacts import load_cf_bundle
from app.recommendations.cf_constants import CATALOG_SIZE_E008, MODEL_VERSION
from app.recommendations.cf_data import (
    RecsysEvalIncompatibleError,
    external_train_pairs,
    item_train_degrees,
    load_recsys_eval_split,
)
from app.recommendations.cf_train import rank_hidden_from_scores
from app.recommendations.constants import EVALUATION_VERSION
from app.recommendations.evaluation import history_size_bin, rank_to_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-version", default=EVALUATION_VERSION)
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--examples", type=int, default=6)
    return parser.parse_args()


def _phase9_metrics(split_dir: Path) -> dict[str, Any]:
    path = split_dir / "metrics.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    settings = get_settings()
    split_dir = Path(args.split_dir or Path(settings.artifacts_root) / "evaluation" / args.evaluation_version)
    model_dir = Path(args.model_dir or Path(settings.artifacts_root) / "models" / args.model_version)
    out_path = model_dir / "evaluation.json"
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite {out_path}; pass --force", file=sys.stderr)
        return 2
    try:
        split = load_recsys_eval_split(split_dir, require_frozen_identity=True)
    except RecsysEvalIncompatibleError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    model, user_ids, product_ids, config, manifest = load_cf_bundle(model_dir)
    if manifest.get("hidden_checksum") and manifest["hidden_checksum"] != split.hidden_checksum:
        print("CF artifact hidden checksum does not match recsys-eval-v1. Stop.", file=sys.stderr)
        return 2
    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    product_to_index = {product_id: index for index, product_id in enumerate(product_ids)}
    train_pairs = external_train_pairs(split)
    degrees = item_train_degrees(train_pairs)
    catalog_size = split.catalog_size or CATALOG_SIZE_E008

    user_indices: list[int] = []
    hidden_indices: list[int | None] = []
    seen_indices: list[list[int]] = []
    meta: list[dict[str, Any]] = []
    missing_users = 0
    for user in split.users:
        user_index = user_to_index.get(user.user_id)
        if user_index is None:
            missing_users += 1
            continue
        hidden = product_to_index.get(user.hidden_product_id)
        seen = [product_to_index[pid] for pid in user.train_product_ids if pid in product_to_index]
        user_indices.append(user_index)
        hidden_indices.append(hidden)
        seen_indices.append(seen)
        meta.append(
            {
                "user_id": user.user_id,
                "history_bin": history_size_bin(len(user.train_product_ids)),
                "item_degree": int(degrees.get(user.hidden_product_id, 0)),
                "hidden_product_id": user.hidden_product_id,
                "train_product_ids": user.train_product_ids,
                "hidden_covered": hidden is not None,
            }
        )

    started = time.perf_counter()
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    ids = np.asarray(product_ids)
    all_rows: list[dict[str, float]] = []
    covered_rows: list[dict[str, float]] = []
    by_history: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_degree: dict[str, list[dict[str, float]]] = defaultdict(list)
    rec_counter: Counter[str] = Counter()
    example_candidates: list[dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, len(user_indices), args.batch_size):
            stop = start + args.batch_size
            batch_users = torch.tensor(user_indices[start:stop], dtype=torch.long, device=device)
            scores = model.score_items(batch_users).detach().cpu().numpy()
            for offset, info in enumerate(meta[start:stop]):
                hidden = hidden_indices[start + offset]
                seen = seen_indices[start + offset]
                if hidden is None:
                    metrics = rank_to_metrics(None, catalog_size)
                else:
                    rank = rank_hidden_from_scores(
                        scores[offset],
                        ids,
                        seen_indices=seen,
                        hidden_index=int(hidden),
                    )
                    metrics = rank_to_metrics(rank, catalog_size)
                all_rows.append(metrics)
                by_history[info["history_bin"]].append(metrics)
                degree_bin = "0" if info["item_degree"] == 0 else ("1" if info["item_degree"] == 1 else "2+")
                by_degree[degree_bin].append(metrics)
                if info["hidden_covered"]:
                    covered_rows.append(metrics)
                working = np.array(scores[offset], copy=True)
                if seen:
                    working[np.asarray(seen, dtype=np.int64)] = -np.inf
                top = np.lexsort((ids, -working))[:10]
                for item_index in top:
                    rec_counter[str(ids[item_index])] += 1
                if len(example_candidates) < 200:
                    example_candidates.append(
                        {
                            **info,
                            "metrics": metrics,
                            "top_ids": [str(ids[item_index]) for item_index in top],
                            "hidden_in_top10": info["hidden_product_id"] in {str(ids[i]) for i in top},
                        }
                    )

    elapsed = time.perf_counter() - started
    phase9 = _phase9_metrics(split_dir)
    hits = [row for row in example_candidates if row["hidden_in_top10"]]
    misses = [row for row in example_candidates if not row["hidden_in_top10"]]
    examples = (hits[:3] + misses[:3])[: args.examples]
    frequent = []
    for product_id, count in rec_counter.most_common(15):
        frequent.append(
            {
                "product_id": product_id,
                "recommended_as_top10": int(count),
                "train_unique_users": int(degrees.get(product_id, 0)),
            }
        )
    test_item_covered = sum(1 for row in meta if row["hidden_covered"])
    payload = {
        "evaluation_version": EVALUATION_VERSION,
        "model_version": args.model_version,
        "label_class": "observed",
        "candidate_policy": "full_catalog_minus_train_seen_untrained_items_unscored",
        "hidden_checksum": split.hidden_checksum,
        "evaluation_users": split.evaluation_users,
        "scored_users": len(all_rows),
        "users_missing_from_model": missing_users,
        "user_coverage": (len(user_indices) / split.evaluation_users) if split.evaluation_users else None,
        "train_known_items": len(product_ids),
        "catalog_size": catalog_size,
        "catalog_coverage": len(product_ids) / float(catalog_size),
        "held_out_pairs": split.evaluation_users,
        "hidden_test_items_with_cf_embedding": test_item_covered,
        "test_item_coverage": test_item_covered / float(len(meta)) if meta else None,
        "popularity": phase9.get("popularity"),
        "content": phase9.get("content"),
        "cf": macro_average(all_rows),
        "cf_test_item_covered": macro_average(covered_rows),
        "cf_by_history": {key: macro_average(rows) for key, rows in sorted(by_history.items())},
        "popularity_by_history": phase9.get("popularity_by_history"),
        "content_by_history": phase9.get("content_by_history"),
        "cf_by_item_train_degree": {key: macro_average(rows) for key, rows in sorted(by_degree.items())},
        "segment_counts": {
            "history": {key: len(rows) for key, rows in sorted(by_history.items())},
            "item_degree": {key: len(rows) for key, rows in sorted(by_degree.items())},
            "phase9": phase9.get("segment_counts"),
        },
        "most_frequent_cf_top10": frequent,
        "examples": [
            {
                "user_id_suffix": row["user_id"][-6:],
                "train_product_ids": list(row["train_product_ids"][:5]),
                "hidden_product_id": row["hidden_product_id"],
                "hidden_in_top10": row["hidden_in_top10"],
                "top_ids": row["top_ids"],
                "recall@10": row["metrics"]["recall@10"],
            }
            for row in examples
        ],
        "runtime_seconds": round(elapsed, 3),
        "device": "cpu",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "note": (
            "Observed hold-out, not explicit preference. Primary CF row includes cold "
            "hidden items as misses. Popularity/content copied from recsys-eval-v1 "
            "metrics.json and were not recomputed."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("cf", "cf_test_item_covered", "test_item_coverage", "catalog_coverage", "runtime_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
