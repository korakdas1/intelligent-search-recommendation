#!/usr/bin/env python3
"""Observed held-out evaluation: popularity vs user content-profile.

Full-catalog candidate policy. Train-only popularity and profiles.
Does not train a model. Does not encode text.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.embeddings.normalize import l2_normalize
from app.embeddings.versioning import embedding_paths
from app.models.interaction import Interaction
from app.ranking.metrics import macro_average
from app.recommendations.constants import CONTENT_REC_VERSION, EVALUATION_VERSION
from app.recommendations.evaluation import history_size_bin, rank_to_metrics
from app.search.runtime import require_semantic_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate content-rec-v1 vs popularity")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-version", default=EVALUATION_VERSION)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_embeddings(settings) -> tuple[np.ndarray, np.ndarray]:
    paths = embedding_paths(Path(settings.artifacts_root), settings.semantic_artifact_version)
    product_ids = np.load(paths["product_ids"], allow_pickle=False)
    if paths["embeddings"].is_file():
        embeddings = np.load(paths["embeddings"], mmap_mode="r")
        return product_ids, np.asarray(embeddings, dtype=np.float32)
    runtime = require_semantic_runtime()
    matrix = np.vstack(
        [np.asarray(runtime.index.reconstruct(i), dtype=np.float32) for i in range(runtime.ntotal)]
    )
    return runtime.product_ids, matrix


def _rank_hidden(
    scores: np.ndarray,
    product_ids: np.ndarray,
    *,
    seen_rows: np.ndarray,
    hidden_row: int,
) -> int:
    hidden_score = float(scores[hidden_row])
    hidden_id = str(product_ids[hidden_row])
    eligible = np.ones(scores.shape[0], dtype=bool)
    if seen_rows.size:
        eligible[seen_rows] = False
    eligible[hidden_row] = True
    better = (eligible & (scores > hidden_score)) | (
        eligible & (scores == hidden_score) & (product_ids < hidden_id)
    )
    return int(np.count_nonzero(better)) + 1


def main() -> int:
    args = parse_args()
    settings = get_settings()
    split_dir = Path(args.split_dir or Path(settings.artifacts_root) / "evaluation" / args.evaluation_version)
    metrics_path = split_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"refusing to overwrite {metrics_path}; pass --force")
        return 2
    split_path = split_dir / "split.parquet"
    if not split_path.is_file():
        print(f"missing split {split_path}; run scripts/build_recommendation_eval_split.py")
        return 2

    started = time.perf_counter()
    split = pd.read_parquet(split_path)
    eval_ids = split["user_id"].drop_duplicates().tolist()
    if args.max_users is not None:
        rng = np.random.default_rng(args.seed)
        eval_ids = list(rng.choice(eval_ids, size=min(args.max_users, len(eval_ids)), replace=False))
        eval_ids = [str(item) for item in eval_ids]
        split = split[split["user_id"].isin(eval_ids)]

    train = split[split["split"] == "train"]
    test = split[split["split"] == "test"]
    train_by_user = train.groupby("user_id")["product_id"].apply(list).to_dict()
    hidden_by_user = dict(zip(test["user_id"], test["product_id"], strict=False))

    product_ids, embeddings = _load_embeddings(settings)
    id_to_row = {str(product_id): index for index, product_id in enumerate(product_ids.tolist())}
    catalog_size = int(len(product_ids))

    factory = get_session_factory()
    with factory() as session:
        rows = session.execute(select(Interaction.user_id, Interaction.product_id)).all()
    hidden_pairs = set(zip(test["user_id"].astype(str), test["product_id"].astype(str), strict=False))
    train_counts: dict[str, int] = defaultdict(int)
    train_interaction_count = 0
    for user_id, product_id in rows:
        pair = (str(user_id), str(product_id))
        if pair in hidden_pairs:
            continue
        train_counts[str(product_id)] += 1
        train_interaction_count += 1

    pop_scores = np.array(
        [float(train_counts.get(str(product_id), 0)) for product_id in product_ids],
        dtype=np.float32,
    )

    skipped = {"hidden_not_in_index": 0, "no_train_embedding": 0}
    pop_rows: list[dict[str, float]] = []
    content_rows: list[dict[str, float]] = []
    pop_segments: dict[str, list[dict[str, float]]] = defaultdict(list)
    content_segments: dict[str, list[dict[str, float]]] = defaultdict(list)

    users = [user_id for user_id in eval_ids if user_id in hidden_by_user]
    profiles: list[np.ndarray] = []
    meta: list[tuple[str, np.ndarray, int, str]] = []
    for user_id in users:
        hidden_id = str(hidden_by_user[user_id])
        hidden_row = id_to_row.get(hidden_id)
        if hidden_row is None:
            skipped["hidden_not_in_index"] += 1
            continue
        train_ids = [str(item) for item in train_by_user.get(user_id, [])]
        train_rows = np.array(
            [id_to_row[item] for item in train_ids if item in id_to_row],
            dtype=np.int64,
        )
        if train_rows.size == 0:
            skipped["no_train_embedding"] += 1
            continue
        profile = l2_normalize(np.mean(embeddings[train_rows], axis=0).astype(np.float32))
        profiles.append(profile)
        meta.append((user_id, train_rows, int(hidden_row), history_size_bin(int(train_rows.size))))

    batch = max(1, args.batch_size)
    for start in range(0, len(profiles), batch):
        chunk = np.stack(profiles[start : start + batch])
        content_scores = chunk @ np.asarray(embeddings).T
        for offset, (user_id, train_rows, hidden_row, segment) in enumerate(meta[start : start + batch]):
            c_rank = _rank_hidden(
                content_scores[offset],
                product_ids,
                seen_rows=train_rows,
                hidden_row=hidden_row,
            )
            p_rank = _rank_hidden(
                pop_scores,
                product_ids,
                seen_rows=train_rows,
                hidden_row=hidden_row,
            )
            c_metrics = rank_to_metrics(c_rank, catalog_size)
            p_metrics = rank_to_metrics(p_rank, catalog_size)
            content_rows.append(c_metrics)
            pop_rows.append(p_metrics)
            content_segments[segment].append(c_metrics)
            pop_segments[segment].append(p_metrics)

    elapsed = time.perf_counter() - started
    payload = {
        "evaluation_version": args.evaluation_version,
        "label_class": "observed",
        "content_rec_version": CONTENT_REC_VERSION,
        "candidate_policy": "full_catalog_minus_train_seen",
        "k_values": [5, 10, 20],
        "seed": args.seed,
        "git_commit": _git_commit(),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "device": "cpu",
        "evaluation_users_in_split": int(split["user_id"].nunique()) if args.max_users is None else len(eval_ids),
        "scored_users": len(content_rows),
        "skipped": skipped,
        "train_interaction_count": train_interaction_count,
        "held_out_pairs": int(len(hidden_pairs)),
        "catalog_size": catalog_size,
        "runtime_seconds": round(elapsed, 3),
        "popularity": macro_average(pop_rows),
        "content": macro_average(content_rows),
        "popularity_by_history": {key: macro_average(rows) for key, rows in sorted(pop_segments.items())},
        "content_by_history": {key: macro_average(rows) for key, rows in sorted(content_segments.items())},
        "segment_counts": {key: len(rows) for key, rows in sorted(content_segments.items())},
        "note": (
            "Held-out observed interaction evaluation. Not explicit preference. "
            "Recall@K with one hidden item is hit-rate@K. Metrics are on the "
            "controlled 2-core evaluation population unless the split says otherwise."
        ),
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
