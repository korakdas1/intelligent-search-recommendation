#!/usr/bin/env python3
"""Evaluate frozen hybrid-rec-v1 once on recsys-eval-v1 external test.

Does not tune. Copies existing popularity/content/CF baseline rows; does not overwrite them.
Uses production bpr-mf-v1, not the inner-train temporary CF.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from app.core.config import get_settings
from app.core.events import DATASET_VERSION_FULL
from app.db.repositories.artifacts import upsert_artifact_version
from app.db.session import get_session_factory, reset_engine
from app.ranking.metrics import macro_average
from app.recommendations.cf_artifacts import load_cf_bundle
from app.recommendations.cf_constants import CATALOG_SIZE_E008, MODEL_VERSION
from app.recommendations.cf_data import (
    RecsysEvalIncompatibleError,
    external_train_pairs,
    item_train_degrees,
    load_recsys_eval_split,
)
from app.recommendations.constants import CONTENT_REC_VERSION, EVALUATION_VERSION
from app.recommendations.evaluation import history_size_bin, rank_to_metrics
from app.recommendations.hybrid_constants import (
    DEFAULT_RRF_K,
    HYBRID_REC_VERSION,
    TUNE_PROTOCOL,
)
from app.recommendations.hybrid_eval import fuse_channel_lists, rank_hidden_in_fused
from app.recommendations.hybrid_fusion import resolve_hybrid_candidate_k
from app.recommendations.hybrid_offline import (
    cf_candidates_from_scores,
    content_candidates_from_scores,
    interaction_counts_excluding,
    load_catalog_embeddings,
    mean_profile,
    popularity_candidates_from_counts,
    popularity_ranking,
    score_cf_users,
    exclude_row_indices,
)
from app.recommendations.hybrid_policy import load_hybrid_policy, policy_to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-version", default=EVALUATION_VERSION)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--register-artifact", action="store_true")
    parser.add_argument("--examples", type=int, default=6)
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _baseline_rows(split_dir: Path, model_dir: Path) -> dict[str, Any]:
    phase9 = {}
    metrics_path = split_dir / "metrics.json"
    if metrics_path.is_file():
        phase9 = json.loads(metrics_path.read_text(encoding="utf-8"))
    phase10 = {}
    eval_path = model_dir / "evaluation.json"
    if eval_path.is_file():
        phase10 = json.loads(eval_path.read_text(encoding="utf-8"))
    return {"phase9": phase9, "phase10": phase10}


def _round_metrics(row: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in row.items()}


def main() -> int:
    args = parse_args()
    settings = get_settings()
    split_dir = Path(args.split_dir or Path(settings.artifacts_root) / "evaluation" / args.evaluation_version)
    model_dir = Path(args.model_dir or Path(settings.artifacts_root) / "models" / MODEL_VERSION)
    out_dir = Path(args.out_dir or Path(settings.artifacts_root) / "evaluation" / HYBRID_REC_VERSION)
    out_path = out_dir / "evaluation.json"
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite {out_path}; pass --force", file=sys.stderr)
        return 2
    try:
        split = load_recsys_eval_split(split_dir, require_frozen_identity=True)
    except RecsysEvalIncompatibleError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    policy = load_hybrid_policy()
    model, user_ids, product_ids, _config, manifest = load_cf_bundle(model_dir)
    if manifest.get("hidden_checksum") and manifest["hidden_checksum"] != split.hidden_checksum:
        print("CF artifact hidden checksum does not match recsys-eval-v1. Stop.", file=sys.stderr)
        return 2
    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    cf_id_to_row = {str(product_id): index for index, product_id in enumerate(product_ids)}
    candidate_k = resolve_hybrid_candidate_k(
        20,
        configured=args.candidate_k,
        minimum=policy.candidate_k_min,
        maximum=policy.candidate_k_max,
        multiplier=policy.candidate_k_multiplier,
    )
    catalog_product_ids, embeddings, id_to_row = load_catalog_embeddings(settings)
    catalog_size = split.catalog_size or int(len(catalog_product_ids)) or CATALOG_SIZE_E008
    train_pairs = external_train_pairs(split)
    degrees = item_train_degrees(train_pairs)
    blocked = {(user.user_id, user.hidden_product_id) for user in split.users}
    factory = get_session_factory()
    with factory() as session:
        pop_counts = interaction_counts_excluding(session, blocked)
    ranked_pop = popularity_ranking(pop_counts)

    started = time.perf_counter()
    cf_user_indices = [user_to_index[user.user_id] for user in split.users]
    cf_scores = score_cf_users(model, cf_user_indices, batch_size=args.batch_size)

    profiles: list[np.ndarray] = []
    profile_index: list[int] = []
    for index, user in enumerate(split.users):
        train_rows = np.array(
            [id_to_row[pid] for pid in user.train_product_ids if pid in id_to_row],
            dtype=np.int64,
        )
        profile = mean_profile(embeddings, train_rows)
        if profile is None:
            continue
        profiles.append(profile)
        profile_index.append(index)
    content_by_user: dict[int, list] = {}
    embedding_matrix = np.asarray(embeddings)
    for start in range(0, len(profiles), args.batch_size):
        chunk = np.stack(profiles[start : start + args.batch_size])
        scores = chunk @ embedding_matrix.T
        for offset, user_i in enumerate(profile_index[start : start + args.batch_size]):
            exclude = set(split.users[user_i].train_product_ids)
            content_by_user[user_i] = content_candidates_from_scores(
                scores[offset],
                catalog_product_ids,
                exclude=exclude,
                top_k=candidate_k,
                exclude_indices=exclude_row_indices(exclude, id_to_row),
            )

    hybrid_rows: list[dict[str, float]] = []
    by_history: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_degree: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_cf_item: dict[str, list[dict[str, float]]] = defaultdict(list)
    coverage_sum = defaultdict(int)
    examples: list[dict[str, Any]] = []
    for index, user in enumerate(split.users):
        exclude = set(user.train_product_ids)
        content = content_by_user.get(index)
        cf = cf_candidates_from_scores(
            cf_scores[index],
            product_ids,
            exclude=exclude,
            top_k=candidate_k,
            exclude_indices=exclude_row_indices(exclude, cf_id_to_row),
        )
        popularity = popularity_candidates_from_counts(
            pop_counts,
            exclude=exclude,
            top_k=candidate_k,
            ranked=ranked_pop,
        )
        ranked = fuse_channel_lists(
            content,
            cf,
            popularity,
            fusion_method=policy.fusion_method,
            rrf_k=policy.rrf_k,
            weights=policy.weights,
        )
        rank = rank_hidden_in_fused(user.hidden_product_id, ranked)
        metrics = rank_to_metrics(rank, catalog_size)
        hybrid_rows.append(metrics)
        hist = history_size_bin(len(user.train_product_ids))
        degree = int(degrees.get(user.hidden_product_id, 0))
        degree_bin = "0" if degree == 0 else "1" if degree == 1 else "2+"
        cf_hidden = "cf_covered_item" if user.hidden_product_id in set(product_ids) else "cf_cold_item"
        by_history[hist].append(metrics)
        by_degree[degree_bin].append(metrics)
        by_cf_item[cf_hidden].append(metrics)
        from app.recommendations.hybrid_eval import coverage_flags

        flags = coverage_flags(user.hidden_product_id, content, cf, popularity)
        for key, hit in flags.items():
            coverage_sum[key] += int(bool(hit))
        if len(examples) < args.examples:
            examples.append(
                {
                    "user_id_suffix": user.user_id[-6:],
                    "hidden_product_id": user.hidden_product_id,
                    "hybrid_rank": rank,
                    "history_bin": hist,
                    "hidden_train_degree": degree,
                    "coverage": flags,
                    "top5": [row.product_id for row in ranked[:5]],
                }
            )

    elapsed = time.perf_counter() - started
    n_users = len(hybrid_rows)
    baselines = _baseline_rows(split_dir, model_dir)
    payload: dict[str, Any] = {
        "evaluation_version": EVALUATION_VERSION,
        "hybrid_version": HYBRID_REC_VERSION,
        "label_class": "observed",
        "content_rec_version": CONTENT_REC_VERSION,
        "cf_model_version": MODEL_VERSION,
        "tune_protocol": TUNE_PROTOCOL,
        "hidden_checksum": split.hidden_checksum,
        "evaluation_users": n_users,
        "two_core_pairs": split.two_core_pairs,
        "catalog_size": catalog_size,
        "candidate_k": candidate_k,
        "candidate_policy": f"per_channel_top_{candidate_k}_union",
        "fusion_method": policy.fusion_method,
        "rrf_k": policy.rrf_k,
        "weights": policy.weights.as_dict(),
        "policy": policy_to_dict(policy),
        "coverage": {key: value / n_users for key, value in coverage_sum.items()},
        "hybrid": _round_metrics(macro_average(hybrid_rows)),
        "hybrid_by_history": {key: macro_average(rows) for key, rows in sorted(by_history.items())},
        "hybrid_by_hidden_train_degree": {key: macro_average(rows) for key, rows in sorted(by_degree.items())},
        "hybrid_by_cf_item_coverage": {key: macro_average(rows) for key, rows in sorted(by_cf_item.items())},
        "segment_counts": {
            "history": {key: len(rows) for key, rows in sorted(by_history.items())},
            "hidden_train_degree": {key: len(rows) for key, rows in sorted(by_degree.items())},
            "cf_item": {key: len(rows) for key, rows in sorted(by_cf_item.items())},
        },
        "baselines_copied": {
            "popularity": baselines["phase9"].get("popularity"),
            "content": baselines["phase9"].get("content"),
            "cf": (baselines["phase10"].get("metrics") or baselines["phase10"].get("cf")),
            "note": "Popularity/content/CF rows copied, not recomputed.",
        },
        "examples": examples,
        "git_commit": _git_commit(),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "runtime_seconds": round(elapsed, 3),
        "note": (
            "Observed hold-out. External test was not used for hybrid selection. "
            "Hybrid candidate universe is the documented per-channel union, not a "
            "restriction to CF-known items."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if args.register_artifact:
        reset_engine()
        factory = get_session_factory()
        with factory() as session:
            upsert_artifact_version(
                session,
                artifact_id=f"hybrid:{HYBRID_REC_VERSION}",
                artifact_type="hybrid_recommendation_policy",
                version=HYBRID_REC_VERSION,
                dataset_version=DATASET_VERSION_FULL,
                embedding_model_name=None,
                embedding_dim=None,
                metric="ndcg@10",
                path=str(out_dir),
                metadata={
                    "fusion_method": policy.fusion_method,
                    "rrf_k": policy.rrf_k,
                    "candidate_k": candidate_k,
                    "evaluation_version": EVALUATION_VERSION,
                    "cf_model_version": MODEL_VERSION,
                },
            )
            session.commit()
        print("registered hybrid:hybrid-rec-v1", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
