#!/usr/bin/env python3
"""Select hybrid-rec-v1 fusion on hybrid-rec-tune-v1 (TRAIN history only).

Trains a temporary inner-train BPR-MF for leakage-safe CF channel scores.
Does not overwrite bpr-mf-v1. Does not read recsys-eval-v1 external metrics
for selection.
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
from app.db.session import get_session_factory
from app.recommendations.cf_constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_L2,
    DEFAULT_LEARNING_RATE,
)
from app.recommendations.cf_data import (
    RecsysEvalIncompatibleError,
    assert_no_zero_train_item,
    build_mappings,
    load_recsys_eval_split,
    pairs_to_index_arrays,
    user_positive_sets,
)
from app.recommendations.cf_train import train_bpr_mf
from app.recommendations.cf_tune import assert_tune_leakage_free, build_cf_tune_split
from app.recommendations.constants import EVALUATION_VERSION
from app.recommendations.hybrid_constants import DEFAULT_RRF_K, HYBRID_REC_VERSION, TUNE_PROTOCOL
from app.recommendations.hybrid_fusion import resolve_hybrid_candidate_k
from app.recommendations.hybrid_offline import (
    cf_candidates_from_scores,
    content_candidates_from_scores,
    default_weighted_grid,
    interaction_counts_excluding,
    load_catalog_embeddings,
    mean_profile,
    method_metrics,
    popularity_candidates_from_counts,
    popularity_ranking,
    summarize_rows,
    exclude_row_indices,
)
from app.recommendations.hybrid_policy import default_hybrid_policy, write_hybrid_policy
from app.recommendations.hybrid_types import HybridPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-version", default=EVALUATION_VERSION)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--cf-epochs", type=int, default=44)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--freeze-policy", action="store_true")
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


def _pick_winner(comparison: dict[str, dict[str, float]]) -> tuple[str, str | None]:
    """Select by validation NDCG@10, then R@10, then MRR. External test unused."""

    def key(name: str) -> tuple[float, float, float]:
        row = comparison[name]
        return (float(row.get("ndcg@10", 0.0)), float(row.get("recall@10", 0.0)), float(row.get("mrr", 0.0)))

    ordered = sorted(comparison, key=key, reverse=True)
    winner = ordered[0]
    if winner == "rrf":
        return "rrf", None
    if winner.startswith("weighted:"):
        return "weighted", winner.split(":", 1)[1]
    return winner, None


def main() -> int:
    args = parse_args()
    settings = get_settings()
    split_dir = Path(args.split_dir or Path(settings.artifacts_root) / "evaluation" / args.evaluation_version)
    out_dir = Path(args.out_dir or Path(settings.artifacts_root) / "evaluation" / TUNE_PROTOCOL)
    out_path = out_dir / "tuning_results.json"
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite {out_path}; pass --force", file=sys.stderr)
        return 2
    try:
        split = load_recsys_eval_split(split_dir, require_frozen_identity=True)
    except RecsysEvalIncompatibleError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    tune = build_cf_tune_split(split, protocol=TUNE_PROTOCOL)
    assert_tune_leakage_free(split, tune)
    candidate_k = resolve_hybrid_candidate_k(20, configured=args.candidate_k)
    catalog_product_ids, embeddings, id_to_row = load_catalog_embeddings(settings)
    catalog_size = int(len(catalog_product_ids))
    weighted_grid = default_weighted_grid()

    mappings = build_mappings(tune.inner_train_pairs)
    assert_no_zero_train_item(mappings, tune.inner_train_pairs)
    train_u, train_i = pairs_to_index_arrays(tune.inner_train_pairs, mappings)
    positives = user_positive_sets(tune.inner_train_pairs, mappings)
    print("loaded recsys-eval-v1; training inner CF...", flush=True)
    started = time.perf_counter()
    model, cf_summary = train_bpr_mf(
        n_users=mappings.n_users,
        n_items=mappings.n_items,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        train_user_index=train_u,
        train_item_index=train_i,
        user_positives=positives,
        product_ids=list(mappings.product_ids),
        learning_rate=DEFAULT_LEARNING_RATE,
        batch_size=DEFAULT_BATCH_SIZE,
        l2=DEFAULT_L2,
        max_epochs=args.cf_epochs,
        patience=args.cf_epochs,
        seed=42,
        device="cpu",
        fixed_epochs=args.cf_epochs,
    )
    cf_train_sec = time.perf_counter() - started
    print(f"inner CF trained in {cf_train_sec:.1f}s; scoring validation users...", flush=True)

    blocked = {(user.user_id, user.hidden_product_id) for user in split.users}
    blocked.update((row.user_id, row.hidden_product_id) for row in tune.validation_users)
    factory = get_session_factory()
    with factory() as session:
        pop_counts = interaction_counts_excluding(session, blocked)
    ranked_pop = popularity_ranking(pop_counts)

    val_users = list(tune.validation_users)
    cf_user_indices: list[int] = []
    cf_seen: list[set[str]] = []
    for row in val_users:
        cf_user_indices.append(mappings.user_to_index[row.user_id])
        cf_seen.append(set(row.inner_train_product_ids))
    from app.recommendations.hybrid_offline import score_cf_users

    cf_scores = score_cf_users(model, cf_user_indices, batch_size=args.batch_size)
    cf_product_ids = list(mappings.product_ids)
    cf_id_to_row = {product_id: index for index, product_id in enumerate(cf_product_ids)}

    profiles: list[np.ndarray] = []
    profile_index: list[int] = []
    for index, row in enumerate(val_users):
        train_rows = np.array(
            [id_to_row[pid] for pid in row.inner_train_product_ids if pid in id_to_row],
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
            exclude = set(val_users[user_i].inner_train_product_ids)
            content_by_user[user_i] = content_candidates_from_scores(
                scores[offset],
                catalog_product_ids,
                exclude=exclude,
                top_k=candidate_k,
                exclude_indices=exclude_row_indices(exclude, id_to_row),
            )

    rows_by_method: dict[str, list[dict[str, float]]] = defaultdict(list)
    coverage_sum = defaultdict(int)
    n_users = 0
    for index, row in enumerate(val_users):
        exclude = set(row.inner_train_product_ids)
        content = content_by_user.get(index)
        cf = cf_candidates_from_scores(
            cf_scores[index],
            cf_product_ids,
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
        measured = method_metrics(
            content=content,
            cf=cf,
            popularity=popularity,
            hidden_id=row.hidden_product_id,
            catalog_size=catalog_size,
            rrf_k=args.rrf_k,
            weighted_configs=weighted_grid,
        )
        n_users += 1
        for key in ("content", "cf", "popularity", "rrf"):
            rows_by_method[key].append(measured[key])
        for name, metrics in measured["weighted"].items():
            rows_by_method[f"weighted:{name}"].append(metrics)
        for flag, hit in measured["coverage"].items():
            coverage_sum[flag] += int(bool(hit))

    comparison = {name: summarize_rows(rows) for name, rows in rows_by_method.items()}
    fusion_only = {name: row for name, row in comparison.items() if name == "rrf" or name.startswith("weighted:")}
    fusion_method, weight_name = _pick_winner(fusion_only)
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "tune_protocol": TUNE_PROTOCOL,
        "hybrid_version": HYBRID_REC_VERSION,
        "evaluation_version": split.evaluation_version,
        "hidden_checksum": split.hidden_checksum,
        "validation_users": n_users,
        "inner_train_pairs": len(tune.inner_train_pairs),
        "candidate_k": candidate_k,
        "rrf_k": args.rrf_k,
        "cf_inner_epochs": args.cf_epochs,
        "cf_inner_train_seconds": round(cf_train_sec, 3),
        "cf_inner_note": (
            "Temporary inner-train BPR-MF for leakage-safe fusion selection. "
            "Not a new CF model version. Production remains bpr-mf-v1."
        ),
        "coverage": {key: value / n_users for key, value in coverage_sum.items()},
        "comparison": comparison,
        "selected_fusion_method": fusion_method,
        "selected_weight_name": weight_name,
        "git_commit": _git_commit(),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "runtime_seconds": round(elapsed, 3),
        "note": (
            "Inner validation only. External recsys-eval-v1 test was not used "
            "to select fusion method, weights, candidate_k, or RRF k."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if args.freeze_policy:
        weights = next(w for name, w in weighted_grid if name == (weight_name or "c60_cf30_p10"))
        policy = default_hybrid_policy()
        frozen = HybridPolicy(
            hybrid_version=policy.hybrid_version,
            fusion_method=fusion_method if fusion_method in {"rrf", "weighted"} else "rrf",
            rrf_k=args.rrf_k,
            weights=weights if fusion_method == "weighted" else policy.weights,
            candidate_k_min=policy.candidate_k_min,
            candidate_k_max=policy.candidate_k_max,
            candidate_k_multiplier=policy.candidate_k_multiplier,
            fallback_policy_version=policy.fallback_policy_version,
            content_rec_version=policy.content_rec_version,
            cf_model_version=policy.cf_model_version,
            popularity_rule=policy.popularity_rule,
            evaluation_version=policy.evaluation_version,
            tune_protocol=TUNE_PROTOCOL,
            created_at=payload["created_at"],
            git_commit=payload["git_commit"],
            note=(
                f"Frozen from {TUNE_PROTOCOL} selected_fusion_method={fusion_method} "
                f"weight={weight_name}. External test unused."
            ),
        )
        written = write_hybrid_policy(frozen)
        print(f"wrote {written}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
