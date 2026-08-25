#!/usr/bin/env python3
"""Official observed recommendation comparison on frozen recsys-eval-v1.

Does not train or retune. Label class is observed hold-out, not preference.
Uses the same content, CF, popularity, and hybrid-rec-v1 scorers as serving/eval.
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
from app.evaluation.constants import (
    K_VALUES,
    METRIC_VERSION,
    OFFICIAL_REC_EVAL,
    OFFICIAL_COMPARISON_DIR,
    RECONCILE_ABS_TOL,
    RECOMMENDATION_EXPERIMENT_ID,
)
from app.evaluation.io import write_json_atomic
from app.evaluation.metrics import macro_average
from app.recommendations.cf_artifacts import load_cf_bundle
from app.recommendations.cf_constants import CATALOG_SIZE_E008, MODEL_VERSION
from app.recommendations.cf_data import (
    RecsysEvalIncompatibleError,
    external_train_pairs,
    item_train_degrees,
    load_recsys_eval_split,
)
from app.recommendations.cf_train import rank_hidden_from_scores
from app.recommendations.constants import CONTENT_REC_VERSION, EVALUATION_VERSION
from app.recommendations.evaluation import history_size_bin, rank_to_metrics
from app.recommendations.hybrid_constants import HYBRID_REC_VERSION
from app.recommendations.hybrid_eval import coverage_flags, fuse_channel_lists, rank_hidden_in_fused
from app.recommendations.hybrid_fusion import resolve_hybrid_candidate_k
from app.recommendations.hybrid_offline import (
    cf_candidates_from_scores,
    content_candidates_from_scores,
    exclude_row_indices,
    interaction_counts_excluding,
    load_catalog_embeddings,
    mean_profile,
    popularity_candidates_from_counts,
    popularity_ranking,
    score_cf_users,
)
from app.recommendations.hybrid_policy import load_hybrid_policy, policy_to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-version", default=EVALUATION_VERSION)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


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


def _degree_bin(degree: int) -> str:
    if degree <= 0:
        return "0"
    if degree == 1:
        return "1"
    return "2+"


def _reconcile(live: dict[str, float], prior: dict[str, float]) -> dict[str, Any]:
    keys = sorted(set(live) | set(prior))
    diffs = {}
    max_abs = 0.0
    for key in keys:
        delta = float(live.get(key, 0.0)) - float(prior.get(key, 0.0))
        diffs[key] = delta
        max_abs = max(max_abs, abs(delta))
    return {"max_abs_delta": max_abs, "within_tolerance": max_abs <= RECONCILE_ABS_TOL, "deltas": diffs}


def _segment(rows: dict[str, list[dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {key: macro_average(values) for key, values in sorted(rows.items())}


def main() -> int:
    args = parse_args()
    settings = get_settings()
    split_dir = Path(args.split_dir or Path(settings.artifacts_root) / "evaluation" / args.evaluation_version)
    model_dir = Path(args.model_dir or Path(settings.artifacts_root) / "models" / MODEL_VERSION)
    out_dir = Path(args.output) if args.output else Path(settings.artifacts_root) / "evaluation" / OFFICIAL_COMPARISON_DIR
    out_path = out_dir / "recommendation_comparison.json"
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite {out_path}; pass --force", file=sys.stderr)
        return 2
    try:
        split = load_recsys_eval_split(split_dir, require_frozen_identity=True)
    except RecsysEvalIncompatibleError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    users = list(split.users)
    sampled = False
    if args.max_users and args.max_users < len(users):
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(users))[: args.max_users]
        users = [users[int(index)] for index in sorted(order.tolist())]
        sampled = True

    policy = load_hybrid_policy()
    model, cf_user_ids, cf_item_ids, _config, manifest = load_cf_bundle(model_dir)
    if manifest.get("hidden_checksum") and manifest["hidden_checksum"] != split.hidden_checksum:
        print("CF artifact hidden checksum does not match recsys-eval-v1. Stop.", file=sys.stderr)
        return 2
    user_to_index = {user_id: index for index, user_id in enumerate(cf_user_ids)}
    cf_id_to_row = {str(product_id): index for index, product_id in enumerate(cf_item_ids)}
    candidate_k = resolve_hybrid_candidate_k(
        20,
        configured=None,
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
    pop_scores = np.array(
        [float(pop_counts.get(str(product_id), 0)) for product_id in catalog_product_ids],
        dtype=np.float32,
    )

    started = time.perf_counter()
    cf_user_indices = [user_to_index[user.user_id] for user in users]
    cf_scores = score_cf_users(model, cf_user_indices, batch_size=args.batch_size)
    cf_ids = np.asarray(cf_item_ids)

    profiles: list[np.ndarray] = []
    profile_index: list[int] = []
    for index, user in enumerate(users):
        train_rows = np.array(
            [id_to_row[pid] for pid in user.train_product_ids if pid in id_to_row],
            dtype=np.int64,
        )
        profile = mean_profile(embeddings, train_rows)
        if profile is None:
            continue
        profiles.append(profile)
        profile_index.append(index)
    content_scores_by_user: dict[int, np.ndarray] = {}
    embedding_matrix = np.asarray(embeddings)
    for start in range(0, len(profiles), args.batch_size):
        chunk = np.stack(profiles[start : start + args.batch_size])
        scores = chunk @ embedding_matrix.T
        for offset, user_i in enumerate(profile_index[start : start + args.batch_size]):
            content_scores_by_user[user_i] = scores[offset]
        done = min(start + args.batch_size, len(profiles))
        if done % 256 == 0 or done == len(profiles):
            print(f"  content scores {done}/{len(profiles)}", flush=True)

    systems = ("popularity", "content", "cf", "hybrid")
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in systems}
    by_history: dict[str, dict[str, list[dict[str, float]]]] = {
        name: defaultdict(list) for name in systems
    }
    by_degree: dict[str, dict[str, list[dict[str, float]]]] = {
        name: defaultdict(list) for name in systems
    }
    coverage_sum = defaultdict(int)
    print(f"{OFFICIAL_REC_EVAL}: scoring {len(users)} users", flush=True)
    for index, user in enumerate(users):
        exclude = set(user.train_product_ids)
        hidden_row = id_to_row.get(user.hidden_product_id)
        seen_rows = exclude_row_indices(exclude, id_to_row)
        hist = history_size_bin(len(user.train_product_ids))
        degree = int(degrees.get(user.hidden_product_id, 0))
        degree_bin = _degree_bin(degree)

        if hidden_row is None:
            pop_metrics = rank_to_metrics(None, catalog_size)
            content_metrics = rank_to_metrics(None, catalog_size)
        else:
            pop_rank = _rank_hidden(
                pop_scores, catalog_product_ids, seen_rows=seen_rows, hidden_row=int(hidden_row)
            )
            pop_metrics = rank_to_metrics(pop_rank, catalog_size)
            content_vec = content_scores_by_user.get(index)
            if content_vec is None:
                content_metrics = rank_to_metrics(None, catalog_size)
            else:
                content_rank = _rank_hidden(
                    content_vec, catalog_product_ids, seen_rows=seen_rows, hidden_row=int(hidden_row)
                )
                content_metrics = rank_to_metrics(content_rank, catalog_size)

        hidden_cf = cf_id_to_row.get(user.hidden_product_id)
        seen_cf = [cf_id_to_row[pid] for pid in user.train_product_ids if pid in cf_id_to_row]
        if hidden_cf is None:
            cf_metrics = rank_to_metrics(None, catalog_size)
        else:
            cf_rank = rank_hidden_from_scores(
                cf_scores[index],
                cf_ids,
                seen_indices=seen_cf,
                hidden_index=int(hidden_cf),
            )
            cf_metrics = rank_to_metrics(cf_rank, catalog_size)

        content_cands = None
        if index in content_scores_by_user:
            content_cands = content_candidates_from_scores(
                content_scores_by_user[index],
                catalog_product_ids,
                exclude=exclude,
                top_k=candidate_k,
                exclude_indices=seen_rows,
            )
        cf_cands = cf_candidates_from_scores(
            cf_scores[index],
            cf_item_ids,
            exclude=exclude,
            top_k=candidate_k,
            exclude_indices=exclude_row_indices(exclude, cf_id_to_row),
        )
        pop_cands = popularity_candidates_from_counts(
            pop_counts, exclude=exclude, top_k=candidate_k, ranked=ranked_pop
        )
        fused = fuse_channel_lists(
            content_cands,
            cf_cands,
            pop_cands,
            fusion_method=policy.fusion_method,
            rrf_k=policy.rrf_k,
            weights=policy.weights,
        )
        hybrid_rank = rank_hidden_in_fused(user.hidden_product_id, fused)
        hybrid_metrics = rank_to_metrics(hybrid_rank, catalog_size)
        flags = coverage_flags(user.hidden_product_id, content_cands, cf_cands, pop_cands)
        flags["cf_item_in_model"] = hidden_cf is not None
        flags["hidden_in_catalog"] = hidden_row is not None
        for key, hit in flags.items():
            coverage_sum[key] += int(bool(hit))

        bundle = {
            "popularity": pop_metrics,
            "content": content_metrics,
            "cf": cf_metrics,
            "hybrid": hybrid_metrics,
        }
        for name, metrics in bundle.items():
            rows[name].append(metrics)
            by_history[name][hist].append(metrics)
            by_degree[name][degree_bin].append(metrics)
        if (index + 1) % 2000 == 0 or index + 1 == len(users):
            print(f"  ranked {index + 1}/{len(users)}", flush=True)

    elapsed = time.perf_counter() - started
    n_users = len(users)
    headline = {name: macro_average(rows[name]) for name in systems}
    prior = {}
    phase9 = split_dir / "metrics.json"
    if phase9.is_file():
        prior.update(json.loads(phase9.read_text(encoding="utf-8")))
    phase10 = model_dir / "evaluation.json"
    if phase10.is_file():
        prior["cf_prior"] = json.loads(phase10.read_text(encoding="utf-8"))
    hybrid_prior_path = Path(settings.artifacts_root) / "evaluation" / HYBRID_REC_VERSION / "evaluation.json"
    hybrid_prior = json.loads(hybrid_prior_path.read_text()) if hybrid_prior_path.is_file() else {}
    reconciliation = {}
    if prior.get("popularity"):
        reconciliation["popularity"] = _reconcile(headline["popularity"], prior["popularity"])
    if prior.get("content"):
        reconciliation["content"] = _reconcile(headline["content"], prior["content"])
    cf_prior_metrics = (prior.get("cf_prior") or {}).get("cf")
    if isinstance(cf_prior_metrics, dict) and "recall@10" in cf_prior_metrics:
        reconciliation["cf"] = _reconcile(headline["cf"], cf_prior_metrics)
    if hybrid_prior.get("hybrid"):
        reconciliation["hybrid"] = _reconcile(headline["hybrid"], hybrid_prior["hybrid"])

    payload = {
        "experiment_id": RECOMMENDATION_EXPERIMENT_ID,
        "evaluation_version": OFFICIAL_REC_EVAL,
        "source_split": EVALUATION_VERSION,
        "label_class": "observed",
        "label_note": (
            "Held-out future interaction. Not explicit preference, rating, CTR, or conversion."
        ),
        "metric_version": METRIC_VERSION,
        "k_values": list(K_VALUES),
        "hidden_checksum": split.hidden_checksum,
        "evaluation_users": n_users,
        "two_core_pairs": split.two_core_pairs,
        "catalog_size": catalog_size,
        "sampled": sampled,
        "max_users": args.max_users,
        "seed": args.seed,
        "content_rec_version": CONTENT_REC_VERSION,
        "cf_model_version": MODEL_VERSION,
        "hybrid_version": HYBRID_REC_VERSION,
        "candidate_policies": {
            "popularity": "full_catalog_minus_train_seen",
            "content": "full_catalog_minus_train_seen",
            "cf": "bpr-mf-v1_trained_items_only",
            "hybrid": f"per_channel_top_{candidate_k}_union",
        },
        "hybrid_candidate_k": candidate_k,
        "hybrid_policy": policy_to_dict(policy),
        "all_users": headline,
        "coverage": {key: value / n_users for key, value in coverage_sum.items()},
        "cf_catalog_coverage": len(cf_item_ids) / catalog_size,
        "cf_test_item_coverage": coverage_sum["cf_item_in_model"] / n_users,
        "hybrid_union_target_coverage": coverage_sum["union_candidate"] / n_users,
        "by_history": {name: _segment(by_history[name]) for name in systems},
        "by_hidden_train_degree": {name: _segment(by_degree[name]) for name in systems},
        "segment_counts": {
            "history": {key: len(values) for key, values in sorted(by_history["content"].items())},
            "hidden_train_degree": {
                key: len(values) for key, values in sorted(by_degree["content"].items())
            },
        },
        "reconciliation": reconciliation,
        "git_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": elapsed,
        "not_preference": True,
        "one_positive_precision_note": (
            "One hidden product per user. Recall@K is hit-rate@K. Precision@K is 1/K on a hit."
        ),
    }
    write_json_atomic(out_path, payload)
    print(json.dumps(
        {
            "path": str(out_path),
            "users": n_users,
            "all_users": headline,
            "hybrid_union_target_coverage": payload["hybrid_union_target_coverage"],
            "reconciliation": {key: value["max_abs_delta"] for key, value in reconciliation.items()},
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
