#!/usr/bin/env python3
"""Select personalized-search-v1 on TRAIN-only inner validation.

Temporary inner-train BPR-MF is evaluation infrastructure only.
Does not overwrite bpr-mf-v1. External recsys-eval-v1 test is unused.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.models.product import Product
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
from app.recommendations.hybrid_offline import load_catalog_embeddings
from app.search.personalization_constants import (
    PERSONALIZATION_VERSION,
    TUNE_PROTOCOL,
)
from app.search.personalization_eval import (
    attach_history_bin,
    cf_scores_from_model,
    content_scores_from_matrix,
    evaluate_ranking_pair,
    gamma_grid,
    hybrid_baseline_candidates,
    prepare_query,
    rerank_with_config,
    select_inner_winner,
    signal_grid,
    summarize_eval_rows,
)
from app.search.personalization_policy import (
    load_personalization_policy,
    policy_from_dict,
    policy_to_dict,
    write_personalization_policy,
)
from sqlalchemy import select


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-version", default=EVALUATION_VERSION)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cf-epochs", type=int, default=44)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--freeze-policy", action="store_true")
    parser.add_argument("--fusion-method", default="rrf")
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


def _load_titles(session) -> dict[str, str]:
    return {
        str(product_id): str(title or "")
        for product_id, title in session.execute(select(Product.product_id, Product.title))
    }


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
    users = list(tune.validation_users)
    if args.max_users and args.max_users < len(users):
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(users))[: args.max_users]
        users = [users[int(index)] for index in sorted(order.tolist())]

    mappings = build_mappings(tune.inner_train_pairs)
    assert_no_zero_train_item(mappings, tune.inner_train_pairs)
    train_u, train_i = pairs_to_index_arrays(tune.inner_train_pairs, mappings)
    positives = user_positive_sets(tune.inner_train_pairs, mappings)
    print(f"training temporary inner CF on {len(tune.inner_train_pairs)} pairs...", flush=True)
    model, _summary = train_bpr_mf(
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
    user_factors = model.user_embedding.weight.detach().cpu().numpy()
    item_factors = model.item_embedding.weight.detach().cpu().numpy()
    _catalog_ids, embeddings, id_to_row = load_catalog_embeddings(settings)

    factory = get_session_factory()
    with factory() as session:
        titles = _load_titles(session)
        prepared = []
        skipped: Counter[str] = Counter()
        for user in users:
            row = prepare_query(
                user_id=user.user_id,
                target_product_id=user.hidden_product_id,
                title=titles.get(user.hidden_product_id),
                history_product_ids=user.inner_train_product_ids,
            )
            if not row.usable:
                skipped[row.skip_reason or "unusable"] += 1
                continue
            prepared.append(row)
        print(f"usable inner queries {len(prepared)} / {len(users)}; retrieving...", flush=True)
        started = time.perf_counter()
        cached: list[dict[str, Any]] = []
        for index, row in enumerate(prepared, start=1):
            baseline, _collected = hybrid_baseline_candidates(
                session,
                query=row.query,
                fusion_method=args.fusion_method,
                candidate_k=args.candidate_k,
            )
            ids = [item.product_id for item in baseline]
            covered = row.target_product_id in set(ids)
            content_raw = content_scores_from_matrix(
                history_ids=row.history_product_ids,
                candidate_ids=ids,
                embeddings=embeddings,
                id_to_row=id_to_row,
            )
            cf_raw = cf_scores_from_model(
                user_id=row.user_id,
                candidate_ids=ids,
                user_to_index=mappings.user_to_index,
                product_to_index=mappings.product_to_index,
                user_factors=user_factors,
                item_factors=item_factors,
            )
            cached.append(
                {
                    "prepared": row,
                    "baseline": baseline,
                    "covered": covered,
                    "content_raw": content_raw,
                    "cf_raw": cf_raw,
                }
            )
            if index % 50 == 0 or index == len(prepared):
                elapsed = time.perf_counter() - started
                print(f"  retrieved {index}/{len(prepared)} in {elapsed:.1f}s", flush=True)

    comparison: dict[str, dict[str, float]] = {}
    details: dict[str, Any] = {}
    for signal_name, weights in signal_grid():
        for gamma in gamma_grid():
            name = f"{signal_name}|g{gamma:.2f}"
            rows = []
            for item in cached:
                pers_ids = rerank_with_config(
                    item["baseline"],
                    content_raw=item["content_raw"],
                    cf_raw=item["cf_raw"],
                    weights=weights,
                    gamma=gamma,
                )
                base_ids = [row.product_id for row in item["baseline"]]
                if set(base_ids) != set(pers_ids) or len(base_ids) != len(pers_ids):
                    raise AssertionError("inner tuning violated candidate identity")
                payload = evaluate_ranking_pair(
                    base_ids,
                    pers_ids,
                    item["prepared"].target_product_id,
                    covered=item["covered"],
                )
                rows.append(attach_history_bin(payload, item["prepared"].history_size))
            summary = summarize_eval_rows(rows)
            comparison[name] = {
                **summary["personalized"],
                "gamma": gamma,
                "content_weight": weights.content,
                "cf_weight": weights.cf,
                "candidate_coverage": summary["candidate_coverage"],
            }
            details[name] = {"n": summary["n"], "coverage": summary["candidate_coverage"]}
            print(
                f"{name}: NDCG@10={summary['personalized']['ndcg@10']:.4f} "
                f"R@10={summary['personalized']['recall@10']:.4f}",
                flush=True,
            )

    winner = select_inner_winner(comparison)
    signal_name, gamma_token = winner.split("|", 1)
    gamma = float(gamma_token[1:])
    weights = dict(signal_grid())[signal_name]
    policy = load_personalization_policy()
    payload = policy_to_dict(policy)
    payload["weights"] = weights.as_dict()
    payload["gamma"] = gamma
    payload["created_at"] = datetime.now(UTC).isoformat()
    payload["git_commit"] = _git_commit()
    payload["note"] = (
        f"Frozen from {TUNE_PROTOCOL} winner={winner}. External test unused. "
        "Temporary inner-train BPR-MF is not a new CF model."
    )
    frozen = policy_from_dict(payload)
    if args.freeze_policy:
        write_personalization_policy(frozen)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "tune_protocol": TUNE_PROTOCOL,
        "personalization_version": PERSONALIZATION_VERSION,
        "evaluated_users": len(cached),
        "eligible_validation_users": len(tune.validation_users),
        "sampled": bool(args.max_users),
        "seed": args.seed,
        "candidate_k": args.candidate_k,
        "fusion_method": args.fusion_method,
        "winner": winner,
        "comparison": comparison,
        "selected_weights": weights.as_dict(),
        "selected_gamma": gamma,
        "skipped": dict(skipped),
        "git_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "selection_rule": "max inner NDCG@10; ties prefer smaller gamma then higher content weight",
        "label_class": "synthetic_query_observed_target",
    }
    out_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"winner {winner} wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
