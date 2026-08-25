#!/usr/bin/env python3
"""Evaluate frozen personalized-search-v1 on synthetic personalized-search queries.

Uses recsys-eval-v1 held-out products as observed targets and ltr-synthetic-v1
title queries. Not human search relevance. Does not tune.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.models.product import Product
from app.evaluation.constants import PERSONALIZED_EXPERIMENT_ID, OFFICIAL_COMPARISON_DIR
from app.evaluation.io import write_json_atomic
from app.recommendations.cf_artifacts import load_cf_bundle
from app.recommendations.cf_constants import MODEL_VERSION
from app.recommendations.cf_data import RecsysEvalIncompatibleError, load_recsys_eval_split
from app.recommendations.cf_probe import cf_artifact_path
from app.recommendations.constants import EVALUATION_VERSION
from app.recommendations.evaluation import history_size_bin
from app.recommendations.hybrid_offline import load_catalog_embeddings
from app.search.personalization_constants import (
    EVAL_PROTOCOL,
    EXPERIMENT_ID,
    PERSONALIZATION_VERSION,
    QUERY_GENERATOR_VERSION,
)
from app.search.personalization_eval import (
    attach_history_bin,
    cf_scores_from_model,
    content_scores_from_matrix,
    evaluate_ranking_pair,
    hybrid_baseline_candidates,
    movement_stats,
    prepare_query,
    rerank_with_config,
    segment_by_history,
    summarize_eval_rows,
)
from app.search.personalization_policy import load_personalization_policy, policy_to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-version", default=EVALUATION_VERSION)
    parser.add_argument("--tune-version", default="personalized-search-tune-v1")
    parser.add_argument("--eval-version", default=EVAL_PROTOCOL)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion-method", default="rrf")
    parser.add_argument("--latency-n", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--official",
        action="store_true",
        help="Export an official Table B summary without retuning.",
    )
    parser.add_argument("--official-output", default=None)
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


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "p95": 0.0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "median": float(ordered[n // 2]),
        "p95": float(ordered[min(n - 1, int(round(0.95 * (n - 1))))]),
    }


def _export_official(payload: dict[str, Any], dest: Path) -> None:
    identity_ok = int(payload.get("candidate_identity_ok") or 0)
    usable = int(payload.get("usable_queries") or 0)
    if identity_ok != usable:
        raise RuntimeError(
            f"candidate identity failed: {identity_ok} != usable {usable}"
        )
    summary = {
        "experiment_id": PERSONALIZED_EXPERIMENT_ID,
        "source_experiment": payload.get("experiment_id"),
        "eval_version": payload.get("eval_version"),
        "personalization_version": payload.get("personalization_version"),
        "policy": payload.get("policy"),
        "label_class": payload.get("label_class"),
        "label_note": (
            "Query class: synthetic. Target origin: observed held-out future interaction. "
            "Not human search relevance."
        ),
        "eligible_users": payload.get("eligible_users"),
        "evaluated_users": payload.get("evaluated_users"),
        "usable_queries": usable,
        "skipped": payload.get("skipped"),
        "sampled": payload.get("sampled"),
        "candidate_k": payload.get("candidate_k"),
        "candidate_identity_ok": identity_ok,
        "candidate_identity_fail": usable - identity_ok,
        "signal_coverage": payload.get("signal_coverage"),
        "all_queries": payload.get("all_queries"),
        "candidate_covered": payload.get("candidate_covered"),
        "history_segments": payload.get("history_segments"),
        "cf_segments": payload.get("cf_segments"),
        "latency_ms": payload.get("latency_ms"),
        "not_table_a": True,
        "not_human_relevance": True,
        "source_artifact": "artifacts/evaluation/personalized-search-v1/evaluation.json",
        "reran": False,
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(dest, summary)


def main() -> int:
    args = parse_args()
    settings = get_settings()
    policy = load_personalization_policy()
    split_dir = Path(args.split_dir or Path(settings.artifacts_root) / "evaluation" / args.evaluation_version)
    out_dir = Path(args.out_dir or Path(settings.artifacts_root) / "evaluation" / PERSONALIZATION_VERSION)
    out_path = out_dir / "evaluation.json"
    official_path = Path(args.official_output) if args.official_output else (
        Path(settings.artifacts_root) / "evaluation" / OFFICIAL_COMPARISON_DIR / "personalized_search_comparison.json"
    )
    if args.official and out_path.is_file() and not args.force:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        _export_official(payload, official_path)
        print(f"exported official Table B to {official_path} from existing {out_path}")
        return 0
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

    model_dir = Path(args.model_dir or cf_artifact_path(settings))
    model, cf_user_ids, cf_item_ids, _config, manifest = load_cf_bundle(model_dir)
    if manifest.get("hidden_checksum") and manifest["hidden_checksum"] != split.hidden_checksum:
        print("CF artifact hidden checksum does not match recsys-eval-v1. Stop.", file=sys.stderr)
        return 2
    user_factors = model.user_embedding.weight.detach().cpu().numpy()
    item_factors = model.item_embedding.weight.detach().cpu().numpy()
    user_to_index = {user_id: index for index, user_id in enumerate(cf_user_ids)}
    product_to_index = {product_id: index for index, product_id in enumerate(cf_item_ids)}
    _catalog_ids, embeddings, id_to_row = load_catalog_embeddings(settings)
    candidate_k = args.candidate_k or policy.candidate_k

    factory = get_session_factory()
    with factory() as session:
        titles = {
            str(product_id): str(title or "")
            for product_id, title in session.execute(select(Product.product_id, Product.title))
        }
        skipped: Counter[str] = Counter()
        prepared = []
        for user in users:
            row = prepare_query(
                user_id=user.user_id,
                target_product_id=user.hidden_product_id,
                title=titles.get(user.hidden_product_id),
                history_product_ids=user.train_product_ids,
            )
            if not row.usable:
                skipped[row.skip_reason or "unusable"] += 1
                continue
            prepared.append(row)
        print(f"usable queries {len(prepared)} / {len(users)}; retrieving...", flush=True)
        rows: list[dict[str, Any]] = []
        covered_rows: list[dict[str, Any]] = []
        by_cf: dict[str, list[dict[str, Any]]] = defaultdict(list)
        signal_counts = Counter()
        identity_ok = 0
        promotions: list[dict[str, Any]] = []
        search_ms: list[float] = []
        rerank_ms: list[float] = []
        started = time.perf_counter()
        for index, row in enumerate(prepared, start=1):
            t0 = time.perf_counter()
            baseline, _collected = hybrid_baseline_candidates(
                session,
                query=row.query,
                fusion_method=args.fusion_method,
                candidate_k=candidate_k,
            )
            search_ms.append((time.perf_counter() - t0) * 1000)
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
                user_to_index=user_to_index,
                product_to_index=product_to_index,
                user_factors=user_factors,
                item_factors=item_factors,
            )
            t1 = time.perf_counter()
            pers_ids = rerank_with_config(
                baseline,
                content_raw=content_raw,
                cf_raw=cf_raw,
                weights=policy.weights,
                gamma=policy.gamma,
            )
            rerank_ms.append((time.perf_counter() - t1) * 1000)
            if set(ids) != set(pers_ids) or len(ids) != len(pers_ids):
                raise AssertionError("evaluation violated candidate identity")
            identity_ok += 1
            payload = attach_history_bin(
                evaluate_ranking_pair(ids, pers_ids, row.target_product_id, covered=covered),
                row.history_size,
            )
            payload["content"] = content_raw is not None
            payload["cf"] = cf_raw is not None
            if content_raw is not None and cf_raw is not None:
                signal_counts["both"] += 1
            elif content_raw is not None:
                signal_counts["content_only"] += 1
            elif cf_raw is not None:
                signal_counts["cf_only"] += 1
            else:
                signal_counts["neither"] += 1
            rows.append(payload)
            if covered:
                covered_rows.append(payload)
            by_cf["cf" if cf_raw is not None else "content_only"].append(payload)
            stats = movement_stats(ids, pers_ids, target_id=row.target_product_id)
            if stats["max_promotion"] >= 20:
                promotions.append(
                    {
                        "user_id": row.user_id,
                        "query": row.query,
                        "max_promotion": stats["max_promotion"],
                        "target_baseline_rank": stats["target_baseline_rank"],
                        "target_personalized_rank": stats["target_personalized_rank"],
                    }
                )
            if index % 50 == 0 or index == len(prepared):
                print(
                    f"  {index}/{len(prepared)} in {time.perf_counter() - started:.1f}s "
                    f"coverage={sum(1 for item in rows if item['covered']) / len(rows):.3f}",
                    flush=True,
                )
            if args.latency_n and index >= args.latency_n and index == args.latency_n:
                pass

    all_summary = summarize_eval_rows(rows)
    covered_summary = summarize_eval_rows(covered_rows)
    history = segment_by_history(rows)
    cf_segments = {name: summarize_eval_rows(items) for name, items in by_cf.items()}
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "eval_version": args.eval_version,
        "personalization_version": PERSONALIZATION_VERSION,
        "policy": policy_to_dict(policy),
        "query_generator_version": QUERY_GENERATOR_VERSION,
        "label_class": "synthetic_query_observed_target",
        "base_retrieval_mode": "hybrid",
        "fusion_method": args.fusion_method,
        "ltr": False,
        "candidate_k": candidate_k,
        "eligible_users": len(split.users),
        "evaluated_users": len(users),
        "usable_queries": len(prepared),
        "sampled": sampled,
        "seed": args.seed if sampled else None,
        "skipped": dict(skipped),
        "candidate_identity_ok": identity_ok,
        "signal_coverage": dict(signal_counts),
        "all_queries": all_summary,
        "candidate_covered": covered_summary,
        "history_segments": history,
        "cf_segments": cf_segments,
        "promotions": promotions[:12],
        "latency_ms": {
            "search": _percentiles(search_ms),
            "rerank": _percentiles(rerank_ms),
            "n": len(search_ms),
        },
        "cf_model_version": MODEL_VERSION,
        "git_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_sec": time.perf_counter() - started,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "eval_version": args.eval_version,
        "personalization_version": PERSONALIZATION_VERSION,
        "recsys_eval": args.evaluation_version,
        "hidden_checksum": split.hidden_checksum,
        "usable_queries": len(prepared),
        "candidate_k": candidate_k,
        "gamma": policy.gamma,
        "weights": policy.weights.as_dict(),
        "git_commit": _git_commit(),
        "created_at": payload["created_at"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_queries": all_summary, "covered": covered_summary}, indent=2, default=str))
    print(f"wrote {out_path}", flush=True)
    if args.official:
        _export_official(payload, official_path)
        print(f"exported official Table B to {official_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
