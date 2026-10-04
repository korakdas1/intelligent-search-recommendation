#!/usr/bin/env python3
"""Official synthetic search comparison on the frozen ltr-synthetic-v1 TEST split.

Does not train or retune. Label class is synthetic, not human relevance.
Uses the same keyword/semantic/hybrid/LTR serving functions with log=False.
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

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.evaluation.constants import (
    K_VALUES,
    LATENCY_SAMPLE_DEFAULT,
    METRIC_VERSION,
    OFFICIAL_SEARCH_EVAL,
    OFFICIAL_COMPARISON_DIR,
    RECONCILE_ABS_TOL,
    SEARCH_EXPERIMENT_ID,
    SEARCH_RESULT_K,
)
from app.evaluation.io import write_json_atomic
from app.evaluation.metrics import macro_average, metric_bundle, unique_preserve_order
from app.evaluation.provenance import (
    ProvenanceError,
    ranker_provenance,
    semantic_provenance,
    validate_ltr_dataset,
)
from app.ranking.constants import DATASET_VERSION
from app.ranking.runtime import require_ltr_runtime
from app.search.candidates import resolve_candidate_k
from app.search.hybrid import FUSION_RRF, FUSION_WEIGHTED, collect_hybrid_fused, hybrid_search
from app.search.ltr import ltr_search, score_ltr_union
from app.search.runtime import require_semantic_runtime
from app.search.service import keyword_search
from app.search.semantic import semantic_search

SYSTEMS = (
    "keyword",
    "semantic",
    "hybrid_weighted",
    "hybrid_rrf",
    "hybrid_ltr",
)
HYBRID_SYSTEMS = ("hybrid_weighted", "hybrid_rrf", "hybrid_ltr")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=f"artifacts/ltr/{DATASET_VERSION}")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-k", type=int, default=SEARCH_RESULT_K)
    parser.add_argument("--latency-sample-size", type=int, default=LATENCY_SAMPLE_DEFAULT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _ids(response_or_rows: Any) -> list[str]:
    if hasattr(response_or_rows, "results"):
        return unique_preserve_order([item.product_id for item in response_or_rows.results])
    return unique_preserve_order([str(item) for item in response_or_rows])


def _run_systems(
    session: Session,
    *,
    query: str,
    candidate_k: int,
) -> dict[str, list[str]]:
    keyword = keyword_search(session, query=query, top_k=candidate_k, log=False)
    semantic = semantic_search(session, query=query, top_k=candidate_k, log=False)
    rrf = collect_hybrid_fused(
        session,
        query=query,
        top_k=candidate_k,
        fusion_method=FUSION_RRF,
        candidate_k=candidate_k,
    )
    weighted = collect_hybrid_fused(
        session,
        query=query,
        top_k=candidate_k,
        fusion_method=FUSION_WEIGHTED,
        candidate_k=candidate_k,
    )
    ltr_rows, _depth = score_ltr_union(
        session,
        query=query,
        top_k=candidate_k,
        fusion_method=FUSION_RRF,
        candidate_k=candidate_k,
    )
    return {
        "keyword": _ids(keyword),
        "semantic": _ids(semantic),
        "hybrid_rrf": [row.product_id for row in rrf.fused],
        "hybrid_weighted": [row.product_id for row in weighted.fused],
        "hybrid_ltr": [row.product_id for row in ltr_rows],
        "_rrf_union": rrf.union_count,
        "_weighted_union": weighted.union_count,
        "_rrf_depth": rrf.depth,
        "_weighted_depth": weighted.depth,
    }


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "p95": 0.0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "median": float(ordered[n // 2]),
        "p95": float(ordered[min(n - 1, int(round(0.95 * (n - 1))))]),
    }


def _coverage(ranked: list[str], relevant: list[str]) -> bool:
    return bool(set(relevant) & set(ranked))


def _family_metrics(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_family[row["query_source"]].append(row["metrics"][system])
    return {family: macro_average(items) for family, items in sorted(by_family.items())}


def _load_e007(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "evaluation.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _reconcile(live: dict[str, float], prior: dict[str, float]) -> dict[str, Any]:
    keys = sorted(set(live) | set(prior))
    diffs = {}
    max_abs = 0.0
    for key in keys:
        delta = float(live.get(key, 0.0)) - float(prior.get(key, 0.0))
        diffs[key] = delta
        max_abs = max(max_abs, abs(delta))
    return {
        "max_abs_delta": max_abs,
        "within_tolerance": max_abs <= RECONCILE_ABS_TOL,
        "deltas": diffs,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    dataset_dir = Path(args.dataset)
    queries_path = dataset_dir / "queries.json"
    if not queries_path.is_file():
        print(f"missing {queries_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.output) if args.output else Path(settings.artifacts_root) / "evaluation" / OFFICIAL_COMPARISON_DIR
    out_path = out_dir / "search_comparison.json"
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite {out_path}; pass --force", file=sys.stderr)
        return 2

    try:
        _manifest, dataset_provenance = validate_ltr_dataset(
            dataset_dir, required_files=("queries.json",),
        )
    except ProvenanceError as exc:
        print(f"evaluation provenance rejected: {exc}", file=sys.stderr)
        return 2
    raw_queries = json.loads(queries_path.read_text(encoding="utf-8"))
    split_queries = [row for row in raw_queries if row.get("split") == args.split]
    skipped: dict[str, int] = defaultdict(int)
    usable: list[dict[str, Any]] = []
    for row in split_queries:
        query = str(row.get("query") or "").strip()
        relevant = [str(item) for item in row.get("relevant_ids") or []]
        if not query:
            skipped["blank_query"] += 1
            continue
        if not relevant:
            skipped["no_relevant_ids"] += 1
            continue
        usable.append(row)
    if args.max_queries is not None:
        usable = usable[: max(0, args.max_queries)]

    factory = get_session_factory()
    with factory() as session:
        semantic_runtime = require_semantic_runtime(session)
        ltr_runtime = require_ltr_runtime()
        try:
            ranker_identity = ranker_provenance(ltr_runtime, dataset_provenance)
            semantic_identity = semantic_provenance(
                semantic_runtime, configured_backend=settings.semantic_index_type,
            )
        except ProvenanceError as exc:
            print(f"evaluation provenance rejected: {exc}", file=sys.stderr)
            return 2
        print(
            f"{OFFICIAL_SEARCH_EVAL}: {len(usable)} usable / {len(split_queries)} {args.split} queries",
            flush=True,
        )
        records: list[dict[str, Any]] = []
        identity_ok = 0
        identity_fail = 0
        started = time.perf_counter()
        for index, row in enumerate(usable, start=1):
            query = str(row["query"]).strip()
            relevant = [str(item) for item in row["relevant_ids"]]
            ranked = _run_systems(session, query=query, candidate_k=args.candidate_k)
            rrf_ids = unique_preserve_order(ranked["hybrid_rrf"])
            weighted_ids = unique_preserve_order(ranked["hybrid_weighted"])
            ltr_ids = unique_preserve_order(ranked["hybrid_ltr"])
            same = set(rrf_ids) == set(weighted_ids) == set(ltr_ids) and len(rrf_ids) == len(weighted_ids) == len(
                ltr_ids
            )
            if same:
                identity_ok += 1
            else:
                identity_fail += 1
            metrics = {
                "keyword": metric_bundle(ranked["keyword"], relevant),
                "semantic": metric_bundle(ranked["semantic"], relevant),
                "hybrid_weighted": metric_bundle(weighted_ids, relevant),
                "hybrid_rrf": metric_bundle(rrf_ids, relevant),
                "hybrid_ltr": metric_bundle(ltr_ids, relevant),
            }
            coverage = {
                "keyword": _coverage(ranked["keyword"], relevant),
                "semantic": _coverage(ranked["semantic"], relevant),
                "hybrid_weighted": _coverage(weighted_ids, relevant),
                "hybrid_rrf": _coverage(rrf_ids, relevant),
                "hybrid_ltr": _coverage(ltr_ids, relevant),
            }
            records.append(
                {
                    "query_id": row["query_id"],
                    "query_source": row.get("query_source"),
                    "metrics": metrics,
                    "coverage": coverage,
                    "hybrid_candidate_identity": same,
                    "hybrid_union_size": len(rrf_ids),
                }
            )
            if index % 50 == 0 or index == len(usable):
                elapsed = time.perf_counter() - started
                rate = index / elapsed if elapsed else 0.0
                remaining = (len(usable) - index) / rate if rate else 0.0
                print(
                    f"  {index}/{len(usable)} in {elapsed:.1f}s remaining~{remaining:.0f}s",
                    flush=True,
                )

        latency: dict[str, Any] = {}
        sample_n = min(args.latency_sample_size, len(usable))
        if sample_n:
            sample = usable[:sample_n]
            for name, runner in (
                ("keyword", lambda q: keyword_search(session, query=q, top_k=20, log=False)),
                ("semantic", lambda q: semantic_search(session, query=q, top_k=20, log=False)),
                (
                    "hybrid_weighted",
                    lambda q: hybrid_search(
                        session,
                        query=q,
                        top_k=20,
                        fusion_method=FUSION_WEIGHTED,
                        log=False,
                        candidate_k=args.candidate_k,
                    ),
                ),
                (
                    "hybrid_rrf",
                    lambda q: hybrid_search(
                        session,
                        query=q,
                        top_k=20,
                        fusion_method=FUSION_RRF,
                        log=False,
                        candidate_k=args.candidate_k,
                    ),
                ),
                (
                    "hybrid_ltr",
                    lambda q: ltr_search(
                        session,
                        query=q,
                        top_k=20,
                        fusion_method=FUSION_RRF,
                        log=False,
                        candidate_k=args.candidate_k,
                    ),
                ),
            ):
                times: list[float] = []
                for row in sample:
                    t0 = time.perf_counter()
                    runner(str(row["query"]).strip())
                    times.append((time.perf_counter() - t0) * 1000)
                latency[name] = _percentiles(times)

    all_metrics = {system: macro_average([row["metrics"][system] for row in records]) for system in SYSTEMS}
    coverage = {
        system: sum(1 for row in records if row["coverage"][system]) / len(records) if records else 0.0
        for system in SYSTEMS
    }
    covered_by_system = {
        system: macro_average([row["metrics"][system] for row in records if row["coverage"][system]])
        for system in SYSTEMS
    }
    family = {system: _family_metrics(records, system) for system in SYSTEMS}

    e007 = _load_e007(Path(ltr_runtime.directory))
    prior_all = (e007.get("all_queries") or {}) if e007 else {}
    reconciliation = {}
    if prior_all:
        for live_name, prior_name in (
            ("hybrid_rrf", "rrf"),
            ("hybrid_weighted", "weighted"),
            ("hybrid_ltr", "ltr"),
        ):
            if prior_name in prior_all:
                reconciliation[live_name] = _reconcile(all_metrics[live_name], prior_all[prior_name])

    ndcg10 = {system: all_metrics[system].get("ndcg@10", 0.0) for system in SYSTEMS}
    winner = max(ndcg10, key=ndcg10.get) if ndcg10 else None
    input_provenance = {
        "status": "verified" if all(status == "verified" for status in (
            dataset_provenance["status"], ranker_identity["status"],
            ranker_identity["dataset_linkage_status"], semantic_identity["status"],
        )) else "legacy_unverified",
        "ltr_dataset": dataset_provenance,
        "ranker": ranker_identity,
        "semantic": semantic_identity,
    }
    effective_config = {
        "split": args.split,
        "candidate_k": args.candidate_k,
        "hybrid_candidate_k": resolve_candidate_k(
            args.candidate_k, configured=args.candidate_k, maximum=settings.hybrid_candidate_k_max,
        ),
        "k_values": list(K_VALUES),
        "rrf_k0": settings.hybrid_rrf_k,
        "weighted_alpha": settings.hybrid_keyword_weight,
        "semantic_backend": semantic_runtime.backend,
        "semantic_faiss_type": semantic_identity["faiss"]["type"],
        "semantic_hnsw_ef_search": settings.semantic_hnsw_ef_search if semantic_runtime.backend == "hnsw" else None,
        "ltr_model_version": ltr_runtime.model_version,
        "feature_version": ltr_runtime.feature_version,
        "sampled": args.max_queries is not None,
        "max_queries": args.max_queries,
        "seed": args.seed,
        "latency_sample_size_requested": args.latency_sample_size,
        "latency_sample_size": sample_n,
        "latency_result_k": 20,
        "latency_hybrid_candidate_k": resolve_candidate_k(
            20, configured=args.candidate_k, maximum=settings.hybrid_candidate_k_max,
        ),
    }
    payload = {
        "experiment_id": SEARCH_EXPERIMENT_ID,
        "evaluation_version": OFFICIAL_SEARCH_EVAL,
        "label_class": "synthetic",
        "label_note": (
            "Queries and labels are deterministic synthetic product-derived strings "
            "(ltr-synthetic-v1). Not human-issued queries. Not human relevance."
        ),
        "metric_version": METRIC_VERSION,
        "k_values": list(K_VALUES),
        "split": args.split,
        "query_generator": DATASET_VERSION,
        "dataset_dir": dataset_provenance["artifact_dir"],
        "input_provenance": input_provenance,
        "effective_config": effective_config,
        "raw_split_queries": len(split_queries),
        "usable_queries": len(records),
        "skipped": dict(skipped),
        "sampled": args.max_queries is not None,
        "max_queries": args.max_queries,
        "seed": args.seed,
        "candidate_k": args.candidate_k,
        "rrf_k0": settings.hybrid_rrf_k,
        "weighted_alpha": settings.hybrid_keyword_weight,
        "ltr_model": ltr_runtime.model_version,
        "feature_version": ltr_runtime.feature_version,
        "semantic_backend": semantic_identity["faiss"]["type"],
        "system_types": {
            "keyword": "retriever",
            "semantic": "retriever",
            "hybrid_weighted": "fusion",
            "hybrid_rrf": "fusion",
            "hybrid_ltr": "reranker",
        },
        "hybrid_candidate_identity_ok": identity_ok,
        "hybrid_candidate_identity_fail": identity_fail,
        "all_queries": all_metrics,
        "candidate_coverage": coverage,
        "candidate_covered_subset": covered_by_system,
        "by_query_family": family,
        "latency_ms": latency,
        "latency_sample_size": sample_n,
        "latency_note": "Local development latency on a modest query sample. Not a production SLA.",
        "e007_reconciliation": reconciliation,
        "synthetic_ndcg@10_winner": winner,
        "git_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "one_positive_precision_note": (
            "Each query has one designated synthetic target. Recall@K is 0/1 per query. "
            "Precision@K is 1/K on a hit."
        ),
        "not_human_relevance": True,
        "not_table_b": True,
    }
    print(json.dumps(
        {
            "path": str(out_path),
            "usable_queries": len(records),
            "all_queries": all_metrics,
            "candidate_coverage": coverage,
            "hybrid_candidate_identity_fail": identity_fail,
            "winner_ndcg@10": winner,
        },
        indent=2,
    ))
    if identity_fail:
        print("hybrid candidate identity failed; official Table A is invalid", file=sys.stderr)
        return 1
    write_json_atomic(out_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
