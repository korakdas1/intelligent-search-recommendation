#!/usr/bin/env python3
"""Hybrid fusion diagnostics. Synthetic and qualitative only — not NDCG."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select

from app.core.config import get_settings
from app.db.repositories.products import get_products_by_ids
from app.db.repositories.search import keyword_candidates
from app.search.candidates import resolve_candidate_k, union_candidates
from app.search.fusion import fuse_rrf, fuse_weighted
from app.search.hybrid import hybrid_search
from app.search.runtime import load_semantic_runtime, reset_semantic_runtime, set_semantic_runtime
from app.search.semantic import semantic_candidates, semantic_search
from app.search.service import keyword_search
from app.db.session import get_session_factory, reset_engine
from app.models.product import Product

QUALITATIVE_QUERIES = [
    "leather conditioner",
    "howard leather",
    "sunscreen",
    "product for restoring leather",
    "face moisturizer for sensitive skin",
    "product that reduces frizz",
    "xyzzyqwerty12345notaproduct",
    "the",
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 95), 3),
        "max": round(max(values), 3),
    }


def _compact(response) -> list[dict]:
    return [
        {
            "rank": item.rank,
            "product_id": item.product_id,
            "title": item.title,
            "score": round(item.score, 6),
            "source": item.source,
        }
        for item in response.results
    ]


def _overlap(keyword_ids: set[str], semantic_ids: set[str]) -> dict:
    intersection = keyword_ids & semantic_ids
    union = keyword_ids | semantic_ids
    only_keyword = keyword_ids - semantic_ids
    only_semantic = semantic_ids - keyword_ids
    union_n = len(union)
    return {
        "keyword_count": len(keyword_ids),
        "semantic_count": len(semantic_ids),
        "intersection": len(intersection),
        "union": union_n,
        "only_keyword": len(only_keyword),
        "only_semantic": len(only_semantic),
        "jaccard": round(len(intersection) / union_n, 4) if union_n else None,
        "pct_only_keyword": round(100 * len(only_keyword) / union_n, 2) if union_n else None,
        "pct_only_semantic": round(100 * len(only_semantic) / union_n, 2) if union_n else None,
        "pct_both": round(100 * len(intersection) / union_n, 2) if union_n else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--compare-k", type=int, default=5)
    parser.add_argument("--synthetic-n", type=int, default=100)
    parser.add_argument("--out", default="artifacts/evaluation/E-005/hybrid_diagnostics.json")
    args = parser.parse_args(argv)

    settings = get_settings()
    reset_engine()
    reset_semantic_runtime()
    runtime = load_semantic_runtime(settings)
    set_semantic_runtime(runtime)
    factory = get_session_factory()
    candidate_k = resolve_candidate_k(
        args.top_k,
        configured=settings.hybrid_candidate_k,
        maximum=settings.hybrid_candidate_k_max,
    )

    qualitative: list[dict] = []
    overlaps: list[dict] = []
    keyword_ms: list[float] = []
    encode_ms: list[float] = []
    faiss_ms: list[float] = []
    fusion_ms: list[float] = []
    fetch_ms: list[float] = []
    rrf_e2e_ms: list[float] = []
    weighted_e2e_ms: list[float] = []

    with factory() as session:
        from app.search.semantic import _ensure_encoder

        encoder = _ensure_encoder(runtime)

        for _ in range(args.warmup):
            for query in QUALITATIVE_QUERIES:
                hybrid_search(
                    session,
                    query=query,
                    top_k=args.top_k,
                    fusion_method="rrf",
                    log=False,
                    candidate_k=candidate_k,
                )
                session.rollback()

        for query in QUALITATIVE_QUERIES:
            keyword = keyword_search(session, query=query, top_k=args.compare_k, log=False)
            semantic = semantic_search(
                session, query=query, top_k=args.compare_k, log=False, runtime=runtime
            )
            rrf = hybrid_search(
                session,
                query=query,
                top_k=args.compare_k,
                fusion_method="rrf",
                log=False,
                candidate_k=candidate_k,
            )
            weighted = hybrid_search(
                session,
                query=query,
                top_k=args.compare_k,
                fusion_method="weighted",
                log=False,
                candidate_k=candidate_k,
                keyword_weight=0.5,
            )
            k_cands = keyword_candidates(session, query, candidate_k)
            s_cands = semantic_candidates(
                session, query=query, top_k=candidate_k, runtime=runtime
            )
            overlap = _overlap(
                {row.product_id for row in k_cands},
                {row.product_id for row in s_cands},
            )
            overlap["query"] = query
            overlaps.append(overlap)
            qualitative.append(
                {
                    "query": query,
                    "keyword": _compact(keyword),
                    "semantic": _compact(semantic),
                    "hybrid_rrf": _compact(rrf),
                    "hybrid_weighted_0_50": _compact(weighted),
                }
            )
            session.rollback()

        timed_queries = [q for q in QUALITATIVE_QUERIES if q not in {"the"}]
        for _ in range(args.repeats):
            for query in timed_queries:
                started = time.perf_counter()
                k_cands = keyword_candidates(session, query, candidate_k)
                keyword_ms.append((time.perf_counter() - started) * 1000)

                started = time.perf_counter()
                vector = encoder.encode([query], batch_size=1, show_progress=False)
                encode_ms.append((time.perf_counter() - started) * 1000)
                started = time.perf_counter()
                s_cands = semantic_candidates(
                    session, query=query, top_k=candidate_k, runtime=runtime
                )
                faiss_ms.append((time.perf_counter() - started) * 1000)

                started = time.perf_counter()
                merged = union_candidates(k_cands, s_cands)
                fused = fuse_rrf(merged, k0=settings.hybrid_rrf_k)[: args.top_k]
                fusion_ms.append((time.perf_counter() - started) * 1000)

                started = time.perf_counter()
                get_products_by_ids(session, [row.product_id for row in fused])
                fetch_ms.append((time.perf_counter() - started) * 1000)

                started = time.perf_counter()
                hybrid_search(
                    session,
                    query=query,
                    top_k=args.top_k,
                    fusion_method="rrf",
                    log=False,
                    candidate_k=candidate_k,
                )
                rrf_e2e_ms.append((time.perf_counter() - started) * 1000)

                started = time.perf_counter()
                hybrid_search(
                    session,
                    query=query,
                    top_k=args.top_k,
                    fusion_method="weighted",
                    log=False,
                    candidate_k=candidate_k,
                )
                weighted_e2e_ms.append((time.perf_counter() - started) * 1000)
                session.rollback()

        stmt = (
            select(Product.product_id, Product.title)
            .where(Product.dataset_version == runtime.dataset_version)
            .where(Product.title.is_not(None))
            .order_by(Product.product_id.asc())
            .limit(args.synthetic_n)
        )
        samples = list(session.execute(stmt))
        systems = {
            "keyword": 0,
            "semantic": 0,
            "hybrid_rrf": 0,
            "hybrid_weighted_0_25": 0,
            "hybrid_weighted_0_50": 0,
            "hybrid_weighted_0_75": 0,
        }
        for product_id, title in samples:
            if not title:
                continue
            keyword = keyword_search(session, query=title, top_k=args.top_k, log=False)
            semantic = semantic_search(
                session, query=title, top_k=args.top_k, log=False, runtime=runtime
            )
            rrf = hybrid_search(
                session,
                query=title,
                top_k=args.top_k,
                fusion_method="rrf",
                log=False,
                candidate_k=candidate_k,
            )
            weights = {
                "hybrid_weighted_0_25": 0.25,
                "hybrid_weighted_0_50": 0.50,
                "hybrid_weighted_0_75": 0.75,
            }
            if any(item.product_id == product_id for item in keyword.results):
                systems["keyword"] += 1
            if any(item.product_id == product_id for item in semantic.results):
                systems["semantic"] += 1
            if any(item.product_id == product_id for item in rrf.results):
                systems["hybrid_rrf"] += 1
            for name, alpha in weights.items():
                weighted = hybrid_search(
                    session,
                    query=title,
                    top_k=args.top_k,
                    fusion_method="weighted",
                    log=False,
                    candidate_k=candidate_k,
                    keyword_weight=alpha,
                )
                if any(item.product_id == product_id for item in weighted.results):
                    systems[name] += 1
            session.rollback()

        n = len(samples)
        synthetic = {
            "label": "synthetic title-as-query self-retrieval",
            "label_class": "synthetic",
            "n": n,
            "top_k": args.top_k,
            "candidate_k": candidate_k,
            "hits": systems,
            "hit_rate": {name: round(hits / n, 4) if n else None for name, hits in systems.items()},
            "note": (
                "Title used as query; this is not a human relevance judgment "
                "and is not NDCG. It only checks catastrophic fusion failures."
            ),
        }

    overlap_summary = {
        "queries": overlaps,
        "mean_jaccard": round(
            statistics.mean([row["jaccard"] for row in overlaps if row["jaccard"] is not None]),
            4,
        )
        if any(row["jaccard"] is not None for row in overlaps)
        else None,
    }

    report = {
        "dataset_version": runtime.dataset_version,
        "semantic_artifact_version": runtime.artifact_version,
        "embedding_model": runtime.model_name,
        "semantic_backend": runtime.backend,
        "product_count": runtime.ntotal,
        "candidate_k": candidate_k,
        "rrf_k": settings.hybrid_rrf_k,
        "weighted_alphas": [0.25, 0.50, 0.75],
        "provisional_fusion_default": settings.hybrid_fusion_method,
        "query_set": QUALITATIVE_QUERIES,
        "top_k": args.top_k,
        "compare_k": args.compare_k,
        "warmup_passes": args.warmup,
        "timed_repeats": args.repeats,
        "timed_query_count": len(timed_queries),
        "qualitative": qualitative,
        "candidate_overlap": overlap_summary,
        "synthetic_self_retrieval": synthetic,
        "latency_ms": {
            "keyword_candidates": _summary(keyword_ms),
            "query_encode_plus_faiss_candidates": _summary(faiss_ms),
            "query_encode": _summary(encode_ms),
            "fusion": _summary(fusion_ms),
            "metadata_fetch": _summary(fetch_ms),
            "hybrid_rrf_end_to_end": _summary(rrf_e2e_ms),
            "hybrid_weighted_end_to_end": _summary(weighted_e2e_ms),
        },
        "limitations": [
            "Not human relevance.",
            "Not NDCG.",
            "Hybrid is slower because both retrievers run.",
            "Local laptop timings are not a production SLA.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_k": candidate_k,
                "backend": runtime.backend,
                "synthetic_self_retrieval": synthetic,
                "latency_ms": report["latency_ms"],
                "mean_jaccard": overlap_summary["mean_jaccard"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
