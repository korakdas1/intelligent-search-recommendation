#!/usr/bin/env python3
"""Local semantic-search diagnostics. Not production latency or NDCG."""

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
from app.db.session import get_session_factory, reset_engine
from app.embeddings.normalize import l2_normalize
from app.models.product import Product
from app.search.faiss_index import search_index
from app.search.runtime import load_semantic_runtime, reset_semantic_runtime, set_semantic_runtime
from app.search.semantic import semantic_search

SANITY_QUERIES = [
    "product for restoring leather",
    "sun protection lotion",
    "hair washing product for dry hair",
    "face moisturizer for sensitive skin",
    "product that reduces frizz",
    "leather conditioner",
    "shampoo",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--synthetic-n", type=int, default=25)
    parser.add_argument("--out", default="artifacts/evaluation/E-003/semantic_latency.json")
    args = parser.parse_args(argv)

    settings = get_settings()
    reset_engine()
    reset_semantic_runtime()
    runtime = load_semantic_runtime(settings)
    set_semantic_runtime(runtime)
    factory = get_session_factory()

    encode_ms: list[float] = []
    faiss_ms: list[float] = []
    fetch_ms: list[float] = []
    e2e_ms: list[float] = []
    sanity_results: list[dict] = []

    with factory() as session:
        encoder = runtime.encoder
        if encoder is None:
            from app.search.semantic import _ensure_encoder

            encoder = _ensure_encoder(runtime)

        for _ in range(args.warmup):
            for query in SANITY_QUERIES:
                semantic_search(session, query=query, top_k=args.top_k, log=False, runtime=runtime)
                session.rollback()

        for query in SANITY_QUERIES:
            semantic_search(session, query=query, top_k=args.top_k, log=False, runtime=runtime)
            # capture titles from a logged-off call
            response = semantic_search(session, query=query, top_k=5, log=False, runtime=runtime)
            sanity_results.append(
                {
                    "query": query,
                    "titles": [item.title for item in response.results],
                    "product_ids": [item.product_id for item in response.results],
                    "scores": [round(item.score, 4) for item in response.results],
                }
            )
            session.rollback()

        for _ in range(args.repeats):
            for query in SANITY_QUERIES:
                started = time.perf_counter()
                vector = l2_normalize(encoder.encode([query], batch_size=1, show_progress=False))
                encode_ms.append((time.perf_counter() - started) * 1000)
                started = time.perf_counter()
                search_index(runtime.index, vector, args.top_k)
                faiss_ms.append((time.perf_counter() - started) * 1000)
                started = time.perf_counter()
                semantic_search(session, query=query, top_k=args.top_k, log=False, runtime=runtime)
                e2e_ms.append((time.perf_counter() - started) * 1000)
                session.rollback()

        stmt = (
            select(Product.product_id, Product.title)
            .where(Product.dataset_version == runtime.dataset_version)
            .order_by(Product.product_id.asc())
            .limit(args.synthetic_n)
        )
        hits = 0
        samples = list(session.execute(stmt))
        for product_id, title in samples:
            response = semantic_search(
                session,
                query=title,
                top_k=args.top_k,
                log=False,
                runtime=runtime,
            )
            session.rollback()
            if any(item.product_id == product_id for item in response.results):
                hits += 1
        synthetic = {
            "label_class": "synthetic",
            "n": len(samples),
            "top_k": args.top_k,
            "hits": hits,
            "hit_rate": round(hits / len(samples), 4) if samples else None,
            "note": "Title used as query; not a human relevance judgment.",
        }

    report = {
        "artifact_version": runtime.artifact_version,
        "model_name": runtime.model_name,
        "backend": runtime.backend,
        "device": getattr(runtime.encoder, "device", None),
        "product_count": runtime.ntotal,
        "query_count": len(SANITY_QUERIES),
        "queries": SANITY_QUERIES,
        "warmup_passes": args.warmup,
        "timed_repeats": args.repeats,
        "top_k": args.top_k,
        "latency_ms": {
            "encode": _summary(encode_ms),
            "faiss": _summary(faiss_ms),
            "end_to_end": _summary(e2e_ms),
        },
        "sanity_results": sanity_results,
        "synthetic_self_retrieval": synthetic,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("latency_ms", "synthetic_self_retrieval", "backend")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
