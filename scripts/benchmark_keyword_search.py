#!/usr/bin/env python3
"""Local keyword-search latency baseline. Not a production load test."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from app.core.config import get_settings
from app.db.session import get_engine, get_session_factory, reset_engine
from app.search.service import keyword_search

QUERIES = [
    "leather conditioner",
    "sunscreen",
    "shampoo",
    "howard",
    '"leather conditioner"',
    "moisturizer cream",
    "xyzzy-no-such-product-aaa",
    "the",
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-n", type=int, default=25)
    args = parser.parse_args(argv)

    settings = get_settings()
    reset_engine()
    engine = get_engine()
    factory = get_session_factory()
    times_ms: list[float] = []
    with factory() as session:
        for _ in range(args.warmup):
            for query in QUERIES:
                keyword_search(session, query=query, top_k=args.top_k, log=False)
                session.rollback()
        for _ in range(args.repeats):
            for query in QUERIES:
                started = time.perf_counter()
                keyword_search(session, query=query, top_k=args.top_k, log=False)
                times_ms.append((time.perf_counter() - started) * 1000)
                session.rollback()

        report = {
            "postgres_db": settings.postgres_db,
            "query_count": len(QUERIES),
            "queries": QUERIES,
            "warmup_passes": args.warmup,
            "timed_repeats": args.repeats,
            "timed_executions": len(times_ms),
            "top_k": args.top_k,
            "latency_ms": {
                "min": round(min(times_ms), 3),
                "median": round(statistics.median(times_ms), 3),
                "p95": round(_percentile(times_ms, 95), 3),
                "max": round(max(times_ms), 3),
            },
        }
        if args.synthetic:
            from sqlalchemy import select

            from app.models.product import Product

            titles = session.execute(
                select(Product.product_id, Product.title)
                .where(Product.title.is_not(None))
                .order_by(Product.product_id)
                .limit(args.synthetic_n * 4)
            ).all()
            hits_at_10 = 0
            considered = 0
            for product_id, title in titles:
                tokens = [tok for tok in title.replace("/", " ").split() if tok.isalpha()]
                if len(tokens) < 2:
                    continue
                query = " ".join(tokens[:2])
                response = keyword_search(session, query=query, top_k=10, log=False)
                session.rollback()
                ids = [item.product_id for item in response.results]
                considered += 1
                if product_id in ids:
                    hits_at_10 += 1
                if considered >= args.synthetic_n:
                    break
            report["synthetic_self_retrieval"] = {
                "label_class": "synthetic",
                "n": considered,
                "source_in_top_10": hits_at_10,
                "hit_rate_top_10": round(hits_at_10 / considered, 4) if considered else None,
                "note": "Title-token self-retrieval. Not human relevance.",
            }
    print(json.dumps(report, indent=2))
    out = Path("artifacts/evaluation/E-002")
    out.mkdir(parents=True, exist_ok=True)
    (out / "keyword_latency.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
