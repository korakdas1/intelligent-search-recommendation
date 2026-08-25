#!/usr/bin/env python3
"""Real-catalog diagnostics for content recommendations. Not NDCG."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import func, select, text

from app.db.session import get_session_factory
from app.models.interaction import Interaction
from app.models.product import Product
from app.recommendations.service import (
    recommend_content_for_user,
    recommend_popular,
    recommend_similar,
)
from app.search.runtime import load_semantic_runtime, set_semantic_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose similar-item / user / popularity recs")
    parser.add_argument("--product-id", action="append", default=[])
    parser.add_argument("--repeats", type=int, default=30)
    return parser.parse_args()


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[index]


def main() -> int:
    args = parse_args()
    runtime = load_semantic_runtime()
    set_semantic_runtime(runtime)
    factory = get_session_factory()
    with factory() as session:
        product_ids = list(args.product_id)
        if not product_ids:
            howard = session.scalar(
                select(Product.product_id).where(Product.title.ilike("%leather conditioner%")).limit(1)
            )
            if howard:
                product_ids.append(howard)
            extra = list(
                session.scalars(select(Product.product_id).order_by(Product.product_id).limit(4))
            )
            for item in extra:
                if item not in product_ids:
                    product_ids.append(item)
        zero = session.execute(
            text(
                """
                SELECT p.product_id, p.title
                FROM products p
                LEFT JOIN interactions i ON i.product_id = p.product_id
                WHERE i.product_id IS NULL
                ORDER BY p.product_id
                LIMIT 1
                """
            )
        ).first()
        sql_top = session.execute(
            text(
                """
                SELECT product_id, COUNT(*) AS cnt
                FROM interactions
                GROUP BY product_id
                ORDER BY cnt DESC, product_id ASC
                LIMIT 5
                """
            )
        ).all()
        users = list(
            session.execute(
                select(Interaction.user_id, func.count(func.distinct(Interaction.product_id)))
                .group_by(Interaction.user_id)
                .having(func.count(func.distinct(Interaction.product_id)) >= 3)
                .order_by(Interaction.user_id)
                .limit(3)
            )
        )
        single = session.execute(
            select(Interaction.user_id)
            .group_by(Interaction.user_id)
            .having(func.count(func.distinct(Interaction.product_id)) == 1)
            .order_by(Interaction.user_id)
            .limit(1)
        ).scalar_one_or_none()

        similar_examples = []
        similar_ms: list[float] = []
        for product_id in product_ids[:6]:
            started = time.perf_counter()
            response = recommend_similar(session, product_id, top_k=5, runtime=runtime)
            similar_ms.append((time.perf_counter() - started) * 1000)
            similar_examples.append(
                {
                    "product_id": product_id,
                    "source_title": session.get(Product, product_id).title if session.get(Product, product_id) else None,
                    "results": [
                        {"product_id": row.product_id, "title": row.title, "score": row.score}
                        for row in response.results
                    ],
                }
            )
        for _ in range(max(0, args.repeats - len(product_ids))):
            started = time.perf_counter()
            recommend_similar(session, product_ids[0], top_k=10, runtime=runtime)
            similar_ms.append((time.perf_counter() - started) * 1000)

        trending = recommend_popular(session, top_k=5)
        pop_ms: list[float] = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            recommend_popular(session, top_k=10)
            pop_ms.append((time.perf_counter() - started) * 1000)

        user_examples = []
        user_ms: list[float] = []
        sample_users = [row[0] for row in users]
        if single:
            sample_users.append(single)
        for user_id in sample_users:
            history = list(
                session.scalars(
                    select(Product.title)
                    .join(Interaction, Interaction.product_id == Product.product_id)
                    .where(Interaction.user_id == user_id)
                    .distinct()
                    .limit(6)
                )
            )
            started = time.perf_counter()
            recs = recommend_content_for_user(session, user_id, top_k=5, runtime=runtime)
            user_ms.append((time.perf_counter() - started) * 1000)
            user_examples.append(
                {
                    "history_titles": history,
                    "history_items": recs.history_items,
                    "results": [
                        {"product_id": row.product_id, "title": row.title, "score": round(row.score, 4)}
                        for row in recs.results
                    ],
                }
            )
        if zero:
            zero_id, zero_title = zero
            zero_rec = recommend_similar(session, zero_id, top_k=5, runtime=runtime)

        payload = {
            "similar_examples": similar_examples,
            "zero_interaction_product": (
                {
                    "product_id": zero[0],
                    "title": zero[1],
                    "results": [
                        {"product_id": row.product_id, "title": row.title, "score": row.score}
                        for row in zero_rec.results
                    ],
                }
                if zero
                else None
            ),
            "sql_top5": [{"product_id": row[0], "count": int(row[1])} for row in sql_top],
            "trending_top5": [
                {"product_id": row.product_id, "title": row.title, "score": row.score}
                for row in trending.results
            ],
            "user_examples": user_examples,
            "sample_user_ids": sample_users,
            "latency_ms": {
                "similar_median": round(_pct(similar_ms, 50), 3),
                "similar_p95": round(_pct(similar_ms, 95), 3),
                "popularity_median": round(_pct(pop_ms, 50), 3),
                "popularity_p95": round(_pct(pop_ms, 95), 3),
                "user_content_median": round(_pct(user_ms, 50), 3),
                "user_content_p95": round(_pct(user_ms, 95), 3),
                "similar_n": len(similar_ms),
                "popularity_n": len(pop_ms),
                "user_n": len(user_ms),
            },
        }
        print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
