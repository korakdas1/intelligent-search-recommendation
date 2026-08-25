#!/usr/bin/env python3
"""Hybrid serving diagnostics: warm user, non-CF user, latency, cold item.

Not an online A/B test. Does not tune on these examples.
"""

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

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.models.interaction import Interaction
from app.models.product import Product
from app.models.user import User
from app.recommendations.cf_artifacts import load_cf_bundle
from app.recommendations.cf_constants import MODEL_VERSION
from app.recommendations.cf_data import load_recsys_eval_split
from app.recommendations.constants import EVALUATION_VERSION
from app.recommendations.service import (
    recommend_cf_for_user,
    recommend_content_for_user,
    recommend_hybrid_for_user,
    recommend_similar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--n-users", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cold-item", default="B00DHA1GY4")
    return parser.parse_args()


def _titles(session, product_ids: list[str]) -> dict[str, str]:
    if not product_ids:
        return {}
    rows = session.execute(
        select(Product.product_id, Product.title).where(Product.product_id.in_(product_ids))
    ).all()
    return {str(product_id): str(title) for product_id, title in rows}


def _summarize(response) -> dict:
    return {
        "mode": response.recommendation_mode,
        "channels_used": response.channels_used,
        "fallback_reason": response.fallback_reason,
        "reason": response.reason,
        "top": [{"product_id": row.product_id, "score": row.score, "source": row.source} for row in response.results[:5]],
    }


def main() -> int:
    args = parse_args()
    settings = get_settings()
    model_dir = Path(args.model_dir or Path(settings.artifacts_root) / "models" / MODEL_VERSION)
    split_dir = Path(args.split_dir or Path(settings.artifacts_root) / "evaluation" / EVALUATION_VERSION)
    _model, user_ids, product_ids, _config, _manifest = load_cf_bundle(model_dir)
    cf_users = set(user_ids)
    cf_items = set(product_ids)
    split = load_recsys_eval_split(split_dir, require_frozen_identity=True)
    sample = [user.user_id for user in split.users[: args.n_users]]
    factory = get_session_factory()
    latencies: list[float] = []
    payload: dict = {}
    with factory() as session:
        warm = sample[0]
        outside = None
        cold_user = None
        for user_id in session.scalars(select(User.user_id).limit(20000)):
            uid = str(user_id)
            if uid in cf_users:
                continue
            n_hist = session.scalar(
                select(func.count()).select_from(Interaction).where(Interaction.user_id == uid)
            )
            if n_hist and n_hist > 0 and outside is None:
                outside = uid
            if (not n_hist) and cold_user is None:
                cold_user = uid
            if outside and cold_user:
                break
        payload["warm_user_id"] = warm
        payload["non_cf_user_id"] = outside
        payload["no_history_user_id"] = cold_user
        payload["warm"] = {
            "content": _summarize(recommend_content_for_user(session, warm, top_k=args.top_k)),
            "cf": _summarize(recommend_cf_for_user(session, warm, top_k=args.top_k)),
            "hybrid": _summarize(recommend_hybrid_for_user(session, warm, top_k=args.top_k)),
        }
        if outside:
            payload["non_cf"] = {
                "cf": _summarize(recommend_cf_for_user(session, outside, top_k=args.top_k)),
                "hybrid": _summarize(recommend_hybrid_for_user(session, outside, top_k=args.top_k)),
            }
        if cold_user:
            payload["no_history"] = {
                "hybrid": _summarize(recommend_hybrid_for_user(session, cold_user, top_k=args.top_k)),
            }
        cold_item = str(args.cold_item)
        n_item = session.scalar(
            select(func.count()).select_from(Interaction).where(Interaction.product_id == cold_item)
        )
        similar = None
        try:
            similar = _summarize(recommend_similar(session, cold_item, top_k=3))
        except Exception as exc:
            similar = {"error": str(exc)}
        payload["cold_item"] = {
            "product_id": cold_item,
            "interaction_count": int(n_item or 0),
            "in_cf_item_universe": cold_item in cf_items,
            "similar_item": similar,
        }
        for user_id in sample:
            started = time.perf_counter()
            recommend_hybrid_for_user(session, user_id, top_k=args.top_k)
            latencies.append((time.perf_counter() - started) * 1000.0)
        warm_titles = _titles(
            session,
            [row["product_id"] for row in payload["warm"]["hybrid"]["top"]]
            + [row["product_id"] for row in payload["warm"]["content"]["top"]]
            + [row["product_id"] for row in payload["warm"]["cf"]["top"]],
        )
        payload["titles"] = warm_titles
    latencies.sort()
    payload["latency_ms"] = {
        "n": len(latencies),
        "median": statistics.median(latencies) if latencies else None,
        "p95": latencies[max(0, int(round(0.95 * (len(latencies) - 1))))] if latencies else None,
        "top_k": args.top_k,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
