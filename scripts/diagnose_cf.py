#!/usr/bin/env python3
"""CF serving diagnostics: latency, titles, and a user outside the model.

Not an online A/B test. Uses the trained bpr-mf-v1 artifact and PostgreSQL.
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

from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.models.product import Product
from app.models.user import User
from app.recommendations.cf_artifacts import load_cf_bundle
from app.recommendations.cf_constants import MODEL_VERSION, REASON_CF_USER_UNAVAILABLE
from app.recommendations.cf_data import load_recsys_eval_split
from app.recommendations.constants import EVALUATION_VERSION
from app.recommendations.service import recommend_cf_for_user


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def _titles(session, product_ids: list[str]) -> dict[str, str]:
    if not product_ids:
        return {}
    rows = session.execute(select(Product.product_id, Product.title).where(Product.product_id.in_(product_ids))).all()
    return {str(product_id): str(title) for product_id, title in rows}


def main() -> int:
    args = parse_args()
    settings = get_settings()
    model_dir = Path(args.model_dir or Path(settings.artifacts_root) / "models" / MODEL_VERSION)
    split_dir = Path(args.split_dir or Path(settings.artifacts_root) / "evaluation" / EVALUATION_VERSION)
    _model, user_ids, _items, _config, _manifest = load_cf_bundle(model_dir)
    split = load_recsys_eval_split(split_dir, require_frozen_identity=True)
    sample = [user.user_id for user in split.users[: args.n_users]]
    factory = get_session_factory()
    latencies: list[float] = []
    examples: list[dict] = []
    outside_user = None
    with factory() as session:
        for user_id in session.scalars(select(User.user_id).limit(5000)):
            if str(user_id) not in set(user_ids[:1000]) and str(user_id) not in sample:
                if str(user_id) not in set(user_ids):
                    outside_user = str(user_id)
                    break
        for user_id in sample:
            started = time.perf_counter()
            response = recommend_cf_for_user(session, user_id, top_k=args.top_k)
            latencies.append(time.perf_counter() - started)
            if len(examples) < 4:
                eval_user = next(user for user in split.users if user.user_id == user_id)
                ids = [item.product_id for item in response.results]
                wanted = list(eval_user.train_product_ids[:4]) + [eval_user.hidden_product_id] + ids[:5]
                titles = _titles(session, wanted)
                examples.append(
                    {
                        "user_id_suffix": user_id[-6:],
                        "train_titles": [titles.get(pid, pid) for pid in eval_user.train_product_ids[:4]],
                        "hidden_title": titles.get(eval_user.hidden_product_id, eval_user.hidden_product_id),
                        "cf_titles": [titles.get(pid, pid) for pid in ids[:5]],
                        "hidden_in_results": eval_user.hidden_product_id in ids,
                    }
                )
        outside_body = None
        if outside_user:
            outside_body = recommend_cf_for_user(session, outside_user, top_k=args.top_k)
    latencies_ms = [value * 1000.0 for value in latencies]
    payload = {
        "n_users": len(latencies),
        "median_ms": round(statistics.median(latencies_ms), 3) if latencies_ms else None,
        "p95_ms": round(sorted(latencies_ms)[max(0, int(0.95 * (len(latencies_ms) - 1)))], 3) if latencies_ms else None,
        "examples": examples,
        "outside_model_user_id": outside_user,
        "outside_model_reason": None if outside_body is None else outside_body.reason,
        "expected_outside_reason": REASON_CF_USER_UNAVAILABLE,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
