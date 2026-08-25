#!/usr/bin/env python3
"""Profile personalization-feature availability. Does not train models."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.models.product import Product
from app.recommendations.cf_data import RecsysEvalIncompatibleError, load_recsys_eval_split
from app.recommendations.constants import EVALUATION_VERSION
from app.recommendations.hybrid_offline import load_catalog_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-version", default=EVALUATION_VERSION)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    split_dir = Path(args.split_dir or Path(settings.artifacts_root) / "evaluation" / args.evaluation_version)
    try:
        split = load_recsys_eval_split(split_dir, require_frozen_identity=True)
    except RecsysEvalIncompatibleError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    users = list(split.users)
    if args.max_users and args.max_users < len(users):
        rng = __import__("numpy").random.default_rng(args.seed)
        order = rng.permutation(len(users))[: args.max_users]
        users = [users[int(index)] for index in sorted(order.tolist())]

    _ids, _emb, id_to_row = load_catalog_embeddings(settings)
    try:
        from app.recommendations.cf_artifacts import load_cf_bundle
        from app.recommendations.cf_probe import cf_artifact_path

        _model, cf_users, cf_items, _config, _manifest = load_cf_bundle(cf_artifact_path(settings))
        cf_user_set = set(cf_users)
        cf_item_set = set(cf_items)
        cf_artifact = True
    except Exception:
        cf_user_set = set()
        cf_item_set = set()
        cf_artifact = False

    factory = get_session_factory()
    with factory() as session:
        products = {
            str(row.product_id): row
            for row in session.scalars(select(Product)).all()
        }

    categories = Counter()
    brand_history_users = 0
    price_history_users = 0
    content_users = 0
    cf_users_n = 0
    both = 0
    neither = 0
    brand_nonzero = 0
    for user in users:
        history = list(user.train_product_ids)
        has_content = any(pid in id_to_row for pid in history)
        has_cf = cf_artifact and user.user_id in cf_user_set
        content_users += int(has_content)
        cf_users_n += int(has_cf)
        both += int(has_content and has_cf)
        neither += int(not has_content and not has_cf)
        brands = [products[pid].brand for pid in history if pid in products and products[pid].brand]
        if brands:
            brand_history_users += 1
            brand_nonzero += 1
        prices = [products[pid].price for pid in history if pid in products and products[pid].price is not None]
        if len(prices) >= 2:
            price_history_users += 1
        for pid in history:
            product = products.get(pid)
            if product is not None:
                categories[str(product.category or "")] += 1

    catalog_n = len(products) or 1
    brand_catalog = sum(1 for product in products.values() if product.brand)
    price_catalog = sum(1 for product in products.values() if product.price is not None)
    category_mode, category_mode_n = categories.most_common(1)[0] if categories else ("", 0)
    payload = {
        "evaluation_version": args.evaluation_version,
        "sampled_users": len(users),
        "seed": args.seed,
        "content_signal_users": content_users,
        "cf_artifact_available": cf_artifact,
        "cf_signal_users": cf_users_n,
        "both_signals": both,
        "neither_signal": neither,
        "content_coverage": content_users / len(users),
        "cf_coverage": cf_users_n / len(users),
        "brand_catalog_coverage": brand_catalog / catalog_n,
        "price_catalog_coverage": price_catalog / catalog_n,
        "brand_history_users": brand_history_users,
        "price_history_users_with_2plus_prices": price_history_users,
        "category_mode": category_mode,
        "category_mode_share_of_history": category_mode_n / max(1, sum(categories.values())),
        "cf_item_universe": len(cf_item_set),
        "decision": {
            "content": "usable",
            "cf": "usable_where_available" if cf_artifact else "artifact_unavailable",
            "brand": "diagnostic_only_incomplete_coverage",
            "category": "unusable_near_constant",
            "price": "deferred_high_missingness",
            "recency": "diagnostic_only",
        },
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
