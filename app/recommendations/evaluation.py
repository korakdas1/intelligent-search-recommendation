"""Time-aware leave-last-product-out evaluation (recsys-eval-v1).

Label class is observed held-out interaction, not explicit preference.
Random interaction splits are forbidden.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.interaction import Interaction
from app.ranking.metrics import metric_bundle
from app.recommendations.constants import EVALUATION_VERSION


@dataclass(frozen=True, slots=True)
class EvalUser:
    user_id: str
    train_product_ids: tuple[str, ...]
    hidden_product_id: str
    hidden_occurred_at: datetime


def iterative_two_core(pairs: Sequence[tuple[str, str]]) -> set[tuple[str, str]]:
    remaining = {(str(user_id), str(product_id)) for user_id, product_id in pairs}
    while True:
        user_counts: Counter[str] = Counter()
        item_counts: Counter[str] = Counter()
        for user_id, product_id in remaining:
            user_counts[user_id] += 1
            item_counts[product_id] += 1
        drop_users = {user_id for user_id, count in user_counts.items() if count < 2}
        drop_items = {product_id for product_id, count in item_counts.items() if count < 2}
        if not drop_users and not drop_items:
            return remaining
        remaining = {
            pair
            for pair in remaining
            if pair[0] not in drop_users and pair[1] not in drop_items
        }


def leave_last_product_split(users: dict[str, list[tuple[str, datetime]]]) -> list[EvalUser]:
    """Hold out the distinct product with the latest last-occurrence time."""

    out: list[EvalUser] = []
    for user_id, items in users.items():
        unique: dict[str, datetime] = {}
        for product_id, occurred_at in items:
            previous = unique.get(product_id)
            if previous is None or occurred_at >= previous:
                unique[product_id] = occurred_at
        if len(unique) < 2:
            continue
        ordered = sorted(unique.items(), key=lambda item: (item[1], item[0]))
        hidden_id, hidden_at = ordered[-1]
        train = tuple(product_id for product_id, _when in ordered[:-1])
        out.append(
            EvalUser(
                user_id=user_id,
                train_product_ids=train,
                hidden_product_id=hidden_id,
                hidden_occurred_at=hidden_at,
            )
        )
    out.sort(key=lambda row: row.user_id)
    return out


def load_user_product_last_times(session: Session) -> dict[str, list[tuple[str, datetime]]]:
    stmt = select(
        Interaction.user_id,
        Interaction.product_id,
        Interaction.occurred_at,
    )
    grouped: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
    for user_id, product_id, occurred_at in session.execute(stmt):
        grouped[str(user_id)].append((str(product_id), occurred_at))
    return grouped


def build_eval_users(
    session: Session,
    *,
    use_two_core: bool = True,
) -> tuple[list[EvalUser], dict[str, int]]:
    grouped = load_user_product_last_times(session)
    unique_pairs = {
        (user_id, product_id)
        for user_id, items in grouped.items()
        for product_id, _when in items
    }
    stats = {
        "total_users": len(grouped),
        "users_with_ge2_distinct_products": sum(
            1
            for items in grouped.values()
            if len({product_id for product_id, _ in items}) >= 2
        ),
        "unique_user_product_pairs": len(unique_pairs),
    }
    if use_two_core:
        core = iterative_two_core(list(unique_pairs))
        core_users = {user_id for user_id, _product in core}
        core_grouped = {
            user_id: [
                (product_id, when)
                for product_id, when in items
                if (user_id, product_id) in core
            ]
            for user_id, items in grouped.items()
            if user_id in core_users
        }
        stats["two_core_users"] = len(core_users)
        stats["two_core_pairs"] = len(core)
        eval_users = leave_last_product_split(core_grouped)
    else:
        eval_users = leave_last_product_split(grouped)
    stats["evaluation_users"] = len(eval_users)
    return eval_users, stats


def train_interaction_mask(
    session: Session,
    eval_users: Sequence[EvalUser],
) -> list[int]:
    """Interaction ids allowed for train-only popularity (exclude hidden pairs)."""

    hidden = {(row.user_id, row.hidden_product_id) for row in eval_users}
    stmt = select(
        Interaction.interaction_id,
        Interaction.user_id,
        Interaction.product_id,
    )
    allowed: list[int] = []
    for interaction_id, user_id, product_id in session.execute(stmt):
        if (str(user_id), str(product_id)) in hidden:
            continue
        allowed.append(int(interaction_id))
    return allowed


def history_size_bin(n_train: int) -> str:
    if n_train <= 1:
        return "train_size=1"
    if n_train <= 4:
        return "train_size=2-4"
    return "train_size>=5"


def rank_to_metrics(rank: int | None, catalog_size: int) -> dict[str, float]:
    if rank is None or rank < 1 or rank > catalog_size:
        return metric_bundle([], ["HIDDEN"])
    ranked_ids = [f"d{index}" for index in range(1, rank)] + ["HIDDEN"]
    return metric_bundle(ranked_ids, ["HIDDEN"])
