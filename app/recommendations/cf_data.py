"""Load recsys-eval-v1 and build unique implicit CF training pairs.

Does not reconstruct the 2-core. The frozen recsys-eval-v1 split artifact is authoritative.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.recommendations.cf_constants import (
    EXPECTED_EVAL_USERS,
    EXPECTED_HIDDEN_CHECKSUM,
    EXPECTED_TWO_CORE_PAIRS,
    PRODUCT_ID_DTYPE,
    USER_ID_DTYPE,
)
from app.recommendations.constants import EVALUATION_VERSION


class RecsysEvalIncompatibleError(RuntimeError):
    """The on-disk recsys-eval-v1 artifact does not match the frozen split checksum."""


@dataclass(frozen=True, slots=True)
class EvalUserRecord:
    user_id: str
    train_product_ids: tuple[str, ...]
    hidden_product_id: str


@dataclass(frozen=True, slots=True)
class RecsysEvalSplit:
    evaluation_version: str
    users: tuple[EvalUserRecord, ...]
    two_core_pairs: int
    evaluation_users: int
    hidden_checksum: str
    catalog_size: int | None
    directory: Path | None = None


@dataclass(frozen=True, slots=True)
class CfMappings:
    user_ids: tuple[str, ...]
    product_ids: tuple[str, ...]
    user_to_index: dict[str, int]
    product_to_index: dict[str, int]

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.product_ids)


def hidden_pairs_checksum(pairs: Sequence[tuple[str, str]]) -> str:
    lines = [f"{user_id}\t{product_id}" for user_id, product_id in sorted((str(u), str(p)) for u, p in pairs)]
    payload = "\n".join(lines) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unique_implicit_pairs(events: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Collapse duplicate (user, product) events to one binary positive."""

    return sorted({(str(user_id), str(product_id)) for user_id, product_id in events})


def build_mappings(pairs: Sequence[tuple[str, str]]) -> CfMappings:
    user_ids = tuple(sorted({user_id for user_id, _product in pairs}))
    product_ids = tuple(sorted({product_id for _user, product_id in pairs}))
    return CfMappings(
        user_ids=user_ids,
        product_ids=product_ids,
        user_to_index={user_id: index for index, user_id in enumerate(user_ids)},
        product_to_index={product_id: index for index, product_id in enumerate(product_ids)},
    )


def pairs_to_index_arrays(
    pairs: Sequence[tuple[str, str]],
    mappings: CfMappings,
) -> tuple[np.ndarray, np.ndarray]:
    users = np.fromiter((mappings.user_to_index[user_id] for user_id, _p in pairs), dtype=np.int64, count=len(pairs))
    items = np.fromiter(
        (mappings.product_to_index[product_id] for _u, product_id in pairs),
        dtype=np.int64,
        count=len(pairs),
    )
    return users, items


def user_positive_sets(pairs: Sequence[tuple[str, str]], mappings: CfMappings) -> list[set[int]]:
    positives: list[set[int]] = [set() for _ in range(mappings.n_users)]
    for user_id, product_id in pairs:
        positives[mappings.user_to_index[user_id]].add(mappings.product_to_index[product_id])
    return positives


def encode_id_array(values: Sequence[str], dtype: str) -> np.ndarray:
    return np.asarray(list(values), dtype=dtype)


def mapping_checksum(user_ids: Sequence[str], product_ids: Sequence[str]) -> tuple[str, str]:
    user_payload = "\n".join(user_ids) + "\n"
    item_payload = "\n".join(product_ids) + "\n"
    return (
        hashlib.sha256(user_payload.encode("utf-8")).hexdigest(),
        hashlib.sha256(item_payload.encode("utf-8")).hexdigest(),
    )


def _records_from_frame(frame: pd.DataFrame) -> list[EvalUserRecord]:
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]
    train_by_user: dict[str, list[str]] = {}
    for user_id, group in train.groupby("user_id", sort=False):
        train_by_user[str(user_id)] = [str(product_id) for product_id in group["product_id"].tolist()]
    hidden_by_user = {
        str(user_id): str(product_id) for user_id, product_id in zip(test["user_id"], test["product_id"], strict=False)
    }
    users = sorted(hidden_by_user)
    records: list[EvalUserRecord] = []
    for user_id in users:
        train_ids = tuple(train_by_user.get(user_id, ()))
        hidden = hidden_by_user[user_id]
        if hidden in train_ids:
            raise RecsysEvalIncompatibleError(f"hidden product leaked into train history for {user_id}")
        records.append(
            EvalUserRecord(user_id=user_id, train_product_ids=train_ids, hidden_product_id=hidden)
        )
    return records


def load_recsys_eval_split(
    directory: Path,
    *,
    require_frozen_identity: bool = True,
) -> RecsysEvalSplit:
    directory = Path(directory)
    split_path = directory / "split.parquet"
    manifest_path = directory / "manifest.json"
    if not split_path.is_file() or not manifest_path.is_file():
        raise RecsysEvalIncompatibleError(
            f"missing recsys-eval-v1 artifact under {directory}. "
            "Do not invent a replacement population."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = str(manifest.get("evaluation_version", ""))
    if version != EVALUATION_VERSION:
        raise RecsysEvalIncompatibleError(f"evaluation_version {version!r} != {EVALUATION_VERSION!r}")
    frame = pd.read_parquet(split_path)
    required = {"user_id", "product_id", "split"}
    if not required.issubset(frame.columns):
        raise RecsysEvalIncompatibleError(f"split.parquet missing columns {required}")
    records = _records_from_frame(frame)
    hidden = [(row.user_id, row.hidden_product_id) for row in records]
    checksum = hidden_pairs_checksum(hidden)
    two_core_pairs = int(manifest.get("two_core_pairs", 0))
    evaluation_users = int(manifest.get("evaluation_users", len(records)))
    catalog_size = None
    metrics_path = directory / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        catalog_size = int(metrics["catalog_size"]) if "catalog_size" in metrics else None
    if require_frozen_identity:
        if evaluation_users != EXPECTED_EVAL_USERS or len(records) != EXPECTED_EVAL_USERS:
            raise RecsysEvalIncompatibleError(
                f"evaluation users {evaluation_users}/{len(records)} != {EXPECTED_EVAL_USERS}. Stop."
            )
        if two_core_pairs != EXPECTED_TWO_CORE_PAIRS:
            raise RecsysEvalIncompatibleError(
                f"two_core_pairs {two_core_pairs} != {EXPECTED_TWO_CORE_PAIRS}. Stop."
            )
        if checksum != EXPECTED_HIDDEN_CHECKSUM:
            raise RecsysEvalIncompatibleError(
                "hidden-item checksum does not match the frozen recsys-eval-v1 artifact. Stop."
            )
    return RecsysEvalSplit(
        evaluation_version=EVALUATION_VERSION,
        users=tuple(records),
        two_core_pairs=two_core_pairs,
        evaluation_users=len(records),
        hidden_checksum=checksum,
        catalog_size=catalog_size,
        directory=directory,
    )


def external_train_pairs(split: RecsysEvalSplit) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for user in split.users:
        for product_id in user.train_product_ids:
            pairs.append((user.user_id, product_id))
        if user.hidden_product_id in user.train_product_ids:
            raise AssertionError("external test product in train pairs")
    unique = unique_implicit_pairs(pairs)
    hidden = {(user.user_id, user.hidden_product_id) for user in split.users}
    if hidden & set(unique):
        raise AssertionError("external test pairs leaked into unique train pairs")
    return unique


def item_train_degrees(pairs: Sequence[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _user_id, product_id in pairs:
        counts[product_id] = counts.get(product_id, 0) + 1
    return counts


def assert_no_zero_train_item(mappings: CfMappings, pairs: Sequence[tuple[str, str]]) -> None:
    seen = {product_id for _user, product_id in pairs}
    missing = [product_id for product_id in mappings.product_ids if product_id not in seen]
    if missing:
        raise AssertionError(f"{len(missing)} mapped items have zero train positives")
