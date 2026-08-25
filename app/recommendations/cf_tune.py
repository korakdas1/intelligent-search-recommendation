"""Inner validation protocol cf-tune-v1. Does not touch recsys-eval-v1 external test."""

from __future__ import annotations

from dataclasses import dataclass

from app.recommendations.cf_data import (
    EvalUserRecord,
    RecsysEvalSplit,
    unique_implicit_pairs,
)


@dataclass(frozen=True, slots=True)
class TuneValUser:
    user_id: str
    inner_train_product_ids: tuple[str, ...]
    hidden_product_id: str


@dataclass(frozen=True, slots=True)
class CfTuneSplit:
    protocol: str
    inner_train_pairs: tuple[tuple[str, str], ...]
    validation_users: tuple[TuneValUser, ...]
    inner_train_only_users: int


def build_cf_tune_split(split: RecsysEvalSplit, *, protocol: str = "cf-tune-v1") -> CfTuneSplit:
    """Hold out the latest remaining TRAIN product when the user has ≥2 train items."""

    inner_events: list[tuple[str, str]] = []
    validation: list[TuneValUser] = []
    train_only = 0
    external_hidden = {(user.user_id, user.hidden_product_id) for user in split.users}
    for user in split.users:
        train = user.train_product_ids
        if len(train) >= 2:
            hidden = train[-1]
            inner_train = train[:-1]
            if hidden in inner_train:
                raise AssertionError("inner validation item leaked into inner train")
            if hidden == user.hidden_product_id:
                raise AssertionError("inner validation used the external test product")
            inner_events.extend((user.user_id, product_id) for product_id in inner_train)
            validation.append(
                TuneValUser(
                    user_id=user.user_id,
                    inner_train_product_ids=inner_train,
                    hidden_product_id=hidden,
                )
            )
        else:
            train_only += 1
            inner_events.extend((user.user_id, product_id) for product_id in train)
    unique = tuple(unique_implicit_pairs(inner_events))
    if external_hidden & set(unique):
        raise AssertionError("external test pairs leaked into cf-tune-v1 inner train")
    for row in validation:
        if (row.user_id, row.hidden_product_id) in set(unique):
            raise AssertionError("inner validation pair still in inner train")
    return CfTuneSplit(
        protocol=protocol,
        inner_train_pairs=unique,
        validation_users=tuple(validation),
        inner_train_only_users=train_only,
    )


def assert_tune_leakage_free(split: RecsysEvalSplit, tune: CfTuneSplit) -> None:
    inner = set(tune.inner_train_pairs)
    for user in split.users:
        if (user.user_id, user.hidden_product_id) in inner:
            raise AssertionError("external test leaked into inner train")
    for row in tune.validation_users:
        if (row.user_id, row.hidden_product_id) in inner:
            raise AssertionError("inner validation leaked into inner train")
        if row.hidden_product_id in row.inner_train_product_ids:
            raise AssertionError("inner validation item listed in inner-train history")


def from_eval_users(users: list[EvalUserRecord], *, protocol: str = "cf-tune-v1") -> CfTuneSplit:
    split = RecsysEvalSplit(
        evaluation_version="fixture",
        users=tuple(users),
        two_core_pairs=sum(1 + len(user.train_product_ids) for user in users),
        evaluation_users=len(users),
        hidden_checksum="",
        catalog_size=None,
    )
    return build_cf_tune_split(split, protocol=protocol)
