"""BPR-MF unit tests. Tiny synthetic fixtures only. No full catalog, no MiniLM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app.recommendations.cf_artifacts import CfArtifactError, load_cf_bundle, save_cf_bundle
from app.recommendations.cf_data import (
    EvalUserRecord,
    RecsysEvalSplit,
    build_mappings,
    external_train_pairs,
    hidden_pairs_checksum,
    unique_implicit_pairs,
    user_positive_sets,
)
from app.recommendations.cf_loss import bpr_ranking_loss
from app.recommendations.cf_model import BPRMatrixFactorization
from app.recommendations.cf_sampling import UniformNegativeSampler, assert_triple_valid
from app.recommendations.cf_train import rank_hidden_from_scores, train_bpr_mf
from app.recommendations.cf_tune import assert_tune_leakage_free, build_cf_tune_split
from app.recommendations.evaluation import rank_to_metrics
from app.recommendations.service import validate_user_method


def _split(users: list[EvalUserRecord]) -> RecsysEvalSplit:
    return RecsysEvalSplit(
        evaluation_version="fixture",
        users=tuple(users),
        two_core_pairs=sum(1 + len(user.train_product_ids) for user in users),
        evaluation_users=len(users),
        hidden_checksum=hidden_pairs_checksum([(user.user_id, user.hidden_product_id) for user in users]),
        catalog_size=10,
    )


def test_dot_product_score_exact() -> None:
    model = BPRMatrixFactorization(2, 3, 2)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1.0, 2.0], [0.0, 1.0]]))
        model.item_embedding.weight.copy_(torch.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 1.0]]))
        score = model.score(torch.tensor([0]), torch.tensor([0]))
    assert float(score.item()) == pytest.approx(11.0)


def test_bpr_loss_prefers_correct_order() -> None:
    good = bpr_ranking_loss(torch.tensor([2.0]), torch.tensor([0.0]))
    bad = bpr_ranking_loss(torch.tensor([0.0]), torch.tensor([2.0]))
    assert float(good) < float(bad)
    expected = float(torch.nn.functional.softplus(torch.tensor(-2.0)))
    assert float(good) == pytest.approx(expected)


def test_unique_pair_collapse() -> None:
    pairs = unique_implicit_pairs(
        [("u1", "p1"), ("u1", "p1"), ("u1", "p2"), ("u2", "p1"), ("u1", "p1")]
    )
    assert pairs == [("u1", "p1"), ("u1", "p2"), ("u2", "p1")]


def test_external_test_not_in_train_pairs() -> None:
    split = _split(
        [
            EvalUserRecord("u1", ("p1", "p2"), "p3"),
            EvalUserRecord("u2", ("p1",), "p2"),
        ]
    )
    pairs = external_train_pairs(split)
    assert ("u1", "p3") not in pairs
    assert ("u2", "p2") not in pairs
    assert ("u1", "p1") in pairs


def test_inner_validation_leakage() -> None:
    split = _split(
        [
            EvalUserRecord("u1", ("p1", "p2", "p3"), "p9"),
            EvalUserRecord("u2", ("p1",), "p8"),
        ]
    )
    tune = build_cf_tune_split(split)
    assert_tune_leakage_free(split, tune)
    assert tune.validation_users[0].hidden_product_id == "p3"
    assert tune.validation_users[0].inner_train_product_ids == ("p1", "p2")
    assert ("u1", "p3") not in set(tune.inner_train_pairs)
    assert ("u1", "p9") not in set(tune.inner_train_pairs)
    assert ("u2", "p1") in set(tune.inner_train_pairs)
    assert tune.inner_train_only_users == 1


def test_split_identity_checksum_changes_when_hidden_changes(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"user_id": "u1", "product_id": "p1", "split": "train"},
            {"user_id": "u1", "product_id": "p2", "split": "test"},
            {"user_id": "u2", "product_id": "p1", "split": "train"},
            {"user_id": "u2", "product_id": "p3", "split": "test"},
        ]
    )
    original = hidden_pairs_checksum([("u1", "p2"), ("u2", "p3")])
    mutated = hidden_pairs_checksum([("u1", "p9"), ("u2", "p3")])
    assert original != mutated
    path = tmp_path / "split.parquet"
    frame.to_parquet(path, index=False)
    loaded = pd.read_parquet(path)
    test = loaded[loaded["split"] == "test"]
    checksum = hidden_pairs_checksum(list(zip(test["user_id"].astype(str), test["product_id"].astype(str))))
    assert checksum == original


def test_zero_train_items_have_no_index() -> None:
    mappings = build_mappings([("u1", "p1"), ("u1", "p2"), ("u2", "p1")])
    assert "p3" not in mappings.product_to_index
    assert set(mappings.product_ids) == {"p1", "p2"}


def test_cold_hidden_item_is_a_miss() -> None:
    metrics = rank_to_metrics(None, 100)
    assert metrics["recall@10"] == 0.0
    assert metrics["ndcg@10"] == 0.0
    assert metrics["mrr"] == 0.0


def test_rank_hidden_excludes_seen() -> None:
    scores = np.array([0.9, 0.8, 0.1], dtype=np.float32)
    product_ids = np.array(["a", "b", "c"])
    rank = rank_hidden_from_scores(scores, product_ids, seen_indices=[0], hidden_index=2)
    # a is seen; b beats c; rank of c is 2
    assert rank == 2


def test_negative_sampler_excludes_positives() -> None:
    positives = [{0, 1}, {2}]
    sampler = UniformNegativeSampler(positives, n_items=5)
    users = np.array([0, 0, 0, 1], dtype=np.int64)
    pos = np.array([0, 1, 0, 2], dtype=np.int64)
    for epoch in range(7):
        negatives = sampler.sample(users, pos, seed=42, epoch=epoch)
        for user_index, positive_index, negative_index in zip(users, pos, negatives, strict=True):
            assert_triple_valid(int(user_index), int(positive_index), int(negative_index), positives)
            if user_index == 0:
                assert int(negative_index) not in {0, 1}


def test_tiny_bpr_positive_outranks_negative() -> None:
    user_index = np.array([0, 0, 1, 1], dtype=np.int64)
    item_index = np.array([0, 1, 0, 2], dtype=np.int64)
    mappings_n_users, mappings_n_items = 2, 3
    positives = [{0, 1}, {0, 2}]
    model, summary = train_bpr_mf(
        n_users=mappings_n_users,
        n_items=mappings_n_items,
        embedding_dim=8,
        train_user_index=user_index,
        train_item_index=item_index,
        user_positives=positives,
        product_ids=["p0", "p1", "p2"],
        learning_rate=0.05,
        batch_size=4,
        l2=0.0,
        max_epochs=40,
        patience=40,
        seed=0,
        device="cpu",
        fixed_epochs=25,
    )
    model.eval()
    with torch.no_grad():
        pos = float(model.score(torch.tensor([0]), torch.tensor([0])).item())
        neg = float(model.score(torch.tensor([0]), torch.tensor([2])).item())
    assert pos > neg
    assert summary["history"][-1]["train_loss"] < summary["history"][0]["train_loss"]


def test_artifact_roundtrip(tmp_path: Path) -> None:
    model = BPRMatrixFactorization(2, 3, 4)
    with torch.no_grad():
        model.user_embedding.weight.fill_(0.2)
        model.item_embedding.weight.fill_(0.3)
    config = {
        "model_version": "bpr-mf-test",
        "model_type": "bpr_mf",
        "embedding_dim": 4,
        "n_users": 2,
        "n_items": 3,
    }
    users = ["u1", "u2"]
    items = ["p1", "p2", "p3"]
    from app.recommendations.cf_data import mapping_checksum

    user_c, item_c = mapping_checksum(users, items)
    save_cf_bundle(
        tmp_path,
        model=model,
        user_ids=users,
        product_ids=items,
        config=config,
        manifest={
            "model_version": "bpr-mf-test",
            "user_mapping_checksum": user_c,
            "item_mapping_checksum": item_c,
        },
    )
    loaded, user_ids, product_ids, _config, _manifest = load_cf_bundle(tmp_path)
    with torch.no_grad():
        original = model.score(torch.tensor([0, 1]), torch.tensor([1, 2]))
        restored = loaded.score(torch.tensor([0, 1]), torch.tensor([1, 2]))
    assert torch.allclose(original, restored)
    assert user_ids == users
    assert product_ids == items


def test_invalid_artifact_shape_fails(tmp_path: Path) -> None:
    model = BPRMatrixFactorization(2, 3, 4)
    config = {
        "model_version": "bpr-mf-test",
        "model_type": "bpr_mf",
        "embedding_dim": 4,
        "n_users": 2,
        "n_items": 3,
    }
    save_cf_bundle(
        tmp_path,
        model=model,
        user_ids=["u1", "u2"],
        product_ids=["p1"],
        config=config,
        manifest={"model_version": "bpr-mf-test"},
    )
    with pytest.raises(CfArtifactError):
        load_cf_bundle(tmp_path)


def test_user_method_allows_cf_and_hybrid() -> None:
    assert validate_user_method("content") == "content"
    assert validate_user_method("cf") == "cf"
    assert validate_user_method("CF") == "cf"
    assert validate_user_method("hybrid") == "hybrid"
    with pytest.raises(ValueError):
        validate_user_method("als")
