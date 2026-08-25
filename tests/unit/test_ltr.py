"""Synthetic LTR labels, splits, scaler, RankNet, and metrics. No catalog. No MiniLM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app.ranking.artifacts import load_ranker_bundle, save_ranker_bundle
from app.ranking.feature_schema import FEATURE_COUNT, FEATURE_NAMES, FEATURE_VERSION
from app.ranking.inference import score_feature_matrix
from app.ranking.labels import make_attribute_query, make_title_query
from app.ranking.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from app.ranking.pairwise import ranknet_loss, sample_hard_negative_pairs
from app.ranking.preprocessing import fit_scaler, load_scaler, save_scaler
from app.ranking.ranker import RankNetMLP
from app.ranking.split import SplitLeakageError, assert_split_disjoint, grouped_source_split
from app.ranking.train import train_ranknet


def test_title_query_howard_leather_conditioner() -> None:
    spec = make_title_query("B09KKXB1SZ", "Howard LC0008 Leather Conditioner 8 Ounce")
    assert spec is not None
    assert spec.query == "leather conditioner"
    assert spec.query_source == "synthetic_title"
    assert spec.label_class == "synthetic"
    assert spec.relevant_ids == ("B09KKXB1SZ",)
    assert spec.source_product_id == "B09KKXB1SZ"


def test_title_query_skips_thin_titles() -> None:
    assert make_title_query("X", "Pack") is None
    assert make_title_query("X", "8 Ounce") is None


def test_attribute_query_requires_brand() -> None:
    assert make_attribute_query("X", "Howard Leather Conditioner", None) is None
    spec = make_attribute_query("X", "Howard LC0008 Leather Conditioner", "Howard Products")
    assert spec is not None
    assert spec.query == "howard conditioner"
    assert spec.query_source == "synthetic_attribute"


def test_ltr_query_id_is_stable() -> None:
    first = make_title_query("P1", "Alpha Leather Balm Extra")
    second = make_title_query("P1", "Alpha Leather Balm Extra")
    assert first is not None and second is not None
    assert first.query_id == second.query_id
    other = make_title_query("P2", "Alpha Leather Balm Extra")
    assert other is not None
    assert other.query_id != first.query_id


def test_grouped_split_no_source_overlap() -> None:
    mapping = grouped_source_split([f"s{i}" for i in range(20)], seed=42)
    by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for source_id, split in mapping.items():
        by_split[split].add(source_id)
    assert_split_disjoint(by_split)
    assert by_split["train"] and by_split["validation"] and by_split["test"]
    with pytest.raises(SplitLeakageError):
        assert_split_disjoint({"train": {"a"}, "validation": {"a"}, "test": {"b"}})


def test_scaler_fits_train_only(tmp_path: Path) -> None:
    train = np.zeros((4, FEATURE_COUNT), dtype=np.float32)
    train[:, 0] = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    val = np.zeros((2, FEATURE_COUNT), dtype=np.float32)
    val[:, 0] = np.array([1000.0, 2000.0], dtype=np.float32)
    scaler = fit_scaler(train)
    assert scaler.mean[0] == pytest.approx(4.0)
    assert 0 not in scaler.zero_variance_indices
    assert 1 in scaler.zero_variance_indices
    scaled_val = scaler.transform(val)
    assert scaled_val[0, 0] == pytest.approx((1000.0 - 4.0) / scaler.std[0])
    path = tmp_path / "scaler.npz"
    save_scaler(path, scaler)
    loaded = load_scaler(path)
    assert np.allclose(loaded.mean, scaler.mean)


def test_ranknet_loss_prefers_correct_order() -> None:
    good = ranknet_loss(torch.tensor([2.0]), torch.tensor([0.0]))
    bad = ranknet_loss(torch.tensor([0.0]), torch.tensor([2.0]))
    assert float(good) < float(bad)
    expected = float(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([2.0]), torch.tensor([1.0])))
    assert float(good) == pytest.approx(expected)


def test_model_shape_and_eval_deterministic() -> None:
    torch.manual_seed(0)
    model = RankNetMLP(hidden_sizes=(8, 4), dropout=0.9)
    batch = torch.randn(4, FEATURE_COUNT)
    out = model(batch)
    assert tuple(out.shape) == (4,)
    model.eval()
    with torch.no_grad():
        first = model(batch).clone()
        second = model(batch).clone()
    assert torch.allclose(first, second)


def test_model_rejects_nan() -> None:
    model = RankNetMLP(hidden_sizes=(8,), dropout=0.0)
    model.eval()
    bad = torch.full((2, FEATURE_COUNT), float("nan"))
    with pytest.raises(ValueError):
        model(bad)


def test_hard_negative_pairs_skip_positive_positive() -> None:
    frame = pd.DataFrame(
        {
            "relevance": [1, 1, 0, 0],
            "candidate_position": [1, 2, 3, 4],
        }
    )
    pairs = sample_hard_negative_pairs(frame, negatives_per_positive=2)
    assert pairs == [(0, 2), (0, 3), (1, 2), (1, 3)]


def test_metrics_hand_checked() -> None:
    ranked = ["A", "B", "C"]
    assert mean_reciprocal_rank(ranked, ["A"]) == 1.0
    assert recall_at_k(ranked, ["Z"], 2) == 0.0
    assert ndcg_at_k(["A", "B"], ["A"], 2) == pytest.approx(1.0)


def test_artifact_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(20, FEATURE_COUNT)).astype(np.float32)
    scaler = fit_scaler(train)
    model = RankNetMLP(hidden_sizes=(8,), dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        list(model.parameters())[-1].fill_(0.1)
    config = {
        "model_version": "ranknet-test",
        "feature_version": FEATURE_VERSION,
        "input_dim": FEATURE_COUNT,
        "hidden_sizes": [8],
        "dropout": 0.0,
    }
    save_ranker_bundle(
        tmp_path,
        model=model,
        scaler=scaler,
        config=config,
        manifest={"model_version": "ranknet-test"},
    )
    loaded, loaded_scaler, loaded_config = load_ranker_bundle(tmp_path)
    sample = train[:3]
    original = score_feature_matrix(model, scaler, sample)
    restored = score_feature_matrix(loaded, loaded_scaler, sample)
    assert np.allclose(original, restored, atol=1e-5)
    assert loaded_config["hidden_sizes"] == [8]


def test_tiny_training_loss_drops() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for query_i in range(12):
        split = "train" if query_i < 8 else "validation"
        pos = rng.normal(loc=1.0, scale=0.1, size=FEATURE_COUNT).astype(np.float32)
        neg = rng.normal(loc=-1.0, scale=0.1, size=FEATURE_COUNT).astype(np.float32)
        for product_id, relevance, vec in (("P", 1, pos), ("N", 0, neg)):
            row = {
                "query_id": f"q{query_i}",
                "split": split,
                "product_id": product_id,
                "relevance": relevance,
                "candidate_position": 1 if relevance else 2,
            }
            for index, name in enumerate(FEATURE_NAMES):
                row[name] = vec[index]
            rows.append(row)
    frame = pd.DataFrame(rows)
    scaler = fit_scaler(frame.loc[frame["split"] == "train", list(FEATURE_NAMES)].to_numpy())
    model, summary = train_ranknet(
        frame,
        scaler,
        hidden_sizes=(16,),
        dropout=0.0,
        learning_rate=0.05,
        batch_size=8,
        max_epochs=8,
        patience=8,
        seed=0,
        device="cpu",
        negatives_per_positive=1,
    )
    assert summary["epochs_run"] >= 1
    assert summary["history"][-1]["train_loss"] < summary["history"][0]["train_loss"]
    assert model.parameter_count() > 0
