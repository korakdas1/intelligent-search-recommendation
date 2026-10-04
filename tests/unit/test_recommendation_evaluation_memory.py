"""Tiny retained-score reference and batch lifetime checks; no real artifacts or training."""

from __future__ import annotations

import json
import sys
import weakref
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from app.core.config import Settings
from app.recommendations.cf_data import EvalUserRecord, RecsysEvalSplit
from app.recommendations.cf_model import BPRMatrixFactorization
from app.recommendations.hybrid_policy import default_hybrid_policy
from scripts import evaluate_official_recommendations as evaluator


def _fixture(*, mixed: bool = True, fusion: str = "rrf") -> SimpleNamespace:
    ids = np.array(["P09", "P07", "P05", "P03", "P01", "P08", "P06", "P04", "P02", "P10", "ZERO", "NEG"])
    embeddings = np.array([[1, 0]] * 9 + [[0, 1], [0, 0], [-1, 0]], dtype=np.float32)
    cf_ids = ["P07", "P05", "P03", "P01", "P08", "ZERO", "NEG"]
    histories = [
        (("P01",), "P09"),
        (("P03", "P05"), "P07"),
        (("P01", "P03", "P05", "P07", "P08"), "P02"),
        (("UNKNOWN",), "P01"),
        ((), "OUTSIDE"),
        (("ZERO",), "P03"),
        (("P01", "NEG"), "P08"),
        (tuple(cf_ids), "P10"),
        (("ZERO", "UNKNOWN"), "OUTSIDE"),
        (("P02", "P04"), "P06"),
        (("P06", "P10", "P08", "P05", "P03"), "P04"),
    ] if mixed else [(("P01",), "P09") for _ in range(11)]
    users = tuple(EvalUserRecord(f"u{i:02d}", history, hidden) for i, (history, hidden) in enumerate(histories))
    # Deliberately reverse model user order to exercise the CF mapping.
    cf_users = [user.user_id for user in users[::-1]]
    model = BPRMatrixFactorization(len(users), len(cf_ids), 2)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1, i % 2] for i in range(len(users))]))
        model.item_embedding.weight.copy_(torch.tensor([[1, 0]] * 5 + [[0, 1], [-1, 0]]))
    policy = replace(default_hybrid_policy(), fusion_method=fusion, candidate_k_min=3, candidate_k_max=3)
    return SimpleNamespace(
        users=users, ids=ids, embeddings=embeddings, id_to_row={str(pid): i for i, pid in enumerate(ids)},
        cf_ids=cf_ids, cf_users=cf_users, model=model, policy=policy,
        pop_counts={**{str(pid): 3 for pid in ids}, "OUTSIDE": 12},
        split=RecsysEvalSplit("fixture", users, 42, len(users), "fixture-checksum", len(ids)),
    )


def _candidate_ids(items):
    return None if items is None else tuple(item.product_id for item in items)


def _retained_reference(fixture, users):
    """Old logical architecture: retain every content and CF score row, then rank."""

    cf_map = {pid: index for index, pid in enumerate(fixture.cf_ids)}
    user_map = {uid: index for index, uid in enumerate(fixture.cf_users)}
    cf_scores = evaluator.score_cf_users(fixture.model, [user_map[user.user_id] for user in users])
    profiles = []
    profile_users = []
    for index, user in enumerate(users):
        seen = np.array([fixture.id_to_row[pid] for pid in user.train_product_ids if pid in fixture.id_to_row], dtype=np.int64)
        profile = evaluator.mean_profile(fixture.embeddings, seen)
        if profile is not None:
            profiles.append(profile)
            profile_users.append(index)
    retained = {}
    if profiles:
        full_content = np.stack(profiles) @ fixture.embeddings.T
        retained = {user_index: full_content[row] for row, user_index in enumerate(profile_users)}
    pop_scores = np.array([fixture.pop_counts[str(pid)] for pid in fixture.ids], dtype=np.float32)
    ranked_pop = evaluator.popularity_ranking(fixture.pop_counts)
    degrees = evaluator.item_train_degrees(evaluator.external_train_pairs(fixture.split))
    records = []
    for index, user in enumerate(users):
        exclude = set(user.train_product_ids)
        seen = evaluator.exclude_row_indices(exclude, fixture.id_to_row)
        seen_cf = [cf_map[pid] for pid in user.train_product_ids if pid in cf_map]
        hidden = fixture.id_to_row.get(user.hidden_product_id)
        hidden_cf = cf_map.get(user.hidden_product_id)
        pop_rank = None if hidden is None else evaluator._rank_hidden(
            pop_scores, fixture.ids, seen_rows=seen, hidden_row=hidden,
        )
        content_rank = None if hidden is None or index not in retained else evaluator._rank_hidden(
            retained[index], fixture.ids, seen_rows=seen, hidden_row=hidden,
        )
        cf_rank = None if hidden_cf is None else evaluator.rank_hidden_from_scores(
            cf_scores[index], np.asarray(fixture.cf_ids), seen_indices=seen_cf, hidden_index=hidden_cf,
        )
        content = None if index not in retained else evaluator.content_candidates_from_scores(
            retained[index], fixture.ids, exclude=exclude, top_k=3, exclude_indices=seen,
        )
        cf = evaluator.cf_candidates_from_scores(
            cf_scores[index], fixture.cf_ids, exclude=exclude, top_k=3,
            exclude_indices=evaluator.exclude_row_indices(exclude, cf_map),
        )
        popularity = evaluator.popularity_candidates_from_counts(
            fixture.pop_counts, exclude=exclude, top_k=3, ranked=ranked_pop,
        )
        fused = evaluator.fuse_channel_lists(
            content, cf, popularity, fusion_method=fixture.policy.fusion_method,
            rrf_k=fixture.policy.rrf_k, weights=fixture.policy.weights,
        )
        ranks = [pop_rank, content_rank, cf_rank, evaluator.rank_hidden_in_fused(user.hidden_product_id, fused)]
        flags = evaluator.coverage_flags(user.hidden_product_id, content, cf, popularity)
        flags.update(cf_item_in_model=hidden_cf is not None, hidden_in_catalog=hidden is not None)
        records.append({
            "user_id": user.user_id, "ranks": ranks,
            "metrics": [evaluator.rank_to_metrics(rank, len(fixture.ids)) for rank in ranks],
            "candidates": tuple(_candidate_ids(items) for items in (content, cf, popularity, fused)),
            "flags": flags, "history": evaluator.history_size_bin(len(user.train_product_ids)),
            "degree": evaluator._degree_bin(degrees.get(user.hidden_product_id, 0)),
        })
    return records


def _run(tmp_path, monkeypatch, fixture, *, batch_size, max_users=None, seed=42):
    """Execute the actual CLI path with only artifact/database I/O replaced."""

    args = SimpleNamespace(
        batch_size=batch_size, max_users=max_users, seed=seed, evaluation_version="fixture",
        split_dir=str(tmp_path / "split"), model_dir=str(tmp_path / "model"), output=str(tmp_path), force=True,
    )
    monkeypatch.setattr(evaluator, "parse_args", lambda: args)
    monkeypatch.setattr(evaluator, "get_settings", lambda: Settings(_env_file=None, artifacts_root=str(tmp_path)))
    loader = Mock(return_value=fixture.split)
    monkeypatch.setattr(evaluator, "load_recsys_eval_split", loader)
    monkeypatch.setattr(evaluator, "load_hybrid_policy", lambda: fixture.policy)
    monkeypatch.setattr(evaluator, "load_cf_bundle", lambda path: (
        fixture.model, fixture.cf_users, fixture.cf_ids, {}, {"hidden_checksum": fixture.split.hidden_checksum},
    ))
    monkeypatch.setattr(evaluator, "load_catalog_embeddings", lambda settings: (
        fixture.ids, fixture.embeddings, fixture.id_to_row,
    ))
    monkeypatch.setattr(evaluator, "get_session_factory", lambda: lambda: nullcontext(object()))
    popularity = Mock(return_value=fixture.pop_counts)
    monkeypatch.setattr(evaluator, "interaction_counts_excluding", popularity)
    monkeypatch.setattr(evaluator, "_git_commit", lambda: "fixture")
    assert evaluator.main() == 0
    loader.assert_called_once_with(tmp_path / "split", require_frozen_identity=True)
    popularity.assert_called_once()
    assert popularity.call_args.args[1] == {(u.user_id, u.hidden_product_id) for u in fixture.users}
    result = json.loads((tmp_path / "recommendation_comparison.json").read_text())
    result.pop("created_at")
    result.pop("runtime_seconds")
    return result


def _trace(monkeypatch):
    trace = {"ranks": [], "metrics": [], "candidates": [], "flags": []}
    real_metrics = evaluator.rank_to_metrics
    real_fuse = evaluator.fuse_channel_lists
    real_flags = evaluator.coverage_flags

    def metrics(rank, catalog_size):
        result = real_metrics(rank, catalog_size)
        trace["ranks"].append(rank)
        trace["metrics"].append(result)
        return result

    def fuse(content, cf, popularity, **kwargs):
        result = real_fuse(content, cf, popularity, **kwargs)
        trace["candidates"].append(tuple(_candidate_ids(items) for items in (content, cf, popularity, result)))
        return result

    def flags(*args):
        result = real_flags(*args)
        trace["flags"].append(result)
        return result

    monkeypatch.setattr(evaluator, "rank_to_metrics", metrics)
    monkeypatch.setattr(evaluator, "fuse_channel_lists", fuse)
    monkeypatch.setattr(evaluator, "coverage_flags", flags)
    return trace


def _assert_reference(payload, trace, records):
    assert trace["ranks"] == [rank for row in records for rank in row["ranks"]]
    assert trace["metrics"] == [metric for row in records for metric in row["metrics"]]
    assert trace["candidates"] == [row["candidates"] for row in records]
    assert trace["flags"] == [row["flags"] for row in records]
    for index, system in enumerate(("popularity", "content", "cf", "hybrid")):
        assert payload["all_users"][system] == evaluator.macro_average([row["metrics"][index] for row in records])
        for field, source in (("by_history", "history"), ("by_hidden_train_degree", "degree")):
            groups = defaultdict(list)
            for row in records:
                groups[row[source]].append(row["metrics"][index])
            assert payload[field][system] == {key: evaluator.macro_average(values) for key, values in groups.items()}
    assert payload["coverage"] == {
        key: sum(row["flags"][key] for row in records) / len(records) for key in records[0]["flags"]
    }
    assert payload["hybrid_union_target_coverage"] == payload["coverage"]["union_candidate"]
    for field, source in (("history", "history"), ("hidden_train_degree", "degree")):
        assert payload["segment_counts"][field] == {
            key: sum(row[source] == key for row in records) for key in {row[source] for row in records}
        }
    assert payload["hybrid_candidate_k"] == 3
    assert payload["candidate_policies"] == {
        "popularity": "full_catalog_minus_train_seen", "content": "full_catalog_minus_train_seen",
        "cf": "bpr-mf-v1_trained_items_only", "hybrid": "per_channel_top_3_union",
    }


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("fusion", ["rrf", "weighted"])
def test_batch_invariant_payload_and_retained_reference(tmp_path, monkeypatch, mixed, fusion) -> None:
    fixture = _fixture(mixed=mixed, fusion=fusion)
    reference = _retained_reference(fixture, fixture.users)
    baseline = None
    for batch_size in (1, 2, 4, 7, 20):
        with monkeypatch.context() as patch:
            trace = _trace(patch)
            payload = _run(tmp_path, patch, fixture, batch_size=batch_size)
            _assert_reference(payload, trace, reference)
            if baseline is None:
                baseline = payload
            assert payload == baseline
    # A hidden full-catalog rank beyond candidate_k still earns standalone metrics.
    assert reference[0]["ranks"][1] > 3
    assert reference[0]["metrics"][1]["mrr"] > 0
    assert fixture.users[0].hidden_product_id not in reference[0]["candidates"][0]
    assert reference[0]["candidates"][0] == ("P02", "P03", "P04")
    if mixed:
        assert reference[3]["candidates"][0] is None  # unknown history
        assert reference[4]["candidates"][0] is None  # empty history
        assert reference[5]["candidates"][0] is None  # zero profile
        assert reference[6]["candidates"][0] is None  # cancelling profile
        assert reference[6]["ranks"][2] > 3  # full CF rank, outside its candidate pool
        assert reference[6]["metrics"][2]["mrr"] > 0
        assert fixture.users[6].hidden_product_id not in reference[6]["candidates"][1]
        assert reference[7]["candidates"][1] == ()  # all CF items seen
        assert reference[8]["flags"]["popularity_candidate"] is True
        assert reference[8]["flags"]["content_candidate"] is False
        assert reference[8]["flags"]["cf_candidate"] is False
        assert set(baseline["segment_counts"]["history"]) == {"train_size=1", "train_size=2-4", "train_size>=5"}
        assert set(baseline["segment_counts"]["hidden_train_degree"]) == {"0", "1", "2+"}


@pytest.mark.parametrize("seed", [7, 42])
def test_sampling_precedes_batching_and_preserves_selected_order(tmp_path, monkeypatch, seed) -> None:
    fixture = _fixture()
    order = sorted(np.random.default_rng(seed).permutation(len(fixture.users))[:5].tolist())
    selected = [fixture.users[index] for index in order]
    reference = _retained_reference(fixture, selected)
    baseline = None
    for batch_size in (1, 2, 7, 20):
        with monkeypatch.context() as patch:
            trace = _trace(patch)
            payload = _run(tmp_path, patch, fixture, batch_size=batch_size, max_users=5, seed=seed)
            _assert_reference(payload, trace, reference)
            assert payload["sampled"] is True and payload["evaluation_users"] == 5
            baseline = baseline or payload
            assert payload == baseline


@pytest.mark.parametrize("batch_size, widths", [(1, [1] * 11), (4, [4, 4, 3]), (20, [11])])
@pytest.mark.parametrize("mixed", [False, True])
def test_score_widths_and_release_before_next_batch(tmp_path, monkeypatch, batch_size, widths, mixed) -> None:
    fixture = _fixture(mixed=mixed)
    content_refs, cf_refs = [], []
    content_widths, cf_widths, processed = [], [], []
    real_content = evaluator._score_content_batch
    real_cf = evaluator.score_cf_users
    user_map = {uid: i for i, uid in enumerate(fixture.cf_users)}

    def cf(model, indices, **kwargs):
        # Check lifetime as well as shape: no prior batch or row views survive.
        assert all(ref() is None for ref in content_refs + cf_refs)
        scores = real_cf(model, indices, **kwargs)
        cf_widths.append(scores.shape[0])
        assert scores.shape[0] <= batch_size and scores.dtype == np.float32
        processed.extend(indices)
        cf_refs.append(weakref.ref(scores))
        return scores

    def content(users, embeddings, id_to_row):
        assert len(users) <= batch_size
        scores, mapping = real_content(users, embeddings, id_to_row)
        if scores is not None:
            assert scores.shape[0] <= batch_size and scores.dtype == np.float32
            content_widths.append(scores.shape[0])
            content_refs.append(weakref.ref(scores))
        return scores, mapping

    monkeypatch.setattr(evaluator, "score_cf_users", cf)
    monkeypatch.setattr(evaluator, "_score_content_batch", content)
    _run(tmp_path, monkeypatch, fixture, batch_size=batch_size)
    assert cf_widths == widths
    assert processed == [user_map[user.user_id] for user in fixture.users]
    assert len(processed) == len(set(processed)) == 11
    assert all(ref() is None for ref in content_refs + cf_refs)
    assert sum(content_widths) == (6 if mixed else 11)
    if not mixed:
        assert content_widths == widths


def test_no_usable_profiles_in_entire_batch(tmp_path, monkeypatch) -> None:
    fixture = _fixture()
    fixture.users = tuple(replace(user, train_product_ids=("UNKNOWN",)) for user in fixture.users)
    fixture.split = replace(fixture.split, users=fixture.users)
    reference = _retained_reference(fixture, fixture.users)
    trace = _trace(monkeypatch)
    payload = _run(tmp_path, monkeypatch, fixture, batch_size=4)
    _assert_reference(payload, trace, reference)
    assert all(row[0] is None for row in trace["candidates"])


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_rejected_before_loading(monkeypatch, capsys, batch_size) -> None:
    monkeypatch.setattr(evaluator, "parse_args", lambda: SimpleNamespace(batch_size=batch_size))
    settings = Mock(side_effect=AssertionError("invalid batch size must fail before artifact setup"))
    monkeypatch.setattr(evaluator, "get_settings", settings)
    assert evaluator.main() == 2
    settings.assert_not_called()
    assert "--batch-size must be at least 1" in capsys.readouterr().err


def test_batch_size_default_remains_256(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_official_recommendations.py"])
    assert evaluator.parse_args().batch_size == 256


def test_missing_cf_user_mapping_still_fails_before_scoring(tmp_path, monkeypatch) -> None:
    fixture = _fixture()
    fixture.cf_users = fixture.cf_users[:-1]
    scorer = Mock(side_effect=AssertionError("missing CF user must not be silently skipped"))
    monkeypatch.setattr(evaluator, "score_cf_users", scorer)
    with pytest.raises(KeyError, match="u00"):
        _run(tmp_path, monkeypatch, fixture, batch_size=4)
    scorer.assert_not_called()
    assert not (tmp_path / "recommendation_comparison.json").exists()
