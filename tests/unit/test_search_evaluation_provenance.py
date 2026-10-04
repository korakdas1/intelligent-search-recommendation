"""Tiny search/LTR provenance fixtures: no database, encoder downloads, or training."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import faiss
import numpy as np
import pandas as pd
import pytest

from app.core.config import Settings
from app.embeddings.checksums import sha256_file
from app.evaluation.provenance import (
    ProvenanceError,
    ltr_artifact_checksums,
    ranker_provenance,
    semantic_provenance,
    validate_ltr_dataset,
)
from app.ranking.constants import DATASET_VERSION, LABEL_CLASS
from app.ranking.feature_schema import FEATURE_COUNT, FEATURE_NAMES, FEATURE_NAMES_SHA256, FEATURE_VERSION
from scripts import build_ltr_dataset, evaluate_official_search, train_ranker


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _dataset(tmp_path: Path, *, legacy: bool = False) -> Path:
    directory = tmp_path / "dataset"
    directory.mkdir()
    _write_json(directory / "queries.json", [{
        "query_id": "q1", "query": "tiny fixture", "split": "test",
        "query_source": "synthetic_title", "relevant_ids": ["P1"],
    }])
    (directory / "candidates.parquet").write_bytes(b"tiny candidate identity fixture")
    manifest = {
        "dataset_version": DATASET_VERSION, "label_class": LABEL_CLASS,
        "feature_version": FEATURE_VERSION, "feature_names_sha256": FEATURE_NAMES_SHA256,
    }
    if not legacy:
        manifest["checksums"] = ltr_artifact_checksums(directory)
    _write_json(directory / "manifest.json", manifest)
    return directory


def _ranker(tmp_path: Path, dataset: dict) -> SimpleNamespace:
    directory = tmp_path / "custom-ranker"
    directory.mkdir()
    (directory / "model_state.pt").write_bytes(b"fake model state; never deserialized")
    (directory / "scaler.npz").write_bytes(b"fake scaler; never deserialized")
    config = {"model_version": "ranknet-fixture", "feature_version": FEATURE_VERSION}
    _write_json(directory / "model_config.json", config)
    _write_json(directory / "manifest.json", {
        **config, "feature_names_sha256": FEATURE_NAMES_SHA256,
        "dataset_version": DATASET_VERSION,
        "model_checksum": sha256_file(directory / "model_state.pt"),
        "scaler_checksum": sha256_file(directory / "scaler.npz"),
        "model_config_checksum": sha256_file(directory / "model_config.json"),
        "ltr_dataset_provenance": dataset,
    })
    return SimpleNamespace(directory=directory, config=config, **config)


def _semantic(backend: str) -> SimpleNamespace:
    index = faiss.IndexFlatIP(2) if backend == "flat" else faiss.IndexHNSWFlat(2, 4, faiss.METRIC_INNER_PRODUCT)
    if backend == "hnsw":
        index.hnsw.efSearch = 43
    return SimpleNamespace(
        artifact_version="semantic-fixture", dataset_version="catalog-fixture", model_name="fixture-encoder",
        backend=backend, index=index, _catalog_verified=True,
        embedding_manifest={
            "model_revision": "fixture-revision", "semantic_text_version": "semantic-text-v1",
            "semantic_catalog_sha256": "a" * 64, "checksums": {"product_ids.npy": "b" * 64},
        },
        index_manifest={"checksums": {f"{backend}.faiss": "c" * 64}},
    )


@pytest.mark.parametrize("seed", [0, 42, 2026])
def test_source_selection_canonicalizes_before_seeded_shuffle(seed: int) -> None:
    rows = [(f"P{i:02d}", "Leather Conditioner", None) for i in range(12)]
    expected = rows.copy()
    random.Random(seed).shuffle(expected)
    for order in (rows, rows[::-1], rows[5:] + rows[:5]):
        assert build_ltr_dataset.select_source_products(order, seed=seed, max_sources=5) == expected[:5]
    assert rows[0][0] == "P00"


def test_tiny_build_publishes_hashes_and_reproducible_query_bytes(tmp_path, monkeypatch) -> None:
    rows = [(f"P{i}", "Howard Leather Conditioner Extra", "Howard") for i in range(5)]
    session = SimpleNamespace(execute=lambda statement: rows)
    monkeypatch.setattr(build_ltr_dataset, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(build_ltr_dataset, "reset_engine", lambda: None)
    monkeypatch.setattr(build_ltr_dataset, "reset_semantic_runtime", lambda: None)
    monkeypatch.setattr(build_ltr_dataset, "set_semantic_runtime", lambda runtime: None)
    monkeypatch.setattr(build_ltr_dataset, "load_semantic_runtime", lambda settings: _semantic("flat"))
    monkeypatch.setattr(build_ltr_dataset, "get_session_factory", lambda: lambda: nullcontext(session))
    monkeypatch.setattr(build_ltr_dataset, "collect_fused_candidates", lambda *a, **k: ([], 100))
    monkeypatch.setattr(build_ltr_dataset, "get_products_by_ids", lambda *a: {})
    monkeypatch.setattr(build_ltr_dataset, "extract_features", lambda *a, **k: SimpleNamespace(
        product_ids=("P1",), values=np.zeros((1, FEATURE_COUNT), dtype=np.float32), feature_version=FEATURE_VERSION,
    ))
    monkeypatch.setattr(build_ltr_dataset, "_git_commit", lambda: "fixture")
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        assert build_ltr_dataset.main(["--out-dir", str(output), "--max-sources", "4"]) == 0
        manifest, provenance = validate_ltr_dataset(output)
        assert manifest["source_order_policy"] == "product_id_asc_before_seeded_shuffle"
        assert manifest["checksums"] == ltr_artifact_checksums(output)
        assert provenance["status"] == "verified"
        assert set(manifest["checksums"]) == {"queries.json", "candidates.parquet"}
        rows.reverse()
    assert (outputs[0] / "queries.json").read_bytes() == (outputs[1] / "queries.json").read_bytes()


@pytest.mark.parametrize("filename", ["queries.json", "candidates.parquet"])
def test_ltr_declared_corruption_rejected(tmp_path, filename) -> None:
    directory = _dataset(tmp_path)
    (directory / filename).write_bytes(b"changed")
    with pytest.raises(ProvenanceError, match=f"{filename} checksum mismatch"):
        validate_ltr_dataset(directory)


@pytest.mark.parametrize("field", ["dataset_version", "label_class", "feature_version", "feature_names_sha256"])
def test_ltr_incompatible_manifest_rejected(tmp_path, field) -> None:
    directory = _dataset(tmp_path)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = "wrong"
    _write_json(path, manifest)
    with pytest.raises(ProvenanceError, match=field):
        validate_ltr_dataset(directory)


@pytest.mark.parametrize("checksums", [None, [], {"queries.json": None}, {"candidates.parquet": "not-a-digest"}])
def test_malformed_checksum_declarations_rejected(tmp_path, checksums) -> None:
    directory = _dataset(tmp_path)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["checksums"] = checksums
    _write_json(path, manifest)
    with pytest.raises(ProvenanceError):
        validate_ltr_dataset(directory)


@pytest.mark.parametrize("missing_manifest", [False, True])
def test_legacy_dataset_warns_without_mutation(tmp_path, missing_manifest) -> None:
    directory = _dataset(tmp_path, legacy=True)
    path = directory / "manifest.json"
    if missing_manifest:
        path.unlink()
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.warns(RuntimeWarning, match="legacy/unverified"):
        _, provenance = validate_ltr_dataset(directory)
    assert provenance["status"] == "legacy_unverified"
    assert provenance["queries_sha256"] == sha256_file(directory / "queries.json")
    assert provenance["declared_checksums"] == {}
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_search_does_not_require_optional_candidate_parquet(tmp_path) -> None:
    directory = _dataset(tmp_path)
    (directory / "candidates.parquet").unlink()
    with pytest.warns(RuntimeWarning, match="legacy/unverified"):
        _, provenance = validate_ltr_dataset(directory, required_files=("queries.json",))
    assert provenance["file_status"] == {"queries.json": "verified", "candidates.parquet": "unavailable"}
    assert provenance["candidates_sha256"] is None


@pytest.mark.parametrize("filename", ["queries.json", "candidates.parquet"])
def test_training_rejects_corruption_before_reading_or_training(tmp_path, monkeypatch, filename, capsys) -> None:
    directory = _dataset(tmp_path)
    (directory / filename).write_bytes(b"changed")
    read = Mock(side_effect=AssertionError("parquet should not be read"))
    train = Mock(side_effect=AssertionError("training should not start"))
    monkeypatch.setattr(train_ranker.pd, "read_parquet", read)
    monkeypatch.setattr(train_ranker, "train_ranknet", train)
    output = tmp_path / "new-model"
    assert train_ranker.main(["--dataset", str(directory), "--out-dir", str(output)]) == 2
    read.assert_not_called()
    train.assert_not_called()
    assert not output.exists()
    assert "provenance rejected" in capsys.readouterr().err


@pytest.mark.parametrize("legacy", [False, True])
def test_future_training_records_dataset_identity_without_real_training(tmp_path, monkeypatch, legacy) -> None:
    directory = _dataset(tmp_path, legacy=legacy)
    with pytest.warns(RuntimeWarning, match="legacy/unverified") if legacy else nullcontext():
        _, provenance = validate_ltr_dataset(directory)
    frame = pd.DataFrame(np.zeros((1, FEATURE_COUNT)), columns=FEATURE_NAMES)
    frame["split"] = "train"
    monkeypatch.setattr(train_ranker.pd, "read_parquet", lambda path: frame)
    model = SimpleNamespace(parameter_count=lambda: 1)
    summary = {"best_validation_loss": 0.1, "best_epoch": 1, "train_pair_count": 1, "epochs_run": 1, "history": []}
    monkeypatch.setattr(train_ranker, "train_ranknet", lambda *a, **k: (model, summary))
    monkeypatch.setattr(train_ranker, "_git_commit", lambda: "fixture")

    def save_fake(directory, *, config, manifest, **kwargs):
        assert manifest["ltr_dataset_provenance"] == provenance
        (directory / "model_state.pt").write_bytes(b"fake state")
        (directory / "scaler.npz").write_bytes(b"fake scaler")
        _write_json(directory / "model_config.json", config)

    monkeypatch.setattr(train_ranker, "save_ranker_bundle", save_fake)
    output = tmp_path / "new-model"
    with pytest.warns(RuntimeWarning, match="legacy/unverified") if legacy else nullcontext():
        assert train_ranker.main([
            "--dataset", str(directory), "--out-dir", str(output), "--skip-architecture-compare",
        ]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["ltr_dataset_provenance"] == provenance
    assert manifest["model_config_checksum"] == sha256_file(output / "model_config.json")


@pytest.mark.parametrize("field", ["queries_sha256", "candidates_sha256"])
def test_ranker_dataset_linkage_match_and_mismatch(tmp_path, field) -> None:
    _, dataset = validate_ltr_dataset(_dataset(tmp_path))
    runtime = _ranker(tmp_path, dataset)
    result = ranker_provenance(runtime, dataset)
    assert result["status"] == result["dataset_linkage_status"] == "verified"
    assert result["model_state_sha256"] == sha256_file(runtime.directory / "model_state.pt")
    path = runtime.directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["ltr_dataset_provenance"][field] = "0" * 64
    _write_json(path, manifest)
    with pytest.raises(ProvenanceError, match="linkage mismatch"):
        ranker_provenance(runtime, dataset)


@pytest.mark.parametrize("filename", ["model_state.pt", "scaler.npz", "model_config.json"])
def test_ranker_declared_file_mismatch_rejected(tmp_path, filename) -> None:
    _, dataset = validate_ltr_dataset(_dataset(tmp_path))
    runtime = _ranker(tmp_path, dataset)
    (runtime.directory / filename).write_bytes(b"tampered")
    with pytest.raises(ProvenanceError, match="checksum mismatch"):
        ranker_provenance(runtime, dataset)


def test_legacy_ranker_does_not_fabricate_dataset_linkage(tmp_path) -> None:
    _, dataset = validate_ltr_dataset(_dataset(tmp_path))
    runtime = _ranker(tmp_path, dataset)
    _write_json(runtime.directory / "manifest.json", runtime.config)
    with pytest.warns(RuntimeWarning, match="legacy/unverified"):
        result = ranker_provenance(runtime, dataset)
    assert result["status"] == result["dataset_linkage_status"] == "legacy_unverified"
    assert result["ltr_dataset_provenance"]["queries_sha256"] is None


@pytest.mark.parametrize("backend, expected", [("flat", "IndexFlatIP"), ("hnsw", "IndexHNSWFlat")])
def test_semantic_provenance_describes_actual_index(backend, expected) -> None:
    result = semantic_provenance(_semantic(backend), configured_backend="hnsw")
    assert result["faiss"]["type"] == expected
    assert result["faiss"]["metric"] == "inner_product"
    assert result["configured_backend"] == "hnsw"
    assert result["loaded_backend"] == backend
    assert result["model_revision"] == "fixture-revision"
    assert result["semantic_catalog_sha256"] == "a" * 64
    assert result["index_sha256"] == "c" * 64
    assert result["hnsw_ef_search"] == (43 if backend == "hnsw" else None)


@pytest.fixture
def tiny_evaluation(tmp_path, monkeypatch):
    directory = _dataset(tmp_path)
    _, dataset = validate_ltr_dataset(directory)
    ranker = _ranker(tmp_path, dataset)
    semantic = _semantic("hnsw")
    monkeypatch.setattr(evaluate_official_search, "get_settings", lambda: Settings(
        _env_file=None, artifacts_root=str(tmp_path), semantic_index_type="hnsw", hybrid_candidate_k_max=50,
        semantic_hnsw_ef_search=71,
    ))
    monkeypatch.setattr(evaluate_official_search, "get_session_factory", lambda: lambda: nullcontext(object()))
    monkeypatch.setattr(evaluate_official_search, "require_semantic_runtime", lambda session: semantic)
    monkeypatch.setattr(evaluate_official_search, "require_ltr_runtime", lambda: ranker)
    monkeypatch.setattr(evaluate_official_search, "_git_commit", lambda: "fixture")
    ranked = {system: ["P1", "P2"] for system in evaluate_official_search.SYSTEMS}
    run = Mock(side_effect=lambda *a, **k: ranked)
    monkeypatch.setattr(evaluate_official_search, "_run_systems", run)
    output = tmp_path / "evaluation" / "search_comparison.json"
    return SimpleNamespace(
        directory=directory, ranker=ranker, semantic=semantic, output=output, ranked=ranked, run=run,
        args=["--dataset", str(directory), "--output", str(output.parent), "--latency-sample-size", "0"],
    )


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("system", ["hybrid_rrf", "hybrid_weighted", "hybrid_ltr"])
def test_candidate_identity_failure_never_writes(tiny_evaluation, monkeypatch, capsys, existing, system) -> None:
    fixture = tiny_evaluation
    fixture.ranked[system] = ["DIFFERENT"]
    if existing:
        fixture.output.parent.mkdir()
        fixture.output.write_bytes(b"historical output must remain unchanged")
    writer = Mock(side_effect=AssertionError("invalid output must not be written"))
    monkeypatch.setattr(evaluate_official_search, "write_json_atomic", writer)
    assert evaluate_official_search.main(fixture.args + ["--force"]) == 1
    writer.assert_not_called()
    if existing:
        assert fixture.output.read_bytes() == b"historical output must remain unchanged"
    else:
        assert not fixture.output.exists()
    captured = capsys.readouterr()
    assert "hybrid candidate identity failed" in captured.err
    assert '"hybrid_candidate_identity_fail": 1' in captured.out


@pytest.mark.parametrize("backend, expected", [("flat", "IndexFlatIP"), ("hnsw", "IndexHNSWFlat")])
def test_valid_evaluation_writes_atomic_provenance(tiny_evaluation, monkeypatch, backend, expected) -> None:
    fixture = tiny_evaluation
    monkeypatch.setattr(evaluate_official_search, "require_semantic_runtime", lambda session: _semantic(backend))
    replacements = []
    replace = Path.replace

    def record_replace(source, target):
        assert source.name == "search_comparison.json.tmp"
        assert not fixture.output.exists()
        assert json.loads(source.read_text())["input_provenance"]["status"] == "verified"
        replacements.append((source, target))
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", record_replace)
    assert evaluate_official_search.main(fixture.args + ["--max-queries", "1"]) == 0
    assert len(replacements) == 1
    payload = json.loads(fixture.output.read_text())
    assert payload["input_provenance"]["ltr_dataset"]["queries_sha256"] == sha256_file(fixture.directory / "queries.json")
    assert payload["input_provenance"]["ranker"]["artifact_dir"] == "models/custom-ranker"
    assert payload["semantic_backend"] == expected
    assert payload["ltr_model"] == "ranknet-fixture"
    config = payload["effective_config"]
    assert config["semantic_backend"] == backend
    assert config["semantic_faiss_type"] == expected
    assert config["semantic_hnsw_ef_search"] == (71 if backend == "hnsw" else None)
    assert config["candidate_k"] == 100 and config["hybrid_candidate_k"] == 50
    assert config["split"] == "test" and config["k_values"] == [5, 10, 20]
    assert config["rrf_k0"] == 60 and config["weighted_alpha"] == 0.5
    assert config["max_queries"] == 1 and config["sampled"] is True
    assert config["latency_sample_size"] == 0
    assert payload["hybrid_candidate_identity_ok"] == 1 and payload["hybrid_candidate_identity_fail"] == 0
    assert not fixture.output.with_suffix(".json.tmp").exists()
    assert str(fixture.directory.parent) not in fixture.output.read_text()


def test_evaluation_refuses_overwrite_without_force(tiny_evaluation) -> None:
    fixture = tiny_evaluation
    fixture.output.parent.mkdir()
    fixture.output.write_bytes(b"historical")
    assert evaluate_official_search.main(fixture.args) == 2
    fixture.run.assert_not_called()
    assert fixture.output.read_bytes() == b"historical"


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["query", "dataset_metadata", "ranker_checksum", "ranker_linkage"])
def test_evaluation_preflight_rejects_before_retrieval(tiny_evaluation, failure, existing, capsys) -> None:
    fixture = tiny_evaluation
    if failure == "query":
        (fixture.directory / "queries.json").write_bytes(b"changed")
    elif failure == "dataset_metadata":
        path = fixture.directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["label_class"] = "wrong"
        _write_json(path, manifest)
    elif failure == "ranker_checksum":
        (fixture.ranker.directory / "scaler.npz").write_bytes(b"changed")
    else:
        path = fixture.ranker.directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["ltr_dataset_provenance"]["queries_sha256"] = "0" * 64
        _write_json(path, manifest)
    if existing:
        fixture.output.parent.mkdir()
        fixture.output.write_bytes(b"historical")
    assert evaluate_official_search.main(fixture.args + ["--force"]) == 2
    fixture.run.assert_not_called()
    if existing:
        assert fixture.output.read_bytes() == b"historical"
    else:
        assert not fixture.output.exists()
    assert "provenance rejected" in capsys.readouterr().err


def test_evaluation_legacy_output_is_explicitly_unverified(tiny_evaluation) -> None:
    fixture = tiny_evaluation
    (fixture.directory / "manifest.json").unlink()
    (fixture.ranker.directory / "manifest.json").unlink()
    fixture.semantic.embedding_manifest.pop("semantic_catalog_sha256")
    with pytest.warns(RuntimeWarning, match="legacy/unverified"):
        assert evaluate_official_search.main(fixture.args) == 0
    provenance = json.loads(fixture.output.read_text())["input_provenance"]
    assert provenance["status"] == "legacy_unverified"
    assert provenance["ltr_dataset"]["status"] == "legacy_unverified"
    assert provenance["ranker"]["dataset_linkage_status"] == "legacy_unverified"
    assert provenance["semantic"]["status"] == "legacy_unverified"


@pytest.mark.parametrize("field", ["dataset_version", "label_class", "feature_version", "feature_names_sha256"])
def test_missing_legacy_metadata_never_claims_verified(tmp_path, field) -> None:
    directory = _dataset(tmp_path)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop(field)
    _write_json(path, manifest)
    with pytest.warns(RuntimeWarning, match="legacy/unverified"):
        _, provenance = validate_ltr_dataset(directory)
    assert provenance["status"] == "legacy_unverified"


@pytest.mark.parametrize("contents", ["{", "[]", "null"])
def test_invalid_manifest_is_rejected_not_legacy(tmp_path, contents) -> None:
    directory = _dataset(tmp_path)
    (directory / "manifest.json").write_text(contents)
    with pytest.raises(ProvenanceError):
        validate_ltr_dataset(directory)


def test_optional_missing_candidate_identity_is_not_falsely_linked(tiny_evaluation) -> None:
    fixture = tiny_evaluation
    (fixture.directory / "candidates.parquet").unlink()
    with pytest.warns(RuntimeWarning, match="legacy/unverified"):
        assert evaluate_official_search.main(fixture.args) == 0
    provenance = json.loads(fixture.output.read_text())["input_provenance"]
    assert provenance["status"] == "legacy_unverified"
    assert provenance["ranker"]["dataset_linkage_status"] == "legacy_unverified"
    assert provenance["ltr_dataset"]["queries_sha256"] == sha256_file(fixture.directory / "queries.json")


def test_missing_candidate_file_still_rejects_conflicting_declared_link(tmp_path) -> None:
    directory = _dataset(tmp_path)
    _, dataset = validate_ltr_dataset(directory)
    runtime = _ranker(tmp_path, dataset)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["checksums"]["candidates.parquet"] = "0" * 64
    _write_json(path, manifest)
    (directory / "candidates.parquet").unlink()
    with pytest.warns(RuntimeWarning, match="legacy/unverified"):
        _, dataset = validate_ltr_dataset(directory, required_files=("queries.json",))
    with pytest.raises(ProvenanceError, match="linkage mismatch"):
        ranker_provenance(runtime, dataset)
