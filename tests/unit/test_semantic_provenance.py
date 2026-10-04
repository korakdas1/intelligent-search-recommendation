"""Semantic input identity and model revision pinning, with no model download."""

import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.embeddings.encoder import SentenceTransformerEncoder
from app.embeddings.provenance import SemanticCatalogFingerprint
from app.embeddings.text import build_semantic_text
from scripts import build_semantic_index as build


def _fingerprint(rows):
    result = SemanticCatalogFingerprint()
    for product_id, text in rows:
        result.update(product_id, text)
    return result.hexdigest()


def test_fingerprint_framing_and_utf8_are_stable():
    rows = [("A", "Title: café\nBrand: B"), ("B", "Title: Soap")]
    # Independently frame known bytes to fix the format, including byte lengths.
    framed = (
        b"semantic-catalog-sha256-v1\x00"
        + b"\x00\x00\x00\x00\x00\x00\x00\x01A"
        + b"\x00\x00\x00\x00\x00\x00\x00\x15Title: caf\xc3\xa9\nBrand: B"
        + b"\x00\x00\x00\x00\x00\x00\x00\x01B"
        + b"\x00\x00\x00\x00\x00\x00\x00\x0bTitle: Soap"
    )
    assert _fingerprint(rows) == hashlib.sha256(framed).hexdigest()
    assert _fingerprint(iter(rows)) == _fingerprint(rows)
    assert _fingerprint(sorted(reversed(rows))) == _fingerprint(rows)
    assert _fingerprint(reversed(rows)) != _fingerprint(rows)
    assert _fingerprint([("ab", "c")]) != _fingerprint([("a", "bc")])


@pytest.mark.parametrize("field", ["product_id", "title", "brand", "category", "subcategory", "description"])
def test_fingerprint_changes_with_semantic_input(field):
    product = dict(product_id="P1", title="Cream", brand="Acme", category="Beauty", subcategory="Face", description="Soft")

    def digest(row):
        return _fingerprint([(row["product_id"], build_semantic_text(**{k: v for k, v in row.items() if k != "product_id"}))])

    changed = {**product, field: "different"}
    assert digest(changed) != digest(product)


@pytest.mark.parametrize("revision", [None, "0123456789abcdef"])
def test_sentence_transformer_receives_requested_revision(monkeypatch, revision):
    model = Mock()
    model.get_embedding_dimension.return_value = 4
    model.encode.return_value = np.eye(1, 4, dtype=np.float32)
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    encoder = SentenceTransformerEncoder("fixture-model", device="cpu", model_revision=revision)
    assert encoder.embedding_dim == 4
    encoder.encode(["tiny text"])
    kwargs = {"revision": revision} if revision is not None else {}
    constructor.assert_called_once_with("fixture-model", device="cpu", **kwargs)


@pytest.mark.parametrize("revision", [None, "0123456789abcdef"])
def test_cpu_retry_preserves_model_revision(monkeypatch, revision):
    gpu = SimpleNamespace(
        model_name="fixture-model", model_revision=revision, model_license="fixture",
        device="cuda", encode=Mock(side_effect=RuntimeError("GPU unavailable")),
    )
    cpu = Mock()
    cpu.encode.return_value = np.eye(1, 4, dtype=np.float32)
    constructor = Mock(return_value=cpu)
    monkeypatch.setattr(build, "SentenceTransformerEncoder", constructor)
    vectors, encoder = build._encode_with_cpu_fallback(gpu, ["text"], batch_size=1)
    assert encoder is cpu
    assert vectors.shape == (1, 4)
    constructor.assert_called_once_with(
        "fixture-model", device="cpu", model_revision=revision, model_license="fixture",
    )


def test_cpu_error_is_not_retried(monkeypatch):
    cpu = SimpleNamespace(device="cpu", encode=Mock(side_effect=RuntimeError("encode failed")))
    constructor = Mock()
    monkeypatch.setattr(build, "SentenceTransformerEncoder", constructor)
    with pytest.raises(RuntimeError, match="encode failed"):
        build._encode_with_cpu_fallback(cpu, ["text"], batch_size=1)
    constructor.assert_not_called()


def test_build_rejects_unimplemented_text_version():
    with pytest.raises(SystemExit) as exc:
        build.main(["--text-version", "unimplemented"])
    assert exc.value.code == 2
