"""Tiny on-disk semantic artifacts for offline tests. No model download."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.embeddings.checksums import sha256_file
from app.embeddings.constants import FAISS_METRIC, SEMANTIC_TEXT_VERSION
from app.embeddings.io import save_embeddings, save_product_ids, write_json
from app.embeddings.normalize import l2_normalize
from app.search.faiss_index import build_flat_ip_index, build_hnsw_ip_index, save_index


def write_fixture_artifacts(
    root: Path,
    *,
    artifact_version: str,
    dataset_version: str,
    product_ids: list[str],
    embeddings: np.ndarray,
    model_name: str = "fake-encoder",
    text_version: str = SEMANTIC_TEXT_VERSION,
    hnsw: bool = True,
    semantic_catalog_sha256: str | None = None,
) -> Path:
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    embed_dir = root / "embeddings" / artifact_version
    index_dir = root / "indexes" / artifact_version
    embed_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    embeddings_path = embed_dir / "embeddings.npy"
    ids_path = embed_dir / "product_ids.npy"
    save_embeddings(embeddings_path, embeddings)
    save_product_ids(ids_path, product_ids)
    flat = build_flat_ip_index(embeddings)
    flat_path = index_dir / "flat.faiss"
    save_index(flat, flat_path)
    hnsw_path = index_dir / "hnsw.faiss"
    if hnsw:
        hnsw_index = build_hnsw_ip_index(embeddings, m=8, ef_construction=32, ef_search=32)
        save_index(hnsw_index, hnsw_path)
    embedding_manifest = {
        "artifact_version": artifact_version,
        "dataset_version": dataset_version,
        "product_count": len(product_ids),
        "model_name": model_name,
        "model_revision": "test",
        "model_license": "test-only",
        "embedding_dimension": int(embeddings.shape[1]),
        "dtype": "float32",
        "normalized": True,
        "semantic_text_version": text_version,
        "faiss_metric": FAISS_METRIC,
        "exact_index_type": "IndexFlatIP",
        "ann_index_type": "IndexHNSWFlat" if hnsw else None,
        "checksums": {
            "embeddings.npy": sha256_file(embeddings_path),
            "product_ids.npy": sha256_file(ids_path),
        },
    }
    if semantic_catalog_sha256 is not None:
        embedding_manifest["semantic_catalog_sha256"] = semantic_catalog_sha256
    write_json(embed_dir / "manifest.json", embedding_manifest)
    index_manifest = {
        **embedding_manifest,
        "hnsw": {"M": 8, "efConstruction": 32, "efSearch_default": 32} if hnsw else None,
        "checksums": {
            "flat.faiss": sha256_file(flat_path),
            "hnsw.faiss": sha256_file(hnsw_path) if hnsw else None,
            "product_ids.npy": sha256_file(ids_path),
        },
    }
    write_json(index_dir / "manifest.json", index_manifest)
    return root
