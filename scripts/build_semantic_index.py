#!/usr/bin/env python3
"""Build catalog embeddings plus exact and HNSW FAISS indexes.

Reads products from PostgreSQL (system of record), not Amazon JSONL.
Refuses to overwrite an existing artifact version unless --force.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from app.core.config import get_settings
from app.core.events import DATASET_VERSION_FULL
from app.db.repositories.artifacts import upsert_artifact_version
from app.db.repositories.products import count_products, iter_products_ordered
from app.db.session import get_session_factory, reset_engine
from app.embeddings.checksums import sha256_file
from app.embeddings.constants import (
    ANN_INDEX_TYPE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EXACT_INDEX_TYPE,
    FAISS_METRIC,
    SEMANTIC_TEXT_VERSION,
    VECTOR_DTYPE,
)
from app.embeddings.encoder import SentenceTransformerEncoder, resolve_device
from app.embeddings.io import save_embeddings, save_product_ids, write_json
from app.embeddings.normalize import is_unit_normalized, row_norms
from app.embeddings.provenance import SemanticCatalogFingerprint
from app.embeddings.text import build_semantic_text
from app.embeddings.versioning import (
    build_artifact_version,
    embedding_paths,
    index_paths,
)
from app.search.faiss_index import (
    build_flat_ip_index,
    build_hnsw_ip_index,
    save_index,
)

logger = logging.getLogger("build_semantic_index")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _model_card_facts(model_name: str) -> dict[str, str | None]:
    facts: dict[str, str | None] = {
        "model_revision": None,
        "model_license": None,
        "max_seq_length": None,
        "model_card_url": f"https://huggingface.co/{model_name}",
    }
    try:
        from huggingface_hub import model_info

        info = model_info(model_name)
        facts["model_revision"] = getattr(info, "sha", None)
        card = getattr(info, "card_data", None)
        if card is not None:
            license_value = getattr(card, "license", None)
            if license_value is None and hasattr(card, "get"):
                license_value = card.get("license")
            if license_value is not None:
                facts["model_license"] = str(license_value)
    except Exception as exc:  # noqa: BLE001 — card lookup is best-effort
        logger.warning("could not resolve model card metadata: %s", exc)
    return facts


def _encode_with_cpu_fallback(
    encoder: SentenceTransformerEncoder, texts: list[str], *, batch_size: int
) -> tuple[np.ndarray, SentenceTransformerEncoder]:
    """Retry GPU encoding with the same requested model revision on CPU."""

    try:
        vectors = encoder.encode(texts, batch_size=batch_size, show_progress=True)
    except Exception:
        if encoder.device == "cpu":
            raise
        logger.warning("GPU encode failed; retrying remaining work on CPU")
        encoder = SentenceTransformerEncoder(
            encoder.model_name, device="cpu", model_revision=encoder.model_revision,
            model_license=encoder.model_license,
        )
        vectors = encoder.encode(texts, batch_size=batch_size, show_progress=True)
    return vectors, encoder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default=DATASET_VERSION_FULL)
    parser.add_argument("--model-name", default=EMBEDDING_MODEL_NAME)
    parser.add_argument("--text-version", default=SEMANTIC_TEXT_VERSION)
    parser.add_argument("--artifact-version", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--fetch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hnsw-m", type=int, default=DEFAULT_HNSW_M)
    parser.add_argument("--hnsw-ef-construction", type=int, default=DEFAULT_HNSW_EF_CONSTRUCTION)
    parser.add_argument("--hnsw-ef-search", type=int, default=DEFAULT_HNSW_EF_SEARCH)
    parser.add_argument("--skip-hnsw", action="store_true", help="Build exact IndexFlatIP only")
    parser.add_argument("--force", action="store_true", help="Replace an existing artifact version")
    parser.add_argument(
        "--register",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upsert artifact_versions rows (default: yes)",
    )
    args = parser.parse_args(argv)
    if args.text_version != SEMANTIC_TEXT_VERSION:
        parser.error(f"--text-version must match the implemented builder: {SEMANTIC_TEXT_VERSION}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    artifact_version = args.artifact_version or build_artifact_version(
        dataset_version=args.dataset_version,
        model_name=args.model_name,
        text_version=args.text_version,
    )
    artifacts_root = Path(settings.artifacts_root)
    embed_paths = embedding_paths(artifacts_root, artifact_version)
    idx_paths = index_paths(artifacts_root, artifact_version)

    dest_exists = embed_paths["dir"].exists() or idx_paths["dir"].exists()
    if dest_exists and not args.force:
        logger.error(
            "artifact version %s already exists; pass --force to rebuild",
            artifact_version,
        )
        return 2

    reset_engine()
    factory = get_session_factory()
    with factory() as session:
        product_count = count_products(session, dataset_version=args.dataset_version)
        if product_count < 1:
            logger.error("no products for dataset_version=%s", args.dataset_version)
            return 1

        try:
            device = resolve_device(args.device)
        except RuntimeError as exc:
            logger.warning("%s; falling back to CPU", exc)
            device = "cpu"

        card = _model_card_facts(args.model_name)
        logger.info("loading encoder %s on %s", args.model_name, device)
        load_started = time.perf_counter()
        encoder = SentenceTransformerEncoder(
            args.model_name, device=device, model_revision=card["model_revision"],
            model_license=card["model_license"],
        )
        dim = encoder.embedding_dim
        model_load_s = time.perf_counter() - load_started
        if dim != EMBEDDING_DIMENSION:
            logger.warning("model dimension %s differs from documented %s", dim, EMBEDDING_DIMENSION)
        if encoder._model is not None:
            max_seq = getattr(encoder._model, "max_seq_length", None)
            if max_seq is not None:
                card["max_seq_length"] = str(int(max_seq))

        embeddings = np.empty((product_count, dim), dtype=np.float32)
        product_ids: list[str] = []
        fingerprint = SemanticCatalogFingerprint()
        encode_started = time.perf_counter()
        offset = 0
        while offset < product_count:
            rows = iter_products_ordered(
                session,
                dataset_version=args.dataset_version,
                offset=offset,
                limit=args.fetch_size,
            )
            if not rows:
                break
            texts = [
                build_semantic_text(
                    title=row.title,
                    brand=row.brand,
                    category=row.category,
                    subcategory=row.subcategory,
                    description=row.description,
                )
                for row in rows
            ]
            vectors, encoder = _encode_with_cpu_fallback(encoder, texts, batch_size=args.batch_size)
            device = encoder.device
            if vectors.shape != (len(rows), dim):
                raise RuntimeError(f"unexpected encode shape {vectors.shape}")
            embeddings[offset : offset + len(rows)] = vectors
            product_ids.extend(row.product_id for row in rows)
            for row, text in zip(rows, texts, strict=True):
                fingerprint.update(row.product_id, text)
            offset += len(rows)
            logger.info("encoded %s / %s", offset, product_count)
        encode_s = time.perf_counter() - encode_started

        if len(product_ids) != product_count:
            raise RuntimeError(f"encoded {len(product_ids)} products, expected {product_count}")
        if not is_unit_normalized(embeddings[: min(256, product_count)]):
            logger.warning("sample vectors are not unit-normalized; check encoder settings")

        created_at = datetime.now(UTC).isoformat()
        git_commit = _git_commit()
        tmp_root = Path(tempfile.mkdtemp(prefix="semantic-build-", dir=artifacts_root.parent if artifacts_root.parent.exists() else None))
        try:
            tmp_embed = tmp_root / "embeddings"
            tmp_index = tmp_root / "indexes"
            tmp_embed.mkdir(parents=True)
            tmp_index.mkdir(parents=True)
            embeddings_path = tmp_embed / "embeddings.npy"
            ids_path = tmp_embed / "product_ids.npy"
            save_embeddings(embeddings_path, embeddings)
            save_product_ids(ids_path, product_ids)

            flat_started = time.perf_counter()
            flat = build_flat_ip_index(embeddings)
            flat_s = time.perf_counter() - flat_started
            if int(flat.ntotal) != product_count or int(flat.d) != dim:
                raise RuntimeError("exact index ntotal/dimension mismatch")
            flat_path = tmp_index / "flat.faiss"
            save_index(flat, flat_path)

            hnsw_s = None
            hnsw_path = tmp_index / "hnsw.faiss"
            if not args.skip_hnsw:
                hnsw_started = time.perf_counter()
                hnsw = build_hnsw_ip_index(
                    embeddings,
                    m=args.hnsw_m,
                    ef_construction=args.hnsw_ef_construction,
                    ef_search=args.hnsw_ef_search,
                )
                hnsw_s = time.perf_counter() - hnsw_started
                if int(hnsw.ntotal) != product_count or int(hnsw.d) != dim:
                    raise RuntimeError("HNSW index ntotal/dimension mismatch")
                save_index(hnsw, hnsw_path)

            sample_norms = row_norms(embeddings[: min(1024, product_count)])
            embedding_manifest = {
                "artifact_version": artifact_version,
                "dataset_version": args.dataset_version,
                "product_count": product_count,
                "model_name": args.model_name,
                "model_revision": card["model_revision"],
                "model_license": card["model_license"],
                "model_card_url": card["model_card_url"],
                "max_seq_length": card["max_seq_length"],
                "embedding_dimension": dim,
                "dtype": VECTOR_DTYPE,
                "normalized": True,
                "semantic_text_version": args.text_version,
                "semantic_catalog_sha256": fingerprint.hexdigest(),
                "faiss_metric": FAISS_METRIC,
                "exact_index_type": EXACT_INDEX_TYPE,
                "ann_index_type": None if args.skip_hnsw else ANN_INDEX_TYPE,
                "created_at": created_at,
                "build_device": device,
                "requested_device": args.device,
                "build_batch_size": args.batch_size,
                "fetch_size": args.fetch_size,
                "durations_seconds": {
                    "model_load": round(model_load_s, 3),
                    "encode": round(encode_s, 3),
                    "exact_index": round(flat_s, 3),
                    "hnsw_index": None if hnsw_s is None else round(hnsw_s, 3),
                },
                "vectors_per_second": round(product_count / encode_s, 2) if encode_s else None,
                "sample_norm_mean": float(np.mean(sample_norms)),
                "git_commit": git_commit,
                "relative_paths": {
                    "embeddings": str(embed_paths["embeddings"].as_posix()),
                    "product_ids": str(embed_paths["product_ids"].as_posix()),
                },
                "file_sizes_bytes": {
                    "embeddings.npy": embeddings_path.stat().st_size,
                    "product_ids.npy": ids_path.stat().st_size,
                },
                "checksums": {
                    "embeddings.npy": sha256_file(embeddings_path),
                    "product_ids.npy": sha256_file(ids_path),
                },
            }
            write_json(tmp_embed / "manifest.json", embedding_manifest)

            index_manifest = {
                "artifact_version": artifact_version,
                "dataset_version": args.dataset_version,
                "product_count": product_count,
                "model_name": args.model_name,
                "model_revision": card["model_revision"],
                "model_license": card["model_license"],
                "embedding_dimension": dim,
                "dtype": VECTOR_DTYPE,
                "normalized": True,
                "semantic_text_version": args.text_version,
                "semantic_catalog_sha256": fingerprint.hexdigest(),
                "faiss_metric": FAISS_METRIC,
                "exact_index_type": EXACT_INDEX_TYPE,
                "ann_index_type": None if args.skip_hnsw else ANN_INDEX_TYPE,
                "hnsw": None
                if args.skip_hnsw
                else {
                    "M": args.hnsw_m,
                    "efConstruction": args.hnsw_ef_construction,
                    "efSearch_default": args.hnsw_ef_search,
                },
                "created_at": created_at,
                "build_device": device,
                "build_batch_size": args.batch_size,
                "durations_seconds": embedding_manifest["durations_seconds"],
                "git_commit": git_commit,
                "relative_paths": {
                    "flat": str(idx_paths["flat"].as_posix()),
                    "hnsw": None if args.skip_hnsw else str(idx_paths["hnsw"].as_posix()),
                },
                "file_sizes_bytes": {
                    "flat.faiss": flat_path.stat().st_size,
                    "hnsw.faiss": None if args.skip_hnsw else hnsw_path.stat().st_size,
                },
                "checksums": {
                    "flat.faiss": sha256_file(flat_path),
                    "hnsw.faiss": None if args.skip_hnsw else sha256_file(hnsw_path),
                    "product_ids.npy": embedding_manifest["checksums"]["product_ids.npy"],
                },
            }
            write_json(tmp_index / "manifest.json", index_manifest)

            embed_paths["dir"].parent.mkdir(parents=True, exist_ok=True)
            idx_paths["dir"].parent.mkdir(parents=True, exist_ok=True)
            if dest_exists:
                logger.warning("replacing existing artifact version %s", artifact_version)
                if embed_paths["dir"].exists():
                    shutil.rmtree(embed_paths["dir"])
                if idx_paths["dir"].exists():
                    shutil.rmtree(idx_paths["dir"])
            shutil.move(str(tmp_embed), str(embed_paths["dir"]))
            shutil.move(str(tmp_index), str(idx_paths["dir"]))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        if args.register:
            upsert_artifact_version(
                session,
                artifact_id=f"embeddings:{artifact_version}",
                artifact_type="embedding_model",
                version=artifact_version,
                dataset_version=args.dataset_version,
                embedding_model_name=args.model_name,
                embedding_dim=dim,
                metric=FAISS_METRIC,
                path=str(embed_paths["dir"].as_posix()),
                metadata=embedding_manifest,
            )
            upsert_artifact_version(
                session,
                artifact_id=f"faiss-flat:{artifact_version}",
                artifact_type="faiss_index",
                version=artifact_version,
                dataset_version=args.dataset_version,
                embedding_model_name=args.model_name,
                embedding_dim=dim,
                metric=FAISS_METRIC,
                path=str(idx_paths["flat"].as_posix()),
                metadata={"index_type": EXACT_INDEX_TYPE, **index_manifest},
            )
            if not args.skip_hnsw:
                upsert_artifact_version(
                    session,
                    artifact_id=f"faiss-hnsw:{artifact_version}",
                    artifact_type="faiss_index",
                    version=artifact_version,
                    dataset_version=args.dataset_version,
                    embedding_model_name=args.model_name,
                    embedding_dim=dim,
                    metric=FAISS_METRIC,
                    path=str(idx_paths["hnsw"].as_posix()),
                    metadata={"index_type": ANN_INDEX_TYPE, **index_manifest},
                )
            session.commit()

        logger.info(
            "built %s products dim=%s device=%s encode=%.1fs exact=%.1fs hnsw=%s",
            product_count,
            dim,
            device,
            encode_s,
            flat_s,
            "skipped" if hnsw_s is None else f"{hnsw_s:.1f}s",
        )
        logger.info("artifact version %s", artifact_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
