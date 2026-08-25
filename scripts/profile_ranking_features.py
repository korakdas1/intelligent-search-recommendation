#!/usr/bin/env python3
"""Profile rank-features-v1 on hybrid candidates. Does not train a ranker."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sqlalchemy import select

from app.core.config import get_settings
from app.core.events import DATASET_VERSION_FULL
from app.db.repositories.artifacts import upsert_artifact_version
from app.db.repositories.products import get_products_by_ids
from app.db.session import get_session_factory, reset_engine
from app.embeddings.checksums import sha256_file
from app.models.product import Product
from app.ranking.feature_schema import FEATURE_NAMES, FEATURE_VERSION, schema_document
from app.ranking.features import extract_features
from app.ranking.io import feature_frame, write_feature_parquet
from app.ranking.product_view import ProductView
from app.ranking.query_ids import make_query_id
from app.ranking.retrieval import collect_fused_candidates
from app.ranking.validate import validate_feature_batch
from app.search.runtime import load_semantic_runtime, reset_semantic_runtime, set_semantic_runtime

ISSUED_QUERIES = [
    "leather conditioner",
    "howard leather",
    "sunscreen",
    "product for restoring leather",
    "face moisturizer for sensitive skin",
    "product that reduces frizz",
    "xyzzyqwerty12345notaproduct",
    "the",
]

FALSE_FRIEND_QUERY = "product for restoring leather"
FALSE_FRIEND_IDS = ("B00DMQT52I", "B089N65Q6G", "B0178JZSFC")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 95), 3),
        "max": round(max(values), 3),
    }


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _feature_stats(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for index, name in enumerate(FEATURE_NAMES):
        column = matrix[:, index]
        stats[name] = {
            "min": float(np.min(column)),
            "max": float(np.max(column)),
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
            "median": float(np.median(column)),
            "zero_pct": float(np.mean(column == 0.0) * 100.0),
        }
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issued-limit", type=int, default=len(ISSUED_QUERIES))
    parser.add_argument("--synthetic-n", type=int, default=92)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        default="artifacts/ranking_features/rank-features-v1",
    )
    parser.add_argument("--register-artifact", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    reset_engine()
    reset_semantic_runtime()
    runtime = load_semantic_runtime(settings)
    set_semantic_runtime(runtime)
    factory = get_session_factory()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    extract_ms: list[float] = []
    fetch_ms: list[float] = []
    candidate_counts: list[int] = []
    query_records: list[dict] = []

    with factory() as session:
        issued = ISSUED_QUERIES[: args.issued_limit]
        title_stmt = (
            select(Product.product_id, Product.title)
            .where(Product.dataset_version == runtime.dataset_version)
            .where(Product.title.is_not(None))
            .order_by(Product.product_id.asc())
            .limit(args.synthetic_n)
        )
        synthetic = [
            (str(title), "synthetic_title")
            for _, title in session.execute(title_stmt)
            if title
        ]
        jobs = [(query, "issued") for query in issued] + synthetic

        if args.warmup:
            collect_fused_candidates(
                session, issued[0], candidate_k=args.candidate_k, top_k=args.top_k
            )
            session.rollback()

        for query, source in jobs:
            fused, depth = collect_fused_candidates(
                session, query, candidate_k=args.candidate_k, top_k=args.top_k
            )
            started = time.perf_counter()
            products = get_products_by_ids(session, [row.product_id for row in fused])
            fetch_ms.append((time.perf_counter() - started) * 1000)
            views = {pid: ProductView.from_product(item) for pid, item in products.items()}
            started = time.perf_counter()
            batch = extract_features(
                query,
                fused,
                views,
                candidate_k=depth,
                rrf_k=settings.hybrid_rrf_k,
                keyword_weight=settings.hybrid_keyword_weight,
            )
            extract_ms.append((time.perf_counter() - started) * 1000)
            validate_feature_batch(batch)
            candidate_counts.append(len(batch.product_ids))
            query_id = make_query_id(query, source)
            frames.append(
                feature_frame(
                    query_id=query_id,
                    query=query,
                    query_source=source,
                    candidate_rank_start=1,
                    batch=batch,
                )
            )
            query_records.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "query_source": source,
                    "candidate_count": len(batch.product_ids),
                    "candidate_k": depth,
                }
            )
            session.rollback()

        import pandas as pd

        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        parquet_path = out_dir / "diagnostic_features.parquet"
        write_feature_parquet(parquet_path, table)
        matrix = table[list(FEATURE_NAMES)].to_numpy(dtype=np.float32) if not table.empty else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        stats = _feature_stats(matrix) if matrix.size else {}
        constant = [name for name, row in stats.items() if row["std"] == 0.0]

        false_friend = None
        if FALSE_FRIEND_QUERY in issued:
            fused, depth = collect_fused_candidates(
                session,
                FALSE_FRIEND_QUERY,
                candidate_k=args.candidate_k,
                top_k=args.top_k,
            )
            products = get_products_by_ids(session, [row.product_id for row in fused])
            views = {pid: ProductView.from_product(item) for pid, item in products.items()}
            batch = extract_features(
                FALSE_FRIEND_QUERY,
                fused,
                views,
                candidate_k=depth,
                rrf_k=settings.hybrid_rrf_k,
                keyword_weight=settings.hybrid_keyword_weight,
            )
            interesting = [
                "keyword_present",
                "semantic_present",
                "keyword_score",
                "semantic_score",
                "keyword_rr",
                "semantic_rr",
                "rrf_score",
                "weighted_score",
                "title_exact_phrase",
                "title_all_query_tokens",
                "title_overlap_ratio",
                "brand_present",
                "description_missing",
            ]
            by_id = {pid: i for i, pid in enumerate(batch.product_ids)}
            examples = []
            for product_id in FALSE_FRIEND_IDS:
                index = by_id.get(product_id)
                if index is None:
                    continue
                product = views[product_id]
                row = batch.values[index]
                examples.append(
                    {
                        "product_id": product_id,
                        "title": product.title,
                        "features": {name: float(row[FEATURE_NAMES.index(name)]) for name in interesting},
                    }
                )
            false_friend = {
                "query": FALSE_FRIEND_QUERY,
                "note": "Inspection only. Features are not used to rerank in this phase.",
                "examples": examples,
            }

        if args.register_artifact:
            upsert_artifact_version(
                session,
                artifact_id=f"ranking-features:{FEATURE_VERSION}",
                artifact_type="ranking_features",
                version=FEATURE_VERSION,
                dataset_version=DATASET_VERSION_FULL,
                embedding_model_name=runtime.model_name,
                embedding_dim=runtime.embedding_dim,
                metric=None,
                path=str(parquet_path),
                metadata={"feature_count": len(FEATURE_NAMES), "row_count": int(len(table))},
            )
            session.commit()

    profile = {
        "feature_version": FEATURE_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "row_count": int(len(table)),
        "query_count": len(query_records),
        "issued_query_count": sum(1 for row in query_records if row["query_source"] == "issued"),
        "synthetic_query_count": sum(
            1 for row in query_records if row["query_source"] == "synthetic_title"
        ),
        "candidate_k": args.candidate_k,
        "mean_candidates_per_query": round(float(statistics.mean(candidate_counts)), 3)
        if candidate_counts
        else None,
        "finite": bool(np.isfinite(matrix).all()) if matrix.size else True,
        "constant_features": constant,
        "distributions": stats,
        "extraction_latency_ms": _summary(extract_ms) if extract_ms else None,
        "metadata_fetch_ms": _summary(fetch_ms) if fetch_ms else None,
        "false_friend_inspection": false_friend,
        "note": "Diagnostic only. Not NDCG, not feature importance, not a ranker.",
    }
    manifest = {
        "feature_version": FEATURE_VERSION,
        "dataset_version": DATASET_VERSION_FULL,
        "semantic_artifact_version": runtime.artifact_version,
        "semantic_backend": runtime.backend,
        "product_count": runtime.ntotal,
        "candidate_k": args.candidate_k,
        "hybrid_rrf_k": settings.hybrid_rrf_k,
        "hybrid_keyword_weight": settings.hybrid_keyword_weight,
        "query_count": len(query_records),
        "candidate_row_count": int(len(table)),
        "feature_count": len(FEATURE_NAMES),
        "feature_names_sha256": __import__("hashlib")
        .sha256("\n".join(FEATURE_NAMES).encode("utf-8"))
        .hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "labels_present": False,
        "query_source_classes": sorted({row["query_source"] for row in query_records}),
        "paths": {
            "parquet": str(parquet_path),
            "profile": str(out_dir / "profile.json"),
            "schema": "docs/ranking_feature_schema.json",
        },
        "parquet_sha256": sha256_file(parquet_path) if parquet_path.is_file() else None,
        "parquet_bytes": parquet_path.stat().st_size if parquet_path.is_file() else None,
    }
    (out_dir / "profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "schema.json").write_text(
        json.dumps(schema_document(), indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "feature_version": FEATURE_VERSION,
                "queries": len(query_records),
                "rows": int(len(table)),
                "features": len(FEATURE_NAMES),
                "parquet_bytes": manifest["parquet_bytes"],
                "constant_features": constant,
                "extraction_latency_ms": profile["extraction_latency_ms"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
