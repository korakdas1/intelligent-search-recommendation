#!/usr/bin/env python3
"""Build a synthetic grouped LTR dataset from hybrid candidates. Does not train."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.core.config import get_settings
from app.core.events import DATASET_VERSION_FULL
from app.db.repositories.products import get_products_by_ids
from app.db.session import get_session_factory, reset_engine
from app.evaluation.provenance import ltr_artifact_checksums
from app.models.product import Product
from app.ranking.constants import DATASET_VERSION, DEFAULT_SPLIT_RATIOS, LABEL_CLASS
from app.ranking.feature_schema import FEATURE_NAMES, FEATURE_NAMES_SHA256, FEATURE_VERSION
from app.ranking.features import extract_features
from app.ranking.labels import make_attribute_query, make_title_query
from app.ranking.product_view import ProductView
from app.ranking.retrieval import collect_fused_candidates
from app.ranking.split import assert_split_disjoint, grouped_source_split
from app.search.runtime import load_semantic_runtime, reset_semantic_runtime, set_semantic_runtime

SOURCE_ORDER_POLICY = "product_id_asc_before_seeded_shuffle"


def select_source_products(
    eligible: list[tuple[str, str, str | None]], *, seed: int, max_sources: int,
) -> list[tuple[str, str, str | None]]:
    ordered = sorted(eligible, key=lambda row: row[0])
    random.Random(seed).shuffle(ordered)
    return ordered[:max_sources]


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sources", type=int, default=2500)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--no-attribute", action="store_true")
    parser.add_argument("--output-version", default=DATASET_VERSION)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    include_attribute = not args.no_attribute

    settings = get_settings()
    out_dir = Path(args.out_dir or f"artifacts/ltr/{args.output_version}")
    if out_dir.exists() and not args.force:
        print(f"refusing to overwrite {out_dir}; pass --force", file=sys.stderr)
        return 2

    reset_engine()
    reset_semantic_runtime()
    runtime = load_semantic_runtime(settings)
    set_semantic_runtime(runtime)
    factory = get_session_factory()

    with factory() as session:
        rows = list(
            session.execute(
                select(Product.product_id, Product.title, Product.brand).where(
                    Product.dataset_version == runtime.dataset_version
                )
            )
        )
        eligible: list[tuple[str, str, str | None]] = []
        skipped_title = 0
        for product_id, title, brand in rows:
            if make_title_query(str(product_id), title or "") is None:
                skipped_title += 1
                continue
            eligible.append((str(product_id), title or "", brand))
        selected = select_source_products(eligible, seed=args.seed, max_sources=args.max_sources)

        queries = []
        for product_id, title, brand in selected:
            title_q = make_title_query(product_id, title)
            if title_q is None:
                continue
            queries.append(title_q)
            if include_attribute:
                attr = make_attribute_query(product_id, title, brand)
                if attr is not None:
                    queries.append(attr)

        source_ids = sorted({item.source_product_id for item in queries})
        split_map = grouped_source_split(source_ids, seed=args.seed, ratios=DEFAULT_SPLIT_RATIOS)

        frames: list[pd.DataFrame] = []
        query_records: list[dict] = []
        covered = 0
        started_all = time.perf_counter()
        for index, spec in enumerate(queries, start=1):
            fused, depth = collect_fused_candidates(
                session, spec.query, candidate_k=args.candidate_k, top_k=10
            )
            products = get_products_by_ids(session, [row.product_id for row in fused])
            views = {pid: ProductView.from_product(item) for pid, item in products.items()}
            batch = extract_features(spec.query, fused, views, candidate_k=depth)
            relevant = set(spec.relevant_ids)
            has_positive = any(pid in relevant for pid in batch.product_ids)
            if has_positive:
                covered += 1
            record = {
                "query_id": spec.query_id,
                "query": spec.query,
                "query_source": spec.query_source,
                "label_class": spec.label_class,
                "source_product_id": spec.source_product_id,
                "split": split_map[spec.source_product_id],
                "candidate_count": len(batch.product_ids),
                "covered": bool(has_positive),
                "relevant_ids": list(spec.relevant_ids),
            }
            query_records.append(record)
            if not batch.product_ids:
                continue
            table = {
                "query_id": spec.query_id,
                "query": spec.query,
                "query_source": spec.query_source,
                "label_class": spec.label_class,
                "source_product_id": spec.source_product_id,
                "split": split_map[spec.source_product_id],
                "product_id": list(batch.product_ids),
                "relevance": [1 if pid in relevant else 0 for pid in batch.product_ids],
                "candidate_position": list(range(1, len(batch.product_ids) + 1)),
                "feature_version": batch.feature_version,
            }
            for col_index, name in enumerate(FEATURE_NAMES):
                table[name] = batch.values[:, col_index].astype(np.float32, copy=False)
            frames.append(pd.DataFrame(table))
            if index % 50 == 0 or index == len(queries):
                elapsed = time.perf_counter() - started_all
                print(f"built {index}/{len(queries)} queries in {elapsed:.1f}s", flush=True)

    if not frames:
        print("no candidate rows written", file=sys.stderr)
        return 1

    frame = pd.concat(frames, ignore_index=True)
    leak_sources = {
        name: {row["source_product_id"] for row in query_records if row["split"] == name}
        for name in ("train", "validation", "test")
    }
    leak_queries = {
        name: {row["query_id"] for row in query_records if row["split"] == name}
        for name in ("train", "validation", "test")
    }
    assert_split_disjoint(leak_sources, leak_queries)
    for name in FEATURE_NAMES:
        if name in {"relevance", "source_product_id", "query_source", "split", "candidate_position"}:
            raise RuntimeError("label column collided with a feature name")

    tmp = out_dir.parent / f".{out_dir.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    parquet_path = tmp / "candidates.parquet"
    frame.to_parquet(parquet_path, index=False)
    (tmp / "queries.json").write_text(json.dumps(query_records, indent=2) + "\n", encoding="utf-8")
    usable = [row for row in query_records if row["covered"]]
    train_q = [row for row in query_records if row["split"] == "train"]
    val_q = [row for row in query_records if row["split"] == "validation"]
    test_q = [row for row in query_records if row["split"] == "test"]
    manifest = {
        "dataset_version": args.output_version,
        "source_order_policy": SOURCE_ORDER_POLICY,
        "checksums": ltr_artifact_checksums(tmp),
        "label_class": LABEL_CLASS,
        "query_families": sorted({row["query_source"] for row in query_records}),
        "source_product_count": len(source_ids),
        "eligible_source_count": len(eligible),
        "skipped_unusable_title": skipped_title,
        "raw_query_count": len(query_records),
        "usable_query_count": len(usable),
        "queries_missing_positive": len(query_records) - len(usable),
        "candidate_coverage": (len(usable) / len(query_records)) if query_records else 0.0,
        "candidate_row_count": int(len(frame)),
        "candidate_k": args.candidate_k,
        "feature_version": FEATURE_VERSION,
        "feature_names_sha256": FEATURE_NAMES_SHA256,
        "semantic_artifact_version": runtime.artifact_version,
        "hybrid_rrf_k": settings.hybrid_rrf_k,
        "hybrid_keyword_weight": settings.hybrid_keyword_weight,
        "split_policy": "source_product_id grouped seeded random (synthetic labels only)",
        "split_ratios": list(DEFAULT_SPLIT_RATIOS),
        "seed": args.seed,
        "train_query_count": len(train_q),
        "validation_query_count": len(val_q),
        "test_query_count": len(test_q),
        "train_usable_query_count": sum(1 for row in train_q if row["covered"]),
        "pair_sampling_policy": "up to 8 RRF-top hard negatives per positive at train time",
        "catalog_dataset_version": DATASET_VERSION_FULL,
        "git_commit": _git_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "include_attribute": include_attribute,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    print(json.dumps({k: manifest[k] for k in (
        "dataset_version",
        "raw_query_count",
        "usable_query_count",
        "candidate_coverage",
        "candidate_row_count",
        "train_query_count",
        "validation_query_count",
        "test_query_count",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
