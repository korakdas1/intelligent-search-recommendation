#!/usr/bin/env python3
"""Compare HNSW ANN neighbors to exact IndexFlatIP.

ANNRecall@K is neighbor overlap with exact FAISS, not search-relevance Recall@K.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import get_session_factory, reset_engine
from app.embeddings.normalize import l2_normalize
from app.embeddings.versioning import embedding_paths, index_paths
from app.models.product import Product
from app.search.ann_metrics import mean_neighbor_recall_at_k
from app.search.faiss_index import load_index, search_index
from app.search.runtime import load_semantic_runtime
from app.search.semantic import _ensure_encoder

NATURAL_QUERIES = [
    "product for restoring leather",
    "sun protection lotion",
    "hair washing product for dry hair",
    "face moisturizer for sensitive skin",
    "product that reduces frizz",
    "leather conditioner",
    "gentle cleanser for sensitive skin",
    "leave-in treatment for dry hair",
    "mineral sunscreen for face",
    "cuticle oil",
]


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query-count", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--ef-search", default="16,32,64,128")
    parser.add_argument("--out", default="artifacts/evaluation/E-004/ann_benchmark.json")
    args = parser.parse_args(argv)
    ef_values = [int(part.strip()) for part in args.ef_search.split(",") if part.strip()]

    settings = get_settings()
    reset_engine()
    runtime = load_semantic_runtime(settings)
    encoder = _ensure_encoder(runtime)
    paths = index_paths(Path(settings.artifacts_root), runtime.artifact_version)
    embed_paths = embedding_paths(Path(settings.artifacts_root), runtime.artifact_version)
    flat = load_index(paths["flat"])
    hnsw = load_index(paths["hnsw"])
    factory = get_session_factory()

    queries: list[str] = list(NATURAL_QUERIES)
    with factory() as session:
        needed = max(0, args.query_count - len(queries))
        titles = list(
            session.scalars(
                select(Product.title)
                .where(Product.dataset_version == runtime.dataset_version)
                .order_by(Product.product_id.asc())
                .limit(needed)
            )
        )
        queries.extend(titles)

    query_vectors = l2_normalize(encoder.encode(queries, batch_size=32, show_progress=True))
    exact_ids: list[list[int]] = []
    for vector in query_vectors:
        _scores, indices = search_index(flat, vector, args.k)
        exact_ids.append([int(row) for row in indices[0] if int(row) >= 0])

    # Warmup both indexes
    for vector in query_vectors[: args.warmup]:
        search_index(flat, vector, args.k)
        search_index(hnsw, vector, args.k, ef_search=ef_values[-1])

    exact_times: list[float] = []
    for _ in range(args.repeats):
        for vector in query_vectors:
            started = time.perf_counter()
            search_index(flat, vector, args.k)
            exact_times.append((time.perf_counter() - started) * 1000)

    configs = []
    for ef_search in ef_values:
        ann_ids: list[list[int]] = []
        ann_times: list[float] = []
        for vector in query_vectors:
            _scores, indices = search_index(hnsw, vector, args.k, ef_search=ef_search)
            ann_ids.append([int(row) for row in indices[0] if int(row) >= 0])
        recall = mean_neighbor_recall_at_k(exact_ids, ann_ids, args.k)
        for _ in range(args.repeats):
            for vector in query_vectors:
                started = time.perf_counter()
                search_index(hnsw, vector, args.k, ef_search=ef_search)
                ann_times.append((time.perf_counter() - started) * 1000)
        configs.append(
            {
                "efSearch": ef_search,
                "ann_neighbor_recall_at_k": round(recall, 6),
                "k": args.k,
                "latency_ms": _summary(ann_times),
            }
        )

    embedding_manifest = json.loads(embed_paths["manifest"].read_text(encoding="utf-8"))
    index_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    report = {
        "metric_name": "ANN neighbor Recall@K",
        "metric_definition": "|ANN_K ∩ Exact_K| / K averaged over queries",
        "not_search_relevance": True,
        "artifact_version": runtime.artifact_version,
        "exact_index_type": "IndexFlatIP",
        "ann_index_type": "IndexHNSWFlat",
        "hnsw": index_manifest.get("hnsw"),
        "query_count": len(queries),
        "natural_language_queries": NATURAL_QUERIES,
        "k": args.k,
        "exact_latency_ms": _summary(exact_times),
        "hnsw_by_ef_search": configs,
        "build_durations_seconds": index_manifest.get("durations_seconds"),
        "file_sizes_bytes": {
            "embeddings.npy": embedding_manifest.get("file_sizes_bytes", {}).get("embeddings.npy"),
            "flat.faiss": index_manifest.get("file_sizes_bytes", {}).get("flat.faiss"),
            "hnsw.faiss": index_manifest.get("file_sizes_bytes", {}).get("hnsw.faiss"),
        },
        "product_count": runtime.ntotal,
    }
    # Select a serving recommendation: prefer high neighbor recall without
    # erasing the speed gain. Exact search may still win at this catalog size.
    best = max(configs, key=lambda row: (row["ann_neighbor_recall_at_k"], -row["latency_ms"]["median"]))
    exact_median = report["exact_latency_ms"]["median"]
    report["selected_ef_search"] = best["efSearch"]
    report["serving_recommendation"] = (
        "flat"
        if exact_median <= best["latency_ms"]["median"] * 1.5
        or best["ann_neighbor_recall_at_k"] < 0.99
        else "hnsw"
    )
    report["serving_reason"] = (
        "Exact IndexFlatIP remains the serving default when it is already fast "
        "on this ~112k catalog or when ANN recall is not near-exact. HNSW is "
        "still implemented and measured (D-019)."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("exact_latency_ms", "hnsw_by_ef_search", "serving_recommendation")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
