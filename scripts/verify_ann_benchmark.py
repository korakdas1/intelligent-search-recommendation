#!/usr/bin/env python3
"""Verify the existing E-004 ANN benchmark artifact. Does not rebuild HNSW unless required."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.embeddings.versioning import embedding_paths, index_paths
from app.evaluation.constants import ANN_SOURCE_EXPERIMENT_ID, OFFICIAL_COMPARISON_DIR
from app.evaluation.io import write_json_atomic

REQUIRED_FIELDS = (
    "metric_name",
    "metric_definition",
    "not_search_relevance",
    "exact_index_type",
    "ann_index_type",
    "hnsw",
    "query_count",
    "k",
    "exact_latency_ms",
    "hnsw_by_ef_search",
    "build_durations_seconds",
    "file_sizes_bytes",
    "selected_ef_search",
)

LOGGED = {
    "exact_median_ms": 17.897,
    "hnsw_128_recall": 0.920,
    "hnsw_128_median_ms": 0.493,
    "exact_build_s": 0.075,
    "hnsw_build_s": 25.068,
    "exact_bytes": 172919853,
    "hnsw_bytes": 203568762,
    "query_count": 80,
    "k": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artifacts/evaluation/E-004/ann_benchmark.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Unused unless verification fails")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    source = Path(args.source)
    out_dir = Path(args.output) if args.output else Path(settings.artifacts_root) / "evaluation" / OFFICIAL_COMPARISON_DIR
    out_path = out_dir / "ann_verification.json"
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite {out_path}; pass --force", file=sys.stderr)
        return 2
    if not source.is_file():
        print(f"missing E-004 artifact {source}; rerun scripts/benchmark_ann.py", file=sys.stderr)
        return 2
    payload = json.loads(source.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        print(f"E-004 missing fields: {missing}", file=sys.stderr)
        return 1
    if payload.get("exact_index_type") != "IndexFlatIP":
        print("E-004 exact index is not IndexFlatIP", file=sys.stderr)
        return 1
    if payload.get("ann_index_type") != "IndexHNSWFlat":
        print("E-004 ANN index is not IndexHNSWFlat", file=sys.stderr)
        return 1
    hnsw = payload.get("hnsw") or {}
    selected = int(payload["selected_ef_search"])
    selected_row = next(
        row for row in payload["hnsw_by_ef_search"] if int(row["efSearch"]) == selected
    )
    paths = index_paths(Path(settings.artifacts_root), payload["artifact_version"])
    embed_paths = embedding_paths(Path(settings.artifacts_root), payload["artifact_version"])
    files = {
        "flat.faiss": paths["flat"],
        "hnsw.faiss": paths["hnsw"],
        "embeddings.npy": embed_paths["embeddings"],
    }
    existing = {name: str(path) for name, path in files.items() if path.is_file()}
    sizes_ok = (
        int(payload["file_sizes_bytes"]["flat.faiss"]) == LOGGED["exact_bytes"]
        and int(payload["file_sizes_bytes"]["hnsw.faiss"]) == LOGGED["hnsw_bytes"]
    )
    recall_ok = abs(float(selected_row["ann_neighbor_recall_at_k"]) - LOGGED["hnsw_128_recall"]) < 1e-6
    verification = {
        "source_experiment": ANN_SOURCE_EXPERIMENT_ID,
        "reran": False,
        "artifact": str(source),
        "required_fields_present": not missing,
        "exact_index_type": payload["exact_index_type"],
        "ann_index_type": payload["ann_index_type"],
        "hnsw_parameters": hnsw,
        "k": payload["k"],
        "query_count": payload["query_count"],
        "exact_latency_ms": payload["exact_latency_ms"],
        "hnsw_selected_ef_search": selected,
        "hnsw_neighbor_recall_at_k": selected_row["ann_neighbor_recall_at_k"],
        "hnsw_latency_ms": selected_row["latency_ms"],
        "hnsw_by_ef_search": payload["hnsw_by_ef_search"],
        "build_durations_seconds": payload["build_durations_seconds"],
        "file_sizes_bytes": payload["file_sizes_bytes"],
        "files_present": existing,
        "reconciles_experiment_log": recall_ok and sizes_ok and int(payload["query_count"]) == LOGGED["query_count"],
        "metric_is_neighbor_overlap_not_search_relevance": True,
        "serving_recommendation": payload.get("serving_recommendation", "flat"),
        "serving_reason": payload.get("serving_reason"),
        "created_at": datetime.now(UTC).isoformat(),
        "note": "Verified existing E-004. No new ANN experiment. No HNSW rebuild.",
    }
    write_json_atomic(out_path, verification)
    print(json.dumps(verification, indent=2))
    if not verification["reconciles_experiment_log"]:
        print("E-004 does not reconcile with EXPERIMENT_LOG numbers", file=sys.stderr)
        return 1
    if args.rebuild:
        print("--rebuild ignored because verification succeeded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
