#!/usr/bin/env python3
"""Evaluate a trained RankNet against hybrid baselines on the frozen test split. No training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from app.ranking.artifacts import load_ranker_bundle
from app.ranking.constants import DATASET_VERSION, MODEL_VERSION
from app.ranking.feature_schema import FEATURE_NAMES
from app.ranking.inference import score_feature_matrix
from app.ranking.metrics import macro_average, metric_bundle
from app.ranking.pairwise import pairwise_accuracy, ranknet_loss, sample_hard_negative_pairs
from app.ranking.preprocessing import FeatureScaler
import torch


def _rank_ids(group: pd.DataFrame, scores: np.ndarray) -> list[str]:
    order = np.lexsort((group["product_id"].to_numpy(), -np.asarray(scores, dtype=np.float64)))
    return [str(pid) for pid in group["product_id"].to_numpy()[order]]


def _baseline_scores(group: pd.DataFrame, column: str) -> np.ndarray:
    return group[column].to_numpy(dtype=np.float32)


def _pairwise_from_frame(frame: pd.DataFrame, model, scaler: FeatureScaler, split: str) -> dict:
    subset = frame[frame["split"] == split]
    pos_scores = []
    neg_scores = []
    for _, group in subset.groupby("query_id", sort=True):
        reset = group.reset_index(drop=True)
        pairs = sample_hard_negative_pairs(reset)
        if not pairs:
            continue
        matrix = reset.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
        scores = score_feature_matrix(model, scaler, matrix)
        for left, right in pairs:
            pos_scores.append(float(scores[left]))
            neg_scores.append(float(scores[right]))
    if not pos_scores:
        return {"pair_count": 0}
    pos = torch.tensor(pos_scores, dtype=torch.float32)
    neg = torch.tensor(neg_scores, dtype=torch.float32)
    return {
        "pair_count": len(pos_scores),
        "ranknet_loss": float(ranknet_loss(pos, neg).item()),
        "pairwise_accuracy": float(pairwise_accuracy(pos, neg).item()),
    }


def _population_metrics(groups: list[pd.DataFrame], model, scaler: FeatureScaler) -> dict:
    rrf_rows = []
    weighted_rows = []
    ltr_rows = []
    covered = 0
    extract_ms = []
    for group in groups:
        relevant = group.loc[group["relevance"] > 0, "product_id"].astype(str).tolist()
        if any(rel in set(group["product_id"].astype(str)) for rel in relevant):
            covered += 1
        rrf_ids = _rank_ids(group, _baseline_scores(group, "rrf_score"))
        weighted_ids = _rank_ids(group, _baseline_scores(group, "weighted_score"))
        started = time.perf_counter()
        ltr_scores = score_feature_matrix(
            model, scaler, group.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
        )
        extract_ms.append((time.perf_counter() - started) * 1000)
        ltr_ids = _rank_ids(group, ltr_scores)
        rrf_rows.append(metric_bundle(rrf_ids, relevant))
        weighted_rows.append(metric_bundle(weighted_ids, relevant))
        ltr_rows.append(metric_bundle(ltr_ids, relevant))
    return {
        "query_count": len(groups),
        "covered_query_count": covered,
        "rrf": macro_average(rrf_rows),
        "weighted": macro_average(weighted_rows),
        "ltr": macro_average(ltr_rows),
        "ltr_score_ms": {
            "median": float(np.median(extract_ms)) if extract_ms else None,
            "p95": float(np.percentile(extract_ms, 95)) if extract_ms else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=f"artifacts/ltr/{DATASET_VERSION}")
    parser.add_argument("--model-dir", default=f"artifacts/models/{MODEL_VERSION}")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    frame = pd.read_parquet(dataset_dir / "candidates.parquet")
    queries = json.loads((dataset_dir / "queries.json").read_text(encoding="utf-8"))
    model, scaler, config = load_ranker_bundle(Path(args.model_dir))
    split_frame = frame[frame["split"] == args.split]
    groups = [group for _, group in split_frame.groupby("query_id", sort=True)]
    covered_groups = [
        group for group in groups if int((group["relevance"] > 0).sum()) > 0
    ]
    all_metrics = _population_metrics(groups, model, scaler)
    covered_metrics = _population_metrics(covered_groups, model, scaler)
    pairwise = {
        "validation": _pairwise_from_frame(frame, model, scaler, "validation"),
        "test": _pairwise_from_frame(frame, model, scaler, "test"),
    }

    false_friend = None
    match = split_frame[split_frame["query"] == "product for restoring leather"]
    if match.empty:
        match = frame[frame["query"] == "product for restoring leather"]
    if not match.empty:
        group = match
        scores = score_feature_matrix(
            model, scaler, group.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
        )
        ranked = _rank_ids(group, scores)
        rrf_ranked = _rank_ids(group, _baseline_scores(group, "rrf_score"))
        false_friend = {
            "query": "product for restoring leather",
            "split": str(group["split"].iloc[0]),
            "in_evaluation_split": bool((group["split"] == args.split).all()),
            "rrf_top5": rrf_ranked[:5],
            "ltr_top5": ranked[:5],
            "note": "Issued diagnostic query; not necessarily in the synthetic test split.",
        }

    unique_queries = split_frame.drop_duplicates("query_id")
    payload = {
        "label_class": "synthetic",
        "split": args.split,
        "model_version": config.get("model_version"),
        "feature_version": config.get("feature_version"),
        "dataset_version": json.loads((dataset_dir / "manifest.json").read_text())["dataset_version"]
        if (dataset_dir / "manifest.json").is_file()
        else None,
        "candidate_policy": "same hybrid union / candidate_k for RRF, weighted, and LTR",
        "all_queries": all_metrics,
        "covered_queries": covered_metrics,
        "pairwise": pairwise,
        "false_friend": false_friend,
        "query_family_counts": unique_queries["query_source"].value_counts().to_dict(),
        "test_query_manifest_count": sum(1 for row in queries if row["split"] == args.split),
        "note": (
            "Synthetic held-out retrieval/reranking evaluation. "
            "Not human relevance."
        ),
    }

    out_path = Path(args.out or Path(args.model_dir) / "evaluation.json")
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "all_queries": all_metrics,
            "covered_queries": covered_metrics,
            "pairwise": pairwise,
            "false_friend": false_friend,
            "path": str(out_path),
        },
        indent=2,
        default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
