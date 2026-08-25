"""Parquet helpers for ranking-feature diagnostic rows. Not training labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.ranking.feature_schema import FEATURE_NAMES, FEATURE_VERSION
from app.ranking.features import FeatureBatch
from app.ranking.validate import validate_feature_batch


def feature_frame(
    *,
    query_id: str,
    query: str,
    query_source: str,
    candidate_rank_start: int,
    batch: FeatureBatch,
) -> pd.DataFrame:
    validate_feature_batch(batch)
    rows = {
        "query_id": query_id,
        "query": query,
        "query_source": query_source,
        "product_id": list(batch.product_ids),
        "candidate_rank": list(
            range(candidate_rank_start, candidate_rank_start + len(batch.product_ids))
        ),
        "feature_version": batch.feature_version,
    }
    for index, name in enumerate(FEATURE_NAMES):
        rows[name] = batch.values[:, index].astype(np.float32, copy=False)
    return pd.DataFrame(rows)


def write_feature_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def read_feature_parquet(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "feature_version" in frame.columns and not frame.empty:
        versions = set(frame["feature_version"].astype(str))
        if versions != {FEATURE_VERSION}:
            raise ValueError(f"unexpected feature_version in parquet: {versions}")
    return frame
