"""Shared offline hybrid scoring. Not imported by the API."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.versioning import embedding_paths
from app.models.interaction import Interaction
from app.ranking.metrics import macro_average
from app.recommendations.cf_model import BPRMatrixFactorization
from app.recommendations.evaluation import rank_to_metrics
from app.recommendations.hybrid_candidates import topk_ids_from_scores
from app.recommendations.hybrid_eval import (
    coverage_flags,
    fuse_channel_lists,
    grid_weights,
    rank_hidden_in_fused,
)
from app.recommendations.hybrid_fusion import validate_channel_weights
from app.recommendations.hybrid_types import ChannelWeights
from app.recommendations.ranking import ScoredItem
from app.search.runtime import require_semantic_runtime


def load_catalog_embeddings(settings) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    from pathlib import Path

    paths = embedding_paths(Path(settings.artifacts_root), settings.semantic_artifact_version)
    product_ids = np.load(paths["product_ids"], allow_pickle=False)
    if paths["embeddings"].is_file():
        embeddings = np.asarray(np.load(paths["embeddings"], mmap_mode="r"), dtype=np.float32)
    else:
        runtime = require_semantic_runtime()
        embeddings = np.vstack(
            [np.asarray(runtime.index.reconstruct(i), dtype=np.float32) for i in range(runtime.ntotal)]
        )
        product_ids = runtime.product_ids
    id_to_row = {str(product_id): index for index, product_id in enumerate(product_ids.tolist())}
    return product_ids, embeddings, id_to_row


def interaction_counts_excluding(session: Session, blocked: set[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for user_id, product_id in session.execute(select(Interaction.user_id, Interaction.product_id)):
        if (str(user_id), str(product_id)) in blocked:
            continue
        counts[str(product_id)] += 1
    return dict(counts)


def exclude_row_indices(exclude: set[str], id_to_row: dict[str, int]) -> np.ndarray:
    return np.fromiter(
        (id_to_row[product_id] for product_id in exclude if product_id in id_to_row),
        dtype=np.int64,
        count=-1,
    )


def mean_profile(embeddings: np.ndarray, rows: np.ndarray) -> np.ndarray | None:
    if rows.size == 0:
        return None
    vector = np.mean(embeddings[rows], axis=0).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return None
    return vector / np.float32(norm)


def content_candidates_from_scores(
    scores: np.ndarray,
    product_ids: np.ndarray,
    *,
    exclude: set[str],
    top_k: int,
    exclude_indices: np.ndarray | None = None,
) -> list[ScoredItem]:
    return topk_ids_from_scores(
        product_ids,
        scores,
        exclude=exclude,
        top_k=top_k,
        exclude_indices=exclude_indices,
    )


def cf_candidates_from_scores(
    scores: np.ndarray,
    product_ids: Sequence[str],
    *,
    exclude: set[str],
    top_k: int,
    exclude_indices: np.ndarray | None = None,
) -> list[ScoredItem]:
    return topk_ids_from_scores(
        product_ids,
        scores,
        exclude=exclude,
        top_k=top_k,
        exclude_indices=exclude_indices,
    )


def popularity_ranking(counts: dict[str, int]) -> list[tuple[str, float]]:
    return sorted(((product_id, float(count)) for product_id, count in counts.items()), key=lambda row: (-row[1], row[0]))


def popularity_candidates_from_counts(
    counts: dict[str, int],
    *,
    exclude: set[str],
    top_k: int,
    ranked: list[tuple[str, float]] | None = None,
) -> list[ScoredItem]:
    ordered = ranked if ranked is not None else popularity_ranking(counts)
    items: list[ScoredItem] = []
    for product_id, score in ordered:
        if product_id in exclude:
            continue
        items.append(ScoredItem(product_id=product_id, score=score))
        if len(items) >= top_k:
            break
    return items


def method_metrics(
    *,
    content: list[ScoredItem] | None,
    cf: list[ScoredItem] | None,
    popularity: list[ScoredItem] | None,
    hidden_id: str,
    catalog_size: int,
    rrf_k: int,
    weighted_configs: Sequence[tuple[str, ChannelWeights]],
) -> dict[str, Any]:
    flags = coverage_flags(hidden_id, content, cf, popularity)
    payload: dict[str, Any] = {"coverage": flags}
    payload["content"] = rank_to_metrics(_rank_in_list(hidden_id, content), catalog_size)
    payload["cf"] = rank_to_metrics(_rank_in_list(hidden_id, cf), catalog_size)
    payload["popularity"] = rank_to_metrics(_rank_in_list(hidden_id, popularity), catalog_size)
    rrf = fuse_channel_lists(content, cf, popularity, fusion_method="rrf", rrf_k=rrf_k)
    payload["rrf"] = rank_to_metrics(rank_hidden_in_fused(hidden_id, rrf), catalog_size)
    payload["weighted"] = {}
    for name, weights in weighted_configs:
        ranked = fuse_channel_lists(
            content,
            cf,
            popularity,
            fusion_method="weighted",
            rrf_k=rrf_k,
            weights=weights,
        )
        payload["weighted"][name] = rank_to_metrics(rank_hidden_in_fused(hidden_id, ranked), catalog_size)
    return payload


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return macro_average(rows) if rows else {}


def score_cf_users(
    model: BPRMatrixFactorization,
    user_indices: Sequence[int],
    *,
    batch_size: int = 256,
) -> np.ndarray:
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(user_indices), batch_size):
            batch = torch.tensor(list(user_indices[start : start + batch_size]), dtype=torch.long, device=device)
            chunks.append(model.score_items(batch).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, model.n_items), dtype=np.float32)


def _rank_in_list(hidden_id: str, items: list[ScoredItem] | None) -> int | None:
    if items is None:
        return None
    target = str(hidden_id)
    for index, item in enumerate(items, start=1):
        if item.product_id == target:
            return index
    return None


def default_weighted_grid() -> list[tuple[str, ChannelWeights]]:
    return grid_weights()


def selected_weights(name: str) -> ChannelWeights:
    for label, weights in grid_weights():
        if label == name:
            return weights
    if name == "equal":
        return validate_channel_weights(1 / 3, 1 / 3, 1 / 3)
    raise KeyError(name)
