"""BPR-MF training loop. Validation ranking selects the checkpoint; never the external test."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.optim import Adam

from app.ranking.metrics import macro_average
from app.ranking.reproducibility import set_ltr_seeds
from app.recommendations.cf_loss import bpr_objective, bpr_ranking_loss
from app.recommendations.cf_model import BPRMatrixFactorization
from app.recommendations.cf_sampling import UniformNegativeSampler
from app.recommendations.evaluation import rank_to_metrics


def resolve_device(requested: str) -> torch.device:
    name = requested.strip().lower()
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unsupported device {requested!r}")


def rank_hidden_from_scores(
    scores: np.ndarray,
    product_ids: np.ndarray,
    *,
    seen_indices: Sequence[int],
    hidden_index: int,
) -> int:
    """1-based rank among trained CF items. Seen items are excluded. Cold catalog items are not scored."""

    working = np.array(scores, dtype=np.float64, copy=True)
    if seen_indices:
        seen = np.fromiter((int(index) for index in seen_indices if int(index) != int(hidden_index)), dtype=np.int64)
        if seen.size:
            working[seen] = -np.inf
    hidden_score = float(working[hidden_index])
    hidden_id = str(product_ids[hidden_index])
    better = (working > hidden_score) | ((working == hidden_score) & (product_ids < hidden_id))
    return int(np.count_nonzero(better)) + 1


def evaluate_cf_ranking(
    model: BPRMatrixFactorization,
    *,
    user_indices: Sequence[int],
    hidden_item_indices: Sequence[int | None],
    seen_item_indices: Sequence[Sequence[int]],
    product_ids: Sequence[str],
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Rank each hidden item among trained CF items (untrained catalog items unscored)."""

    model.eval()
    ids = np.asarray(list(product_ids))
    rows: list[dict[str, float]] = []
    covered = 0
    misses_cold = 0
    with torch.no_grad():
        for start in range(0, len(user_indices), batch_size):
            stop = start + batch_size
            batch_users = torch.tensor(list(user_indices[start:stop]), dtype=torch.long, device=device)
            scores = model.score_items(batch_users).detach().cpu().numpy()
            for offset, (hidden, seen) in enumerate(
                zip(hidden_item_indices[start:stop], seen_item_indices[start:stop], strict=True)
            ):
                if hidden is None:
                    rows.append(rank_to_metrics(None, model.n_items))
                    misses_cold += 1
                    continue
                covered += 1
                rank = rank_hidden_from_scores(
                    scores[offset],
                    ids,
                    seen_indices=seen,
                    hidden_index=int(hidden),
                )
                rows.append(rank_to_metrics(rank, model.n_items))
    metrics = macro_average(rows) if rows else {}
    metrics["n_users"] = float(len(user_indices))
    metrics["n_covered"] = float(covered)
    metrics["n_cold_hidden"] = float(misses_cold)
    return metrics


def _validation_bpr_loss(
    model: BPRMatrixFactorization,
    sampler: UniformNegativeSampler,
    user_indices: np.ndarray,
    positive_indices: np.ndarray,
    *,
    seed: int,
    epoch: int,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(user_indices), batch_size):
            stop = min(start + batch_size, len(user_indices))
            users = user_indices[start:stop]
            positives = positive_indices[start:stop]
            negatives = sampler.sample_unseen_for_users(users, seed=seed, epoch=epoch)
            user_t = torch.tensor(users, dtype=torch.long, device=device)
            pos_t = torch.tensor(positives, dtype=torch.long, device=device)
            neg_t = torch.tensor(negatives, dtype=torch.long, device=device)
            pos_s = model.score(user_t, pos_t)
            neg_s = model.score(user_t, neg_t)
            losses.append(float(bpr_ranking_loss(pos_s, neg_s).item()))
    return float(sum(losses) / len(losses)) if losses else float("nan")


def train_bpr_mf(
    *,
    n_users: int,
    n_items: int,
    embedding_dim: int,
    train_user_index: np.ndarray,
    train_item_index: np.ndarray,
    user_positives: Sequence[set[int]],
    product_ids: Sequence[str],
    val_user_indices: Sequence[int] | None = None,
    val_hidden_item_indices: Sequence[int | None] | None = None,
    val_seen_item_indices: Sequence[Sequence[int]] | None = None,
    val_pairs_user: np.ndarray | None = None,
    val_pairs_item: np.ndarray | None = None,
    learning_rate: float,
    batch_size: int,
    l2: float,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device | str,
    negatives_per_positive: int = 1,
    fixed_epochs: int | None = None,
) -> tuple[BPRMatrixFactorization, dict[str, Any]]:
    """Train BPR-MF. If ``fixed_epochs`` is set, run that many epochs with no selection."""

    if negatives_per_positive != 1:
        raise ValueError("bpr-mf-v1 uses exactly one negative per positive")
    device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    set_ltr_seeds(seed)
    model = BPRMatrixFactorization(n_users, n_items, embedding_dim).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    sampler = UniformNegativeSampler(user_positives, n_items)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_ndcg = -1.0
    best_val_loss = float("inf")
    stale = 0
    epochs_to_run = int(fixed_epochs) if fixed_epochs is not None else int(max_epochs)
    n_pairs = len(train_user_index)
    if n_pairs == 0:
        raise ValueError("no training pairs")

    for epoch in range(1, epochs_to_run + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(n_pairs)
        train_losses: list[float] = []
        for start in range(0, n_pairs, batch_size):
            chosen = order[start : start + batch_size]
            users = train_user_index[chosen]
            positives = train_item_index[chosen]
            negatives = sampler.sample(users, positives, seed=seed, epoch=epoch)
            user_t = torch.tensor(users, dtype=torch.long, device=device)
            pos_t = torch.tensor(positives, dtype=torch.long, device=device)
            neg_t = torch.tensor(negatives, dtype=torch.long, device=device)
            user_vec = model.user_embedding(user_t)
            pos_vec = model.item_embedding(pos_t)
            neg_vec = model.item_embedding(neg_t)
            pos_s = (user_vec * pos_vec).sum(dim=-1)
            neg_s = (user_vec * neg_vec).sum(dim=-1)
            total, ranking, _penalty = bpr_objective(
                pos_s,
                neg_s,
                user_vectors=user_vec,
                positive_vectors=pos_vec,
                negative_vectors=neg_vec,
                l2=l2,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            train_losses.append(float(ranking.item()))
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(sum(train_losses) / len(train_losses)),
        }
        if val_user_indices is not None and fixed_epochs is None:
            val_metrics = evaluate_cf_ranking(
                model,
                user_indices=val_user_indices,
                hidden_item_indices=val_hidden_item_indices or [],
                seen_item_indices=val_seen_item_indices or [],
                product_ids=product_ids,
                device=device,
            )
            val_loss = float("nan")
            if val_pairs_user is not None and val_pairs_item is not None and len(val_pairs_user):
                val_loss = _validation_bpr_loss(
                    model,
                    sampler,
                    val_pairs_user,
                    val_pairs_item,
                    seed=seed,
                    epoch=epoch,
                    device=device,
                    batch_size=batch_size,
                )
            ndcg = float(val_metrics.get("ndcg@10", 0.0))
            row.update(
                {
                    "validation_loss": val_loss,
                    "validation_R@10": float(val_metrics.get("recall@10", 0.0)),
                    "validation_NDCG@10": ndcg,
                    "validation_MRR": float(val_metrics.get("mrr", 0.0)),
                }
            )
            improved = ndcg > best_ndcg + 1e-12 or (
                abs(ndcg - best_ndcg) <= 1e-12 and val_loss < best_val_loss
            )
            if improved:
                best_ndcg = ndcg
                best_val_loss = val_loss
                best_epoch = epoch
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    history.append(row)
                    break
        history.append(row)

    if fixed_epochs is None and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    elif fixed_epochs is None:
        best_epoch = epochs_to_run
    else:
        best_epoch = epochs_to_run

    summary = {
        "best_epoch": best_epoch,
        "best_validation_ndcg@10": None if fixed_epochs is not None else best_ndcg,
        "best_validation_loss": None if fixed_epochs is not None else best_val_loss,
        "epochs_run": history[-1]["epoch"] if history else 0,
        "history": history,
        "embedding_dim": embedding_dim,
        "n_users": n_users,
        "n_items": n_items,
        "train_pairs": int(n_pairs),
        "parameter_count": model.parameter_count(),
        "device": str(device),
        "selection_metric": "ndcg@10",
        "note": "External recsys-eval-v1 test was not used for selection.",
    }
    return model, summary
