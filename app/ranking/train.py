"""Train RankNet from grouped candidate rows. Validation selects the checkpoint."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.ranking.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_NEGATIVES_PER_POSITIVE,
    DEFAULT_PATIENCE,
)
from app.ranking.pairwise import pairwise_accuracy, ranknet_loss, sample_hard_negative_pairs
from app.ranking.preprocessing import FeatureScaler
from app.ranking.ranker import RankNetMLP
from app.ranking.reproducibility import set_ltr_seeds


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


def _pairs_for_split(
    frame: pd.DataFrame,
    split: str,
    *,
    negatives_per_positive: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    subset = frame[frame["split"] == split]
    pos_rows: list[np.ndarray] = []
    neg_rows: list[np.ndarray] = []
    from app.ranking.feature_schema import FEATURE_NAMES

    for _, group in subset.groupby("query_id", sort=True):
        reset = group.reset_index(drop=True)
        pairs = sample_hard_negative_pairs(reset, negatives_per_positive=negatives_per_positive)
        if not pairs:
            continue
        matrix = reset.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
        for left, right in pairs:
            pos_rows.append(matrix[left])
            neg_rows.append(matrix[right])
    if not pos_rows:
        return None
    return np.stack(pos_rows), np.stack(neg_rows)


def _loader(
    positives: np.ndarray,
    negatives: np.ndarray,
    scaler: FeatureScaler,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    pos = torch.from_numpy(scaler.transform(positives))
    neg = torch.from_numpy(scaler.transform(negatives))
    dataset = TensorDataset(pos, neg)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def train_ranknet(
    frame: pd.DataFrame,
    scaler: FeatureScaler,
    *,
    hidden_sizes: Sequence[int],
    dropout: float,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    seed: int = 42,
    device: str = "cpu",
    negatives_per_positive: int = DEFAULT_NEGATIVES_PER_POSITIVE,
) -> tuple[RankNetMLP, dict[str, Any]]:
    set_ltr_seeds(seed)
    torch_device = resolve_device(device)
    train_pairs = _pairs_for_split(frame, "train", negatives_per_positive=negatives_per_positive)
    val_pairs = _pairs_for_split(frame, "validation", negatives_per_positive=negatives_per_positive)
    if train_pairs is None:
        raise ValueError("no training pairs")
    train_loader = _loader(
        train_pairs[0],
        train_pairs[1],
        scaler,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = None
    if val_pairs is not None:
        val_loader = _loader(
            val_pairs[0],
            val_pairs[1],
            scaler,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )

    model = RankNetMLP(hidden_sizes=hidden_sizes, dropout=dropout).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float | int]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_acc: list[float] = []
        for pos_batch, neg_batch in train_loader:
            pos_batch = pos_batch.to(torch_device)
            neg_batch = neg_batch.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            pos_scores = model(pos_batch)
            neg_scores = model(neg_batch)
            loss = ranknet_loss(pos_scores, neg_scores)
            if not torch.isfinite(loss):
                raise ValueError("non-finite RankNet loss")
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_acc.append(float(pairwise_accuracy(pos_scores.detach(), neg_scores.detach()).cpu()))
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(sum(train_losses) / len(train_losses)),
            "train_pairwise_accuracy": float(sum(train_acc) / len(train_acc)),
        }
        if val_loader is not None:
            model.eval()
            val_losses: list[float] = []
            val_acc: list[float] = []
            with torch.no_grad():
                for pos_batch, neg_batch in val_loader:
                    pos_batch = pos_batch.to(torch_device)
                    neg_batch = neg_batch.to(torch_device)
                    pos_scores = model(pos_batch)
                    neg_scores = model(neg_batch)
                    val_loss = ranknet_loss(pos_scores, neg_scores)
                    val_losses.append(float(val_loss.cpu()))
                    val_acc.append(float(pairwise_accuracy(pos_scores, neg_scores).cpu()))
            record["validation_loss"] = float(sum(val_losses) / len(val_losses))
            record["validation_pairwise_accuracy"] = float(sum(val_acc) / len(val_acc))
            current_val = float(record["validation_loss"])
            if current_val + 1e-8 < best_val:
                best_val = current_val
                best_epoch = epoch
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    history.append(record)
                    break
        else:
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(record)

    model.load_state_dict(best_state)
    model.eval()
    summary = {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val if val_loader is not None else None,
        "epochs_run": history[-1]["epoch"] if history else 0,
        "device": str(torch_device),
        "train_pair_count": int(len(train_pairs[0])),
        "validation_pair_count": int(len(val_pairs[0])) if val_pairs is not None else 0,
        "hidden_sizes": list(hidden_sizes),
        "dropout": dropout,
    }
    return model.cpu(), summary
