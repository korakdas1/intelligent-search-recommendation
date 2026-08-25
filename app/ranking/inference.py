"""Score rank-features-v1 rows with a loaded RankNet. Inference only."""

from __future__ import annotations

import numpy as np
import torch

from app.ranking.preprocessing import FeatureScaler
from app.ranking.ranker import RankNetMLP
from app.ranking.validate import FeatureValidationError


def score_feature_matrix(
    model: RankNetMLP,
    scaler: FeatureScaler,
    values: np.ndarray,
    *,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    if values.ndim != 2:
        raise FeatureValidationError("values must be 2-D")
    if not np.isfinite(values).all():
        raise FeatureValidationError("feature matrix contains NaN or Inf")
    torch_device = torch.device(device)
    model.eval()
    model.to(torch_device)
    scaled = scaler.transform(values)
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(scaled)).to(torch_device)
        scores = model(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(scores).all():
        raise FeatureValidationError("ranker scores contain NaN or Inf")
    return scores
