"""Source class centroids μ_c^(0): compute from θ_0 on D_csID and cache."""
import torch
import torch.nn as nn
from pathlib import Path

from src.bn_affine import evaluate


def compute(
    model: nn.Module,
    ckpt0_path: Path,
    x_csid: torch.Tensor,
    y_csid: torch.Tensor,
    device: str = "cpu",
    cache_path: Path | None = None,
) -> dict[int, torch.Tensor]:
    """Compute {class_id: centroid [d]} from θ_0 on D_csID. Loads from cache if available."""
    if cache_path is not None and Path(cache_path).exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    features, _ = evaluate(model, ckpt0_path, x_csid, device)

    centroids = {
        int(c): features[y_csid == c].mean(dim=0)
        for c in y_csid.unique().tolist()
    }

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(centroids, cache_path)

    return centroids
