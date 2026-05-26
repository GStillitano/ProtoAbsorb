"""Source class centroids μ_c^(0): computed from clean CIFAR-10 with original model weights."""
import torch
import torch.nn as nn
from pathlib import Path

from src.model import get_embeddings


def compute(
    model: nn.Module,
    x_clean: torch.Tensor,
    y_clean: torch.Tensor,
    device: str = "cpu",
    cache_path: Path | None = None,
) -> dict[int, torch.Tensor]:
    """Compute {class_id: centroid [d]} from original model weights on clean CIFAR-10.

    Caller must pass model in original (pre-adaptation) state — no checkpoint injection.
    Loads from cache if available.
    """
    if cache_path is not None and Path(cache_path).exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    features, _ = get_embeddings(model, x_clean, device)

    centroids = {
        int(c): features[y_clean == c].mean(dim=0)
        for c in y_clean.unique().tolist()
    }

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(centroids, cache_path)

    return centroids
