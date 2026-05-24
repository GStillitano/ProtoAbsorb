"""Rome32 OOD dataset loader — to be implemented when data is available."""
import torch


def load_rome32_c(
    folder: str,
    corruption: str,
    severity: int = 5,
    indices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load images from folder, resize to 32×32, apply corruption.

    Returns (x: Tensor[N, 3, 32, 32], y: Tensor[N]) — y is zeros, labels unused.
    """
    raise NotImplementedError("Rome32 dataset not yet available.")
