"""BN affine state (γ, β): extract, inject, save, load, evaluate."""
import torch
import torch.nn as nn
from pathlib import Path

from src.model import get_embeddings


def extract(model: nn.Module, t: int = 0) -> dict:
    """Pull current (γ, β) from all BN layers."""
    gamma, beta = {}, {}
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                gamma[name] = m.weight.detach().cpu().clone()
            if m.bias is not None:
                beta[name] = m.bias.detach().cpu().clone()
    return {"t": t, "gamma": gamma, "beta": beta}


def inject(model: nn.Module, state: dict) -> None:
    """Push (γ, β) from state into model in-place."""
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if name in state["gamma"] and m.weight is not None:
                m.weight.data.copy_(state["gamma"][name].to(m.weight.device))
            if name in state["beta"] and m.bias is not None:
                m.bias.data.copy_(state["beta"][name].to(m.bias.device))


def save(state: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load(path: Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def evaluate(
    model: nn.Module,
    ckpt_path: Path,
    x: torch.Tensor,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (γ, β) from checkpoint, forward x in BN train mode → (features [N,d], logits [N,K]).

    BN statistics are recomputed from x — results reflect only the affine state at ckpt_path.
    This is the single Phase 2 evaluation primitive.
    """
    inject(model, load(ckpt_path))
    return get_embeddings(model, x, device)
