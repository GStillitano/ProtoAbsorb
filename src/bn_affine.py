"""BN affine state (γ, β): extract, inject, save, load, evaluate_diagnostic_stream."""
import torch
import torch.nn as nn
from pathlib import Path

from src.model import get_embeddings


def extract(model: nn.Module, t: int = 0) -> dict:
    gamma, beta = {}, {}
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                gamma[name] = m.weight.detach().cpu().clone()
            if m.bias is not None:
                beta[name] = m.bias.detach().cpu().clone()
    return {"t": t, "gamma": gamma, "beta": beta}


def inject(model: nn.Module, state: dict) -> None:
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


def evaluate_diagnostic_stream(
    model: nn.Module,
    ckpt_path: Path,
    x_id: torch.Tensor,
    x_ood: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate checkpoint on the diagnostic set using N-sized mixed batches.

    Matches the adaptation batch size so BN statistics are in the same regime.
    Batches are sequential slices of x_id / x_ood — order preserved so that
    returned logits_id[i] corresponds to x_id[i] (and y_id[i] in callers).

    open_set  (x_ood non-empty): K = len(x_id)//n_id batches of n_id ID + n_ood OOD.
    closed_set (x_ood empty):    K = len(x_id)//batch_size batches of batch_size ID.

    Returns: feat_id, logits_id, feat_ood, logits_ood  (assembled from K batches).
             feat_ood / logits_ood are empty tensors for closed_set.
    """
    inject(model, load(ckpt_path))

    open_set = len(x_ood) > 0

    if open_set:
        n_id  = batch_size // 2
        n_ood = batch_size // 2
        K = min(len(x_id) // n_id, len(x_ood) // n_ood)

        feat_id_list, logits_id_list = [], []
        feat_ood_list, logits_ood_list = [], []

        for k in range(K):
            x_batch = torch.cat([
                x_id [k * n_id  : (k + 1) * n_id],
                x_ood[k * n_ood : (k + 1) * n_ood],
            ], dim=0)
            feat, logits = get_embeddings(model, x_batch, device)
            feat_id_list.append(feat[:n_id])
            logits_id_list.append(logits[:n_id])
            feat_ood_list.append(feat[n_id:])
            logits_ood_list.append(logits[n_id:])

        return (torch.cat(feat_id_list),   torch.cat(logits_id_list),
                torch.cat(feat_ood_list),  torch.cat(logits_ood_list))

    else:
        n_id = batch_size
        K    = len(x_id) // n_id

        feat_id_list, logits_id_list = [], []

        for k in range(K):
            feat, logits = get_embeddings(model, x_id[k * n_id : (k + 1) * n_id], device)
            feat_id_list.append(feat)
            logits_id_list.append(logits)

        empty = torch.empty(0)
        return torch.cat(feat_id_list), torch.cat(logits_id_list), empty, empty
