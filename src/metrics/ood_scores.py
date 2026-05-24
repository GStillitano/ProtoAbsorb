"""OOD scoring functions operating on logits."""
import torch


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Energy score: -T * log(sum(exp(logits / T))). Higher = more ID-like."""
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def max_logit_score(logits: torch.Tensor) -> torch.Tensor:
    """Max logit score. Higher = more ID-like."""
    return logits.max(dim=-1).values


def max_softmax_score(logits: torch.Tensor) -> torch.Tensor:
    """Maximum softmax probability. Higher = more ID-like."""
    return torch.softmax(logits, dim=-1).max(dim=-1).values
