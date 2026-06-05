"""Cassano loss: soft-labeled entropy / L1-norm penalty.

Per sample, a soft OOD posterior p_ood (from the GMM on maxcos scores) splits the
objective between two regimes:

    ID-soft  (weight p_id = 1 - p_ood):  minimize softmax entropy   (confident, TENT-style)
    OOD-soft (weight p_ood):             minimize feature L1 norm    (suppress norm inflation)

    loss_i = p_id_i * H(logits_i)  +  p_ood_i * l1_weight * ||feat_i||_1
    loss   = mean_i loss_i

Scale note (WRN features, gaussian_noise/5, t=0):
    H(logits)   in [0, ln 10 ≈ 2.30]      typ. ~1
    ||feat||_1  ≈ 28–30                    (≈ 13–30x larger than entropy)
So l1_weight = 1 lets the OOD term dominate. Use l1_weight ≈ 1/30 ≈ 0.03 to put the
two terms on the same scale (or tune up if stronger OOD suppression is wanted).
"""
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from src.model import classifier_weights, get_embeddings
from src.metrics.geometry import max_cosine_to_weights


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample softmax entropy from logits. Shape: [N]."""
    return -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)


def feature_l1(feat: torch.Tensor) -> torch.Tensor:
    """Per-sample feature L1 norm. Shape: [N]."""
    return feat.abs().sum(dim=-1)


def cassano_loss(
    logits: torch.Tensor,
    feat: torch.Tensor,
    p_ood: torch.Tensor,
    l1_weight: float = 0.03,
) -> torch.Tensor:
    """Soft-labeled loss on the adapted model's outputs.

    logits:    [N, K]  adapted-model logits (grad)
    feat:      [N, d]  adapted-model penultimate features (grad)
    p_ood:     [N]     GMM Bayes posterior P(OOD | x), in [0, 1] (detached)
    l1_weight: scalar scaling the OOD L1 term relative to the ID entropy term
    """
    p_ood = p_ood.detach().clamp(0.0, 1.0)
    p_id = 1.0 - p_ood

    id_term  = p_id  * softmax_entropy(logits)
    ood_term = p_ood * l1_weight * feature_l1(feat)
    return (id_term + ood_term).mean()


# ── GMM soft labeling ─────────────────────────────────────────────────────────
class GmmScorer:
    """Accumulate maxcos scores and emit a per-batch OOD posterior via a 2-Gaussian GMM.

    Scores come from the frozen scorer, so their distribution is ~stationary across
    steps; `pool` accumulation is therefore unbiased and converges fast.
    """

    def __init__(
        self,
        components: int = 2,
        accumulate: str = "pool",      # pool | window | batch
        window: int = 0,               # steps kept when accumulate = window
        warm_start: bool = False,
        reg_covar: float = 1e-6,
        id_component: str = "high_mean",
    ):
        self.components = components
        self.accumulate = accumulate
        self.window = window
        self.warm_start = warm_start
        self.reg_covar = reg_covar
        self.id_component = id_component
        self.history: list[np.ndarray] = []   # one [n,1] array per step
        self.gmm: GaussianMixture | None = None

    def _fit_data(self, batch: np.ndarray) -> np.ndarray:
        if self.accumulate == "batch":
            self.history = [batch]
        else:
            self.history.append(batch)
            if self.accumulate == "window" and self.window > 0:
                self.history = self.history[-self.window:]
        return np.concatenate(self.history, axis=0)

    def update_and_posterior(self, scores: torch.Tensor) -> torch.Tensor:
        """Add `scores` [N], refit GMM on accumulated data, return P(OOD|x) [N] for this batch."""
        batch = scores.detach().cpu().numpy().reshape(-1, 1).astype(np.float64)
        data = self._fit_data(batch)

        gmm = GaussianMixture(
            n_components=self.components,
            reg_covar=self.reg_covar,
            warm_start=self.warm_start,
            means_init=(self.gmm.means_ if self.warm_start and self.gmm is not None else None),
        )
        gmm.fit(data)
        self.gmm = gmm

        # ID = higher-mean component; OOD = the other.
        means = gmm.means_.ravel()
        id_idx = int(means.argmax()) if self.id_component == "high_mean" else int(means.argmin())
        ood_idx = 1 - id_idx

        resp = gmm.predict_proba(batch)          # [N, 2] posterior on this batch
        p_ood = resp[:, ood_idx]
        return torch.from_numpy(p_ood).float()


# ── LR warmup ─────────────────────────────────────────────────────────────────
def warmup_factor(t: int, K: int, shape: str = "linear", exp_tau: float = 0.3) -> float:
    """Ramp in [0,1] reaching 1 at step t >= K. t is 1-indexed (first adapt step = 1)."""
    if K <= 0:
        return 1.0
    x = min(t / K, 1.0)
    if shape == "linear":
        return x
    if shape == "exp":
        return min(1.0, (1.0 - math.exp(-x / exp_tau)) / (1.0 - math.exp(-1.0 / exp_tau)))
    if shape == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * x))
    raise ValueError(f"Unknown warmup shape: {shape}")


# ── Model setup / forward ─────────────────────────────────────────────────────
def configure_model(model: nn.Module) -> nn.Module:
    """Adapted model: train mode, only BN affine (γ,β) trainable, BN uses batch stats."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def forward_feat_logits(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Grad-enabled forward → (feat [N,d], logits [N,K]). feat = input to last Linear."""
    activations = {}
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linear_layers, "No Linear layer found in model"
    handle = linear_layers[-1].register_forward_hook(
        lambda _, inp, __: activations.update({"f": inp[0]})
    )
    logits = model(x)
    handle.remove()
    return activations["f"], logits


@torch.enable_grad()
def forward_and_adapt(
    x: torch.Tensor,
    model: nn.Module,
    scorer: nn.Module,
    W_cpu: torch.Tensor,
    gmm: GmmScorer,
    optimizer: torch.optim.Optimizer,
    l1_weight: float = 0.03,
    device: str = "cpu",
) -> torch.Tensor:
    """One Cassano step: score (frozen) → GMM posterior → soft-labeled update (adapted).

    Returns p_ood [N] (cpu) for logging. LR warmup is applied by the caller.
    """
    # Score with the FROZEN scorer (BN follows batch stats, no grad).
    with torch.no_grad():
        feat_s, _ = get_embeddings(scorer, x, device)        # cpu tensors
        maxcos = max_cosine_to_weights(feat_s, W_cpu)        # [N] cpu
    p_ood = gmm.update_and_posterior(maxcos).to(device)      # [N]

    # Update the adapted model with the soft-labeled loss.
    feat, logits = forward_feat_logits(model, x.to(device))
    loss = cassano_loss(logits, feat, p_ood, l1_weight)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return p_ood.detach().cpu()
