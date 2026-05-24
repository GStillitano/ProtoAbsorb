from dataclasses import dataclass
import numpy as np
import torch
from typing import Iterator


@dataclass
class BatchMeta:
    t: int
    csid_indices: list[int]
    csood_indices: list[int]


@dataclass
class AdaptationStream:
    batches: list[BatchMeta]
    x_csid: torch.Tensor
    y_csid: torch.Tensor
    x_csood: torch.Tensor

    def __iter__(self) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield (t, x_batch) for each batch. Labels not needed during TTA."""
        for meta in self.batches:
            x_id  = self.x_csid[meta.csid_indices]
            x_ood = self.x_csood[meta.csood_indices]
            yield meta.t, torch.cat([x_id, x_ood], dim=0)

    def to_meta_list(self) -> list[dict]:
        return [
            {"t": b.t, "csid_indices": b.csid_indices, "csood_indices": b.csood_indices}
            for b in self.batches
        ]


def build_stream(
    x_csid: torch.Tensor,
    y_csid: torch.Tensor,
    x_csood: torch.Tensor,
    adapt_csid_indices: list[int],
    adapt_csood_indices: list[int],
    N: int,
    T: int,
    alpha: float,
    seed: int = 0,
) -> AdaptationStream:
    """Build a frozen adaptation stream of T batches, each size N with OOD proportion alpha."""
    n_ood = round(alpha * N)
    n_id = N - n_ood

    rng = np.random.default_rng(seed)
    batches = []

    for t in range(1, T + 1):
        csid_batch = rng.choice(adapt_csid_indices, size=n_id, replace=False).tolist()
        csood_batch = rng.choice(adapt_csood_indices, size=n_ood, replace=False).tolist() if n_ood > 0 else []
        batches.append(BatchMeta(t=t, csid_indices=csid_batch, csood_indices=csood_batch))

    return AdaptationStream(
        batches=batches,
        x_csid=x_csid,
        y_csid=y_csid,
        x_csood=x_csood,
    )
