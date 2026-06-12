"""Load the fixed held-out diagnostic set D for a stream.

Single source of truth shared by every Phase 2 script (exp1, exp2, maxcos_dist).
The pool split is keyed by `meta["seed"]`, so the diagnostic indices match the
ones held out during Phase 1 for the same stream.
"""
from pathlib import Path

import torch
import yaml

from src.data.cifar10c import load_cifar10c_data
from src.data.svhnc import load_svhn_c
from src.data.pools import DataPools


def load_diagnostic(meta: dict, data_dir: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x_csid, y_csid, x_csood) for the diagnostic split of `meta`'s stream."""
    diag_cfg = yaml.safe_load(Path("configs/diagnostic.yaml").read_text())
    x_csid, y_csid = load_cifar10c_data(meta["corruption"], meta["severity"], data_dir=data_dir)

    if meta["csood_source"] == "svhn_c":
        x_csood, _ = load_svhn_c(meta["corruption"], meta["severity"], data_dir=data_dir)
    elif meta["csood_source"] == "rome32":
        from src.data.rome32 import load_rome32_c
        x_csood, _ = load_rome32_c(
            folder=str(Path(data_dir) / "rome32/raw"),
            corruption=meta["corruption"], severity=meta["severity"],
        )
    else:
        raise ValueError(f"Unknown csood_source: {meta['csood_source']}")

    pools = DataPools(
        n_csid=len(x_csid), n_csood=len(x_csood),
        n_diag_csid=diag_cfg["n_csid"], n_diag_csood=diag_cfg["n_csood"],
        seed=meta["seed"],
    )
    return x_csid[pools.csid_diag], y_csid[pools.csid_diag], x_csood[pools.csood_diag]
