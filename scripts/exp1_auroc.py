"""Experiment 1: AUROC and csID accuracy trajectory over the adaptation stream.

Outputs results.json per stream. Use scripts/plot.py --exp 1 to visualise.

Usage:
    uv run python scripts/exp1_auroc.py --streams tent/gaussian_noise_svhn_c_0.50_seed0 bn_adapt/gaussian_noise_svhn_c_0.50_seed0
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

from src.model import load_model
from src.bn_affine import evaluate
from src.data.cifar10c import load_cifar10c_data
from src.data.svhnc import load_svhn_c
from src.data.pools import DataPools
from src.metrics.ood_scores import energy_score
from src.metrics.ood_metrics import auroc


def load_diagnostic(meta: dict, data_dir: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def run_stream(stream_id: str, model, data_dir: str, device: str) -> dict:
    ckpt_dir = Path("checkpoints") / stream_id
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    T = meta["T"]
    x_id, y_id, x_ood = load_diagnostic(meta, data_dir)

    auroc_list, acc_list = [], []
    for t in range(T + 1):
        ckpt = ckpt_dir / f"theta_{t:03d}.pt"
        feat_id,  logits_id  = evaluate(model, ckpt, x_id,  device)
        feat_ood, logits_ood = evaluate(model, ckpt, x_ood, device)
        scores_id  = -energy_score(logits_id)
        scores_ood = -energy_score(logits_ood)
        auroc_list.append(auroc(scores_id, scores_ood))
        acc_list.append((logits_id.argmax(dim=-1) == y_id).float().mean().item())

    return {"t": list(range(T + 1)), "auroc": auroc_list, "acc_csid": acc_list}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", nargs="+", required=True)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--device",   default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(data_dir=args.data_dir).to(device)

    for sid in args.streams:
        results = run_stream(sid, model, args.data_dir, device)
        out_dir = Path("results") / sid / "exp1_auroc"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
