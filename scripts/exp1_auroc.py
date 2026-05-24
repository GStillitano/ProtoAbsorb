"""Experiment 1: AUROC and csID accuracy trajectory over the adaptation stream.

Run for both TENT and BN Adapt to compare OOD detection degradation.

Usage:
    uv run python scripts/exp1_auroc.py --streams tent/gaussian_noise_svhn_c_0.50_seed0 bn_adapt/gaussian_noise_svhn_c_0.50_seed0
"""
import argparse
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from src.model import load_model
from src.bn_affine import evaluate
from src.data.cifar10c import load_cifar10c_data
from src.data.svhnc import load_svhn_c
from src.data.pools import DataPools
from src.metrics.ood_scores import energy_score
from src.metrics.ood_metrics import auroc


def load_diagnostic(meta: dict, data_dir: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rebuild the fixed diagnostic set from stream meta. Returns (x_id, y_id, x_ood)."""
    from src.data.rome32 import load_rome32_c
    import yaml

    diag_cfg = yaml.safe_load(Path("configs/diagnostic.yaml").read_text())

    x_csid, y_csid = load_cifar10c_data(meta["corruption"], meta["severity"], data_dir=data_dir)
    if meta["csood_source"] == "svhn_c":
        x_csood, _ = load_svhn_c(meta["corruption"], meta["severity"], data_dir=data_dir)
    elif meta["csood_source"] == "rome32":
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
    return (
        x_csid[pools.csid_diag],
        y_csid[pools.csid_diag],
        x_csood[pools.csood_diag],
    )


def run_stream(stream_id: str, model, data_dir: str, device: str) -> dict:
    ckpt_dir = Path("checkpoints") / stream_id
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    T = meta["T"]

    x_id, y_id, x_ood = load_diagnostic(meta, data_dir)

    auroc_list, acc_list = [], []
    for t in range(T + 1):
        feat_id, logits_id = evaluate(model, ckpt_dir / f"theta_{t:03d}.pt", x_id, device)
        feat_ood, logits_ood = evaluate(model, ckpt_dir / f"theta_{t:03d}.pt", x_ood, device)

        scores_id  = -energy_score(logits_id)
        scores_ood = -energy_score(logits_ood)

        auroc_list.append(auroc(scores_id, scores_ood))
        acc_list.append((logits_id.argmax(dim=-1) == y_id).float().mean().item())

    return {"t": list(range(T + 1)), "auroc": auroc_list, "acc_csid": acc_list}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", nargs="+", required=True,
                        help="One or more stream IDs, e.g. tent/gaussian_noise_svhn_c_0.50_seed0")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--device",   default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(data_dir=args.data_dir).to(device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    all_results = {}

    for sid in args.streams:
        results = run_stream(sid, model, args.data_dir, device)
        label = sid.split("/")[0]  # method name as legend label
        ax1.plot(results["t"], results["auroc"],    label=label)
        ax2.plot(results["t"], results["acc_csid"], label=label)
        all_results[sid] = results

        out_dir = Path("results") / sid / "exp1_auroc"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    ax1.set_xlabel("Step t"); ax1.set_ylabel("AUROC");      ax1.set_title("AUROC over stream");      ax1.legend()
    ax2.set_xlabel("Step t"); ax2.set_ylabel("Acc (csID)"); ax2.set_title("csID accuracy over stream"); ax2.legend()
    fig.tight_layout()

    out_base = Path("results") / args.streams[0].split("/")[1] / "exp1_auroc"
    out_base.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base / "exp1_auroc.png", dpi=150)
    plt.close(fig)
    print(f"Saved to {out_base}")


if __name__ == "__main__":
    main()
