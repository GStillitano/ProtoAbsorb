"""maxcos score distribution at θ_0: csID vs csOOD.

Score = max cosine similarity of feature to any (frozen) class weight vector.
θ_0 = original frozen weights + BN batch-stat adaptation (no gradient step).
Evaluation follows the 5.2 protocol via evaluate_diagnostic_stream.

Usage:
    uv run python scripts/maxcos_dist.py --stream tent/gaussian_noise_svhn_c_open_seed0
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

from src.model import load_model, classifier_weights
from src.bn_affine import evaluate_diagnostic_stream
from src.data.cifar10c import load_cifar10c_data
from src.data.svhnc import load_svhn_c
from src.data.pools import DataPools
from src.metrics.geometry import max_cosine_to_weights
from src.device import get_device


def load_diagnostic(meta: dict, data_dir: str):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream",   required=True)
    parser.add_argument("--t",        type=int, default=0, help="stream step (0 = frozen + BN adapt, no grad)")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--device",   default=None)
    args = parser.parse_args()

    device   = args.device or get_device()
    ckpt_dir = Path("checkpoints") / args.stream
    meta     = json.loads((ckpt_dir / "meta.json").read_text())
    if not meta["open_set"]:
        raise ValueError("csID-vs-csOOD distribution needs an open-set stream.")

    # Fresh model = original frozen weights; classifier weights frozen during TTA.
    model = load_model(data_dir=args.data_dir).to(device)
    W_cpu = classifier_weights(model).cpu()

    x_id, _, x_ood = load_diagnostic(meta, args.data_dir)

    # θ_t via 5.2 protocol: inject affine state, forward in mixed N=200 batches (BN adapt).
    ckpt = ckpt_dir / f"theta_{args.t:03d}.pt"
    feat_id, _, feat_ood, _ = evaluate_diagnostic_stream(model, ckpt, x_id, x_ood, device)

    maxcos_id  = max_cosine_to_weights(feat_id, W_cpu)
    maxcos_ood = max_cosine_to_weights(feat_ood, W_cpu)

    results = {
        "t": args.t,
        "maxcos_id":  maxcos_id.tolist(),
        "maxcos_ood": maxcos_ood.tolist(),
    }
    out_dir = Path("results") / args.stream / "maxcos_dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results))
    print(f"t={args.t} | maxcos csID mean={maxcos_id.mean():.3f}  csOOD mean={maxcos_ood.mean():.3f}")
    print(f"Saved: {out_dir / 'results.json'}")

    from src.viz.maxcos_dist import plot
    plot(results, Path("figures") / args.stream / f"maxcos_dist_t{args.t}.png")


if __name__ == "__main__":
    main()
