"""Experiment 2: Norm vs alignment — norm inflation as the dominant TENT channel.

Outputs results.json. Use scripts/plot.py --exp 2 to visualise.

Usage:
    uv run python scripts/exp2_geometry.py --stream tent/gaussian_noise_svhn_c_0.50_seed0
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

from src.model import load_model, classifier_weights
from src.bn_affine import evaluate
from src.data.cifar10c import load_cifar10c_data
from src.data.svhnc import load_svhn_c
from src.data.pools import DataPools
from src.prototypes import compute as compute_prototypes
from src.metrics.geometry import (
    feature_norms, cosine_to_weights, max_cosine_to_weights, centroid_distances,
)


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
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--device",   default=None)
    args = parser.parse_args()

    device   = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path("checkpoints") / args.stream
    meta     = json.loads((ckpt_dir / "meta.json").read_text())
    T        = meta["T"]
    out_dir  = Path("results") / args.stream / "exp2_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    model  = load_model(data_dir=args.data_dir).to(device)
    W      = classifier_weights(model).to(device)  # [K, d]

    x_id, y_id, x_ood = load_diagnostic(meta, args.data_dir)

    ckpt0 = ckpt_dir / "theta_000.pt"
    centroids = compute_prototypes(
        model, ckpt0, x_id, y_id, device=device,
        cache_path=ckpt_dir / "prototypes.pt",
    )

    # Source-predicted class at t=0 (fixed reference for cosine tracking)
    _, logits0_id  = evaluate(model, ckpt0, x_id,  device)
    _, logits0_ood = evaluate(model, ckpt0, x_ood, device)
    pred0_id  = logits0_id.argmax(dim=-1)
    pred0_ood = logits0_ood.argmax(dim=-1)

    # t=0 diagnostic: fraction with cos > 0
    feat0_id,  _ = evaluate(model, ckpt0, x_id,  device)
    feat0_ood, _ = evaluate(model, ckpt0, x_ood, device)
    W_cpu = W.cpu()
    f_csid  = (cosine_to_weights(feat0_id,  W_cpu, pred0_id)  > 0).float().mean().item()
    f_csood = (cosine_to_weights(feat0_ood, W_cpu, pred0_ood) > 0).float().mean().item()
    print(f"t=0 | F_csID={f_csid:.3f}  F_csOOD={f_csood:.3f}")

    records = {
        "F_csID_t0": f_csid, "F_csOOD_t0": f_csood,
        "t": [], "norm_id": [], "norm_ood": [], "delta_norm": [],
        "cos_id": [], "cos_ood": [], "maxcos_id": [], "maxcos_ood": [],
        "dist_id": [], "dist_ood": [], "conf_ood": [], "change_ood": [],
    }

    prev_pred_ood = pred0_ood.clone()

    for t in range(T + 1):
        ckpt = ckpt_dir / f"theta_{t:03d}.pt"
        feat_id,  logits_id  = evaluate(model, ckpt, x_id,  device)
        feat_ood, logits_ood = evaluate(model, ckpt, x_ood, device)

        cur_pred_ood = logits_ood.argmax(dim=-1)

        records["t"].append(t)
        records["norm_id"].append(feature_norms(feat_id).mean().item())
        records["norm_ood"].append(feature_norms(feat_ood).mean().item())
        records["delta_norm"].append(records["norm_id"][-1] - records["norm_ood"][-1])
        records["cos_id"].append(cosine_to_weights(feat_id,  W_cpu, pred0_id).mean().item())
        records["cos_ood"].append(cosine_to_weights(feat_ood, W_cpu, pred0_ood).mean().item())
        records["maxcos_id"].append(max_cosine_to_weights(feat_id,  W_cpu).mean().item())
        records["maxcos_ood"].append(max_cosine_to_weights(feat_ood, W_cpu).mean().item())
        records["dist_id"].append(centroid_distances(feat_id,  centroids).mean().item())
        records["dist_ood"].append(centroid_distances(feat_ood, centroids).mean().item())
        records["conf_ood"].append(torch.softmax(logits_ood, dim=-1).max(dim=-1).values.mean().item())
        records["change_ood"].append((cur_pred_ood != prev_pred_ood).float().mean().item())
        prev_pred_ood = cur_pred_ood.clone()

    (out_dir / "results.json").write_text(json.dumps(records, indent=2))
    print(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
