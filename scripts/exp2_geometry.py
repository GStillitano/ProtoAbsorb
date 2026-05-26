"""Experiment 2: Norm vs alignment — norm inflation as the dominant TENT channel.

Outputs results.json. Use scripts/plot.py --exp 2 to visualise.

Usage:
    uv run python scripts/exp2_geometry.py --stream tent/gaussian_noise_svhn_c_open_seed0
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

from src.model import load_model, classifier_weights
from src.bn_affine import evaluate_diagnostic_stream
from src.data.cifar10c import load_cifar10c_data
from src.data.cifar10 import load_cifar10_data
from src.data.svhnc import load_svhn_c
from src.data.pools import DataPools
from src.centroids import compute as compute_centroids
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
    open_set = meta["open_set"]
    out_dir  = Path("results") / args.stream / "exp2_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model loaded fresh = original weights (θ_0); no checkpoint injection yet.
    model = load_model(data_dir=args.data_dir).to(device)
    W     = classifier_weights(model).to(device)

    # Centroids from clean CIFAR-10 with original model weights (semantic anchors).
    x_clean, y_clean = load_cifar10_data(n_examples=2000, data_dir=args.data_dir)
    centroids = compute_centroids(
        model, x_clean, y_clean, device=device,
        cache_path=ckpt_dir / "centroids_clean.pt",
    )

    x_id, y_id, x_ood = load_diagnostic(meta, args.data_dir)
    x_ood_eval = x_ood if open_set else torch.empty(0)
    W_cpu = W.cpu()

    records = {
        "open_set": open_set,
        "t": [], "norm_id": [], "norm_ood": [], "delta_norm": [],
        "cos_id": [], "cos_ood": [], "maxcos_id": [], "maxcos_ood": [],
        "dist_id": [], "dist_ood": [], "conf_ood": [], "change_ood": [],
    }

    pred0_id = pred0_ood = prev_pred_ood = None

    for t in range(T + 1):
        ckpt = ckpt_dir / f"theta_{t:03d}.pt"
        feat_id, logits_id, feat_ood, logits_ood = evaluate_diagnostic_stream(
            model, ckpt, x_id, x_ood_eval, device,
        )

        cur_pred_id  = logits_id.argmax(dim=-1)
        cur_pred_ood = logits_ood.argmax(dim=-1) if open_set else None

        if t == 0:
            pred0_id     = cur_pred_id
            pred0_ood    = cur_pred_ood
            prev_pred_ood = cur_pred_ood
            f_csid  = (cosine_to_weights(feat_id, W_cpu, pred0_id) > 0).float().mean().item()
            f_csood = (cosine_to_weights(feat_ood, W_cpu, pred0_ood) > 0).float().mean().item() if open_set else None
            records["F_csID_t0"]  = f_csid
            records["F_csOOD_t0"] = f_csood
            print(f"t=0 | F_csID={f_csid:.3f}" + (f"  F_csOOD={f_csood:.3f}" if open_set else ""))

        change_ood = (cur_pred_ood != prev_pred_ood).float().mean().item() if open_set and t > 0 else 0.0
        if open_set:
            prev_pred_ood = cur_pred_ood

        records["t"].append(t)
        records["norm_id"].append(feature_norms(feat_id).mean().item())
        records["norm_ood"].append(feature_norms(feat_ood).mean().item() if open_set else None)
        records["delta_norm"].append(
            records["norm_id"][-1] - records["norm_ood"][-1] if open_set else None
        )
        records["cos_id"].append(cosine_to_weights(feat_id, W_cpu, pred0_id).mean().item())
        records["cos_ood"].append(
            cosine_to_weights(feat_ood, W_cpu, pred0_ood).mean().item() if open_set else None
        )
        records["maxcos_id"].append(max_cosine_to_weights(feat_id,  W_cpu).mean().item())
        records["maxcos_ood"].append(
            max_cosine_to_weights(feat_ood, W_cpu).mean().item() if open_set else None
        )
        records["dist_id"].append(centroid_distances(feat_id, centroids).mean().item())
        records["dist_ood"].append(
            centroid_distances(feat_ood, centroids).mean().item() if open_set else None
        )
        records["conf_ood"].append(
            torch.softmax(logits_ood, dim=-1).max(dim=-1).values.mean().item() if open_set else None
        )
        records["change_ood"].append(change_ood)

    (out_dir / "results.json").write_text(json.dumps(records, indent=2))
    print(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
