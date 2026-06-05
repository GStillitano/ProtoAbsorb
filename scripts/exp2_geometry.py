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
from src.device import get_device
from src.metrics.geometry import (
    feature_norms, feature_norms_l1, cosine_to_weights, max_cosine_to_weights, centroid_distances,
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

    device   = args.device or get_device()
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

    def _qs(tensor, open_set_flag=True):
        """Return (mean, q25, q75) as floats, or (None, None, None) if flag is False."""
        if not open_set_flag or tensor is None:
            return None, None, None
        return tensor.mean().item(), tensor.quantile(0.25).item(), tensor.quantile(0.75).item()

    records = {
        "open_set": open_set,
        "t": [],
        # L2 norm
        "norm_id": [], "norm_id_q25": [], "norm_id_q75": [],
        "norm_ood": [], "norm_ood_q25": [], "norm_ood_q75": [],
        "delta_norm": [],
        # L1 norm
        "norm_l1_id": [], "norm_l1_id_q25": [], "norm_l1_id_q75": [],
        "norm_l1_ood": [], "norm_l1_ood_q25": [], "norm_l1_ood_q75": [],
        "delta_norm_l1": [],
        # Cosine alignment
        "cos_id": [], "cos_id_q25": [], "cos_id_q75": [],
        "cos_ood": [], "cos_ood_q25": [], "cos_ood_q75": [],
        "maxcos_id": [], "maxcos_id_q25": [], "maxcos_id_q75": [],
        "maxcos_ood": [], "maxcos_ood_q25": [], "maxcos_ood_q75": [],
        # Centroid distance
        "dist_id": [], "dist_id_q25": [], "dist_id_q75": [],
        "dist_ood": [], "dist_ood_q25": [], "dist_ood_q75": [],
        # Confidence
        "conf_ood": [], "conf_ood_q25": [], "conf_ood_q75": [],
        "change_ood": [],
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

        # L2 norms
        l2_id  = feature_norms(feat_id)
        l2_ood = feature_norms(feat_ood) if open_set else None
        records["norm_id"].append(l2_id.mean().item())
        records["norm_id_q25"].append(l2_id.quantile(0.25).item())
        records["norm_id_q75"].append(l2_id.quantile(0.75).item())
        records["norm_ood"].append(l2_ood.mean().item() if open_set else None)
        records["norm_ood_q25"].append(l2_ood.quantile(0.25).item() if open_set else None)
        records["norm_ood_q75"].append(l2_ood.quantile(0.75).item() if open_set else None)
        records["delta_norm"].append(
            records["norm_id"][-1] - records["norm_ood"][-1] if open_set else None
        )

        # L1 norms
        l1_id  = feature_norms_l1(feat_id)
        l1_ood = feature_norms_l1(feat_ood) if open_set else None
        records["norm_l1_id"].append(l1_id.mean().item())
        records["norm_l1_id_q25"].append(l1_id.quantile(0.25).item())
        records["norm_l1_id_q75"].append(l1_id.quantile(0.75).item())
        records["norm_l1_ood"].append(l1_ood.mean().item() if open_set else None)
        records["norm_l1_ood_q25"].append(l1_ood.quantile(0.25).item() if open_set else None)
        records["norm_l1_ood_q75"].append(l1_ood.quantile(0.75).item() if open_set else None)
        records["delta_norm_l1"].append(
            records["norm_l1_id"][-1] - records["norm_l1_ood"][-1] if open_set else None
        )
        cos_id_t   = cosine_to_weights(feat_id, W_cpu, pred0_id)
        cos_ood_t  = cosine_to_weights(feat_ood, W_cpu, pred0_ood) if open_set else None
        m, q25, q75 = cos_id_t.mean().item(), cos_id_t.quantile(0.25).item(), cos_id_t.quantile(0.75).item()
        records["cos_id"].append(m); records["cos_id_q25"].append(q25); records["cos_id_q75"].append(q75)
        m, q25, q75 = _qs(cos_ood_t)
        records["cos_ood"].append(m); records["cos_ood_q25"].append(q25); records["cos_ood_q75"].append(q75)

        maxcos_id_t  = max_cosine_to_weights(feat_id, W_cpu)
        maxcos_ood_t = max_cosine_to_weights(feat_ood, W_cpu) if open_set else None
        m, q25, q75 = maxcos_id_t.mean().item(), maxcos_id_t.quantile(0.25).item(), maxcos_id_t.quantile(0.75).item()
        records["maxcos_id"].append(m); records["maxcos_id_q25"].append(q25); records["maxcos_id_q75"].append(q75)
        m, q25, q75 = _qs(maxcos_ood_t)
        records["maxcos_ood"].append(m); records["maxcos_ood_q25"].append(q25); records["maxcos_ood_q75"].append(q75)

        dist_id_t  = centroid_distances(feat_id, centroids)
        dist_ood_t = centroid_distances(feat_ood, centroids) if open_set else None
        m, q25, q75 = dist_id_t.mean().item(), dist_id_t.quantile(0.25).item(), dist_id_t.quantile(0.75).item()
        records["dist_id"].append(m); records["dist_id_q25"].append(q25); records["dist_id_q75"].append(q75)
        m, q25, q75 = _qs(dist_ood_t)
        records["dist_ood"].append(m); records["dist_ood_q25"].append(q25); records["dist_ood_q75"].append(q75)

        if open_set:
            conf_t = torch.softmax(logits_ood, dim=-1).max(dim=-1).values
            m, q25, q75 = conf_t.mean().item(), conf_t.quantile(0.25).item(), conf_t.quantile(0.75).item()
        else:
            m, q25, q75 = None, None, None
        records["conf_ood"].append(m); records["conf_ood_q25"].append(q25); records["conf_ood_q75"].append(q75)

        records["change_ood"].append(change_ood)

    (out_dir / "results.json").write_text(json.dumps(records, indent=2))
    print(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
