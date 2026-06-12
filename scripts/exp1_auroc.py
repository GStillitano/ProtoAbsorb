"""Experiment 1: AUROC and csID accuracy trajectory over the adaptation stream.

Outputs results.json per stream. Use scripts/plot.py --exp 1 to visualise.

Usage:
    uv run python scripts/exp1_auroc.py --streams tent/gaussian_noise_svhn_c_open_seed0 bn_adapt/gaussian_noise_svhn_c_open_seed0
"""
import argparse
import json
from pathlib import Path

import torch

from src.model import load_model
from src.bn_affine import evaluate_diagnostic_stream
from src.data.diagnostic import load_diagnostic
from src.metrics.ood_scores import energy_score
from src.metrics.ood_metrics import auroc
from src.device import get_device


def run_stream(stream_id: str, model, data_dir: str, device: str) -> dict:
    ckpt_dir = Path("checkpoints") / stream_id
    meta     = json.loads((ckpt_dir / "meta.json").read_text())
    T        = meta["T"]
    open_set = meta["open_set"]

    x_id, y_id, x_ood = load_diagnostic(meta, data_dir)
    x_ood_eval = x_ood if open_set else torch.empty(0)

    auroc_list, acc_list = [], []
    for t in range(T + 1):
        ckpt = ckpt_dir / f"theta_{t:03d}.pt"
        feat_id, logits_id, feat_ood, logits_ood = evaluate_diagnostic_stream(
            model, ckpt, x_id, x_ood_eval, device,
        )
        acc_list.append((logits_id.argmax(dim=-1) == y_id).float().mean().item())
        if open_set:
            scores_id  = -energy_score(logits_id)
            scores_ood = -energy_score(logits_ood)
            auroc_list.append(auroc(scores_id, scores_ood))
        else:
            auroc_list.append(None)

    return {"t": list(range(T + 1)), "auroc": auroc_list, "acc_csid": acc_list, "open_set": open_set}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", nargs="+", required=True)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--device",   default=None)
    args = parser.parse_args()

    device = args.device or get_device()
    model  = load_model(data_dir=args.data_dir).to(device)

    for sid in args.streams:
        results = run_stream(sid, model, args.data_dir, device)
        out_dir = Path("results") / sid / "exp1_auroc"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
