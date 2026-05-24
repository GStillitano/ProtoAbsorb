"""Experiment 3: Layer-wise BN affine drift over the stream.

Pure checkpoint analysis — no forward pass, no data loading.
Outputs results.json. Use scripts/plot.py --exp 3 to visualise.

Usage:
    uv run python scripts/exp3_layerwise.py --stream tent/gaussian_noise_svhn_c_0.50_seed0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.bn_affine import load as load_ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", required=True)
    args = parser.parse_args()

    ckpt_dir = Path("checkpoints") / args.stream
    meta     = json.loads((ckpt_dir / "meta.json").read_text())
    T        = meta["T"]
    out_dir  = Path("results") / args.stream / "exp3_layerwise"
    out_dir.mkdir(parents=True, exist_ok=True)

    state0      = load_ckpt(ckpt_dir / "theta_000.pt")
    layer_names = list(state0["gamma"].keys())
    L           = len(layer_names)

    gamma_drift = np.zeros((T + 1, L))
    beta_drift  = np.zeros((T + 1, L))
    cv_gamma    = np.zeros((T + 1, L))

    for t in range(T + 1):
        state = load_ckpt(ckpt_dir / f"theta_{t:03d}.pt")
        for li, name in enumerate(layer_names):
            dg = (state["gamma"][name] - state0["gamma"][name]).numpy()
            db = (state["beta"][name]  - state0["beta"][name]).numpy()

            gamma_drift[t, li] = float(np.linalg.norm(dg))
            beta_drift[t, li]  = float(np.linalg.norm(db))
            cv_gamma[t, li]    = float(dg.std() / (abs(dg.mean()) + 1e-8))

    records = {
        "layer_names": layer_names,
        "gamma_drift": gamma_drift.tolist(),
        "beta_drift":  beta_drift.tolist(),
        "cv_gamma":    cv_gamma.tolist(),
    }
    (out_dir / "results.json").write_text(json.dumps(records, indent=2))
    print(f"Saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
