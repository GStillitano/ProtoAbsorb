"""Experiment 3: Layer-wise BN affine drift over the stream.

Pure checkpoint analysis — no forward pass, no data loading.

Usage:
    uv run python scripts/exp3_layerwise.py --stream tent/gaussian_noise_svhn_c_0.50_seed0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
    _plot(gamma_drift, beta_drift, cv_gamma, layer_names, T, out_dir)
    print(f"Saved to {out_dir}")


def _plot(gamma_drift, beta_drift, cv_gamma, layer_names, T, out_dir):
    short = [n.rsplit(".", 2)[-2] + "." + n.rsplit(".", 1)[-1] if "." in n else n for n in layer_names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for data, ax, title in [
        (gamma_drift, axes[0], "‖Δγ‖₂"),
        (beta_drift,  axes[1], "‖Δβ‖₂"),
        (cv_gamma,    axes[2], "CV(Δγ)"),
    ]:
        im = ax.imshow(data.T, aspect="auto", origin="lower",
                       extent=[0, T, -0.5, len(layer_names) - 0.5])
        ax.set_xlabel("Step t"); ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(short, fontsize=6); ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_heatmap.png", dpi=150)
    plt.close(fig)

    # Layer profile at t=T
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(range(len(layer_names)), gamma_drift[-1], label="‖Δγ‖")
    ax2.bar(range(len(layer_names)), beta_drift[-1],  label="‖Δβ‖", alpha=0.6)
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(short, rotation=90, fontsize=6)
    ax2.set_title(f"BN drift at t={T}"); ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "exp3_profile.png", dpi=150)
    plt.close(fig2)


if __name__ == "__main__":
    main()
