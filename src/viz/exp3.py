"""Plots for Experiment 3: layer-wise BN affine drift heatmaps and profile."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.viz.common import apply_style


def plot(results: dict, out_dir: Path) -> None:
    """Heatmap over (layer, t) and layer profile at t=T.

    Args:
        results: dict loaded from exp3_layerwise/results.json.
        out_dir: directory for output PNGs (created if needed).
    """
    apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_names  = results["layer_names"]
    gamma_drift  = np.array(results["gamma_drift"])   # (T+1, L)
    beta_drift   = np.array(results["beta_drift"])
    cv_gamma     = np.array(results["cv_gamma"])
    T            = gamma_drift.shape[0] - 1
    L            = len(layer_names)

    short = [
        n.rsplit(".", 2)[-2] + "." + n.rsplit(".", 1)[-1] if "." in n else n
        for n in layer_names
    ]

    # Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for data, ax, title in [
        (gamma_drift, axes[0], "‖Δγ‖₂"),
        (beta_drift,  axes[1], "‖Δβ‖₂"),
        (cv_gamma,    axes[2], "CV(Δγ)"),
    ]:
        im = ax.imshow(
            data.T, aspect="auto", origin="lower",
            extent=[0, T, -0.5, L - 0.5],
        )
        ax.set_xlabel("Step t")
        ax.set_yticks(range(L))
        ax.set_yticklabels(short, fontsize=6)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    heatmap_path = out_dir / "exp3_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {heatmap_path}")

    # Layer profile at t=T
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    x = range(L)
    ax2.bar(x, gamma_drift[-1], label="‖Δγ‖")
    ax2.bar(x, beta_drift[-1],  label="‖Δβ‖", alpha=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, rotation=90, fontsize=6)
    ax2.set_title(f"BN drift at t={T}")
    ax2.legend()
    fig2.tight_layout()
    profile_path = out_dir / "exp3_profile.png"
    fig2.savefig(profile_path, dpi=150)
    plt.close(fig2)
    print(f"Saved: {profile_path}")
