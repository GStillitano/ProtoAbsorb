"""Plots for Experiment 2: norm, cosine, distance, OOD confidence over stream."""
from pathlib import Path

import matplotlib.pyplot as plt

from src.viz.common import apply_style, POP_COLORS


def plot(results: dict, out_dir: Path) -> None:
    """Six-panel geometry plot from exp2_geometry results.

    Args:
        results: dict loaded from exp2_geometry/results.json.
        out_dir: directory for output PNG (created if needed).
    """
    apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t   = results["t"]
    id_c  = POP_COLORS["csID"]
    ood_c = POP_COLORS["csOOD"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(t, results["norm_id"],  label="csID",  color=id_c)
    axes[0, 0].plot(t, results["norm_ood"], label="csOOD", color=ood_c)
    axes[0, 0].set_title("Feature norm")
    axes[0, 0].legend()

    axes[0, 1].plot(t, results["delta_norm"], color="black")
    axes[0, 1].axhline(0, linestyle="--", linewidth=0.8, color="grey")
    axes[0, 1].set_title("Norm gap (csID − csOOD)")

    axes[0, 2].plot(t, results["cos_id"],     label="cos csID",     color=id_c)
    axes[0, 2].plot(t, results["cos_ood"],    label="cos csOOD",    color=ood_c)
    axes[0, 2].plot(t, results["maxcos_id"],  label="maxcos csID",  color=id_c,  linestyle="--")
    axes[0, 2].plot(t, results["maxcos_ood"], label="maxcos csOOD", color=ood_c, linestyle="--")
    axes[0, 2].set_title("Cosine alignment")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(t, results["dist_id"],  label="csID",  color=id_c)
    axes[1, 0].plot(t, results["dist_ood"], label="csOOD", color=ood_c)
    axes[1, 0].set_title("Distance to nearest source centroid")
    axes[1, 0].legend()

    axes[1, 1].plot(t, results["conf_ood"], color=ood_c)
    axes[1, 1].set_title("Mean max confidence (csOOD)")

    axes[1, 2].plot(t, results["change_ood"], color="black")
    axes[1, 2].set_title("Fraction csOOD pred changed vs prev step")

    for ax in axes.flat:
        ax.set_xlabel("Step t")

    fig.tight_layout()
    out_path = out_dir / "exp2_geometry.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
