"""Plots for Experiment 2: norm, cosine, distance, OOD confidence over stream."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.viz.common import apply_style, POP_COLORS


def _lineplot(ax, t, values, label=None, color=None, linestyle="-", marker=None):
    """Plot only non-None values."""
    t_v = [ti for ti, v in zip(t, values) if v is not None]
    y_v = [v  for v in values if v is not None]
    if not t_v:
        return
    sns.lineplot(x=t_v, y=y_v, label=label, color=color,
                 linestyle=linestyle, marker=marker, markersize=3, ax=ax)


def plot(results: dict, out_dir: Path) -> None:
    """Six-panel geometry plot from exp2_geometry results.

    Args:
        results: dict loaded from exp2_geometry/results.json.
        out_dir: directory for output PNG (created if needed).
    """
    apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t     = results["t"]
    id_c  = POP_COLORS["csID"]
    ood_c = POP_COLORS["csOOD"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Feature norm
    _lineplot(axes[0, 0], t, results["norm_id"],  label="csID",  color=id_c,  marker="o")
    _lineplot(axes[0, 0], t, results["norm_ood"], label="csOOD", color=ood_c, marker="o")
    axes[0, 0].set_title("Feature norm")
    axes[0, 0].legend()

    # Norm gap
    _lineplot(axes[0, 1], t, results["delta_norm"], color="#333333", marker="o")
    axes[0, 1].axhline(0, linestyle="--", linewidth=1.0, color="grey", alpha=0.7)
    axes[0, 1].set_title("Norm gap (csID − csOOD)")

    # Cosine alignment
    _lineplot(axes[0, 2], t, results["cos_id"],     label="cos csID",     color=id_c)
    _lineplot(axes[0, 2], t, results["cos_ood"],    label="cos csOOD",    color=ood_c)
    _lineplot(axes[0, 2], t, results["maxcos_id"],  label="maxcos csID",  color=id_c,  linestyle="--")
    _lineplot(axes[0, 2], t, results["maxcos_ood"], label="maxcos csOOD", color=ood_c, linestyle="--")
    axes[0, 2].set_title("Cosine alignment")
    axes[0, 2].legend(fontsize=8)

    # Centroid distance
    _lineplot(axes[1, 0], t, results["dist_id"],  label="csID",  color=id_c,  marker="o")
    _lineplot(axes[1, 0], t, results["dist_ood"], label="csOOD", color=ood_c, marker="o")
    axes[1, 0].set_title("Distance to nearest source centroid")
    axes[1, 0].legend()

    # OOD confidence
    _lineplot(axes[1, 1], t, results["conf_ood"], color=ood_c, marker="o")
    axes[1, 1].set_title("Mean max confidence (csOOD)")

    # Prediction change
    _lineplot(axes[1, 2], t, results["change_ood"], color="#333333", marker="o")
    axes[1, 2].set_title("Fraction csOOD pred changed vs prev step")

    for ax in axes.flat:
        ax.set_xlabel("Step $t$")

    sns.despine(fig=fig)
    fig.tight_layout()
    out_path = out_dir / "exp2_geometry.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
