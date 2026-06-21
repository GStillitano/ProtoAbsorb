"""Plots for Experiment 2: norm, cosine, distance, OOD confidence over stream."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.viz.common import apply_style, POP_COLORS


def _lineplot(ax, t, values, label=None, color=None, linestyle="-", marker=None):
    if not values:
        return
    t_v = [ti for ti, v in zip(t, values) if v is not None]
    y_v = [v for v in values if v is not None]
    if not t_v:
        return
    sns.lineplot(
        x=t_v,
        y=y_v,
        label=label,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=3,
        ax=ax,
    )


def _bandplot(ax, t, q25, q75, color, alpha=0.15):
    if not q25 or not q75:
        return
    t_v = [ti for ti, a, b in zip(t, q25, q75) if a is not None and b is not None]
    q25_v = [a for a in q25 if a is not None]
    q75_v = [b for b in q75 if b is not None]
    if not t_v:
        return
    ax.fill_between(t_v, q25_v, q75_v, color=color, alpha=alpha, linewidth=0)


def _get(r, key, n):
    return r.get(key) or [None] * n


def plot(results: dict, out_dir: Path) -> None:
    apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = results["t"]
    n = len(t)
    id_c = POP_COLORS["csID"]
    ood_c = POP_COLORS["csOOD"]

    # Figure 1: Norms
    fig1, axes = plt.subplots(2, 2, figsize=(12, 8))

    # L2 id vs ood
    ax = axes[0, 0]
    _bandplot(
        ax, t, _get(results, "norm_id_q25", n), _get(results, "norm_id_q75", n), id_c
    )
    _bandplot(
        ax, t, _get(results, "norm_ood_q25", n), _get(results, "norm_ood_q75", n), ood_c
    )
    _lineplot(ax, t, results["norm_id"], label="csID", color=id_c, marker="o")
    _lineplot(ax, t, results["norm_ood"], label="csOOD", color=ood_c, marker="o")
    ax.set_title("L2 norm (band=IQR)")
    ax.legend()

    # L2 gap
    ax = axes[0, 1]
    _lineplot(ax, t, results["delta_norm"], color="#333333", marker="o")
    ax.axhline(0, linestyle="--", linewidth=1.0, color="grey", alpha=0.7)
    ax.set_title("L2 norm gap  csID − csOOD")

    # L1 id vs ood
    ax = axes[1, 0]
    _bandplot(
        ax,
        t,
        _get(results, "norm_l1_id_q25", n),
        _get(results, "norm_l1_id_q75", n),
        id_c,
    )
    _bandplot(
        ax,
        t,
        _get(results, "norm_l1_ood_q25", n),
        _get(results, "norm_l1_ood_q75", n),
        ood_c,
    )
    _lineplot(
        ax, t, _get(results, "norm_l1_id", n), label="csID", color=id_c, marker="o"
    )
    _lineplot(
        ax, t, _get(results, "norm_l1_ood", n), label="csOOD", color=ood_c, marker="o"
    )
    ax.set_title("L1 norm (band=IQR)")
    ax.legend()

    # L1 gap
    ax = axes[1, 1]
    _lineplot(ax, t, _get(results, "delta_norm_l1", n), color="#333333", marker="o")
    ax.axhline(0, linestyle="--", linewidth=1.0, color="grey", alpha=0.7)
    ax.set_title("L1 norm gap  csID − csOOD")

    for ax in axes.flat:
        ax.set_xlabel("Step $t$")
    sns.despine(fig=fig1)
    fig1.tight_layout()
    p1 = out_dir / "geometry1.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {p1}")

    # Figure 2: Alignment
    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Cosine id vs ood
    ax = axes[0, 0]
    _bandplot(
        ax, t, _get(results, "cos_id_q25", n), _get(results, "cos_id_q75", n), id_c
    )
    _bandplot(
        ax, t, _get(results, "cos_ood_q25", n), _get(results, "cos_ood_q75", n), ood_c
    )
    _lineplot(ax, t, results["cos_id"], label="csID", color=id_c)
    _lineplot(ax, t, results["cos_ood"], label="csOOD", color=ood_c)
    ax.set_title("Cosine alignment (band=IQR)")
    ax.legend()

    # Max cosine id vs ood
    ax = axes[0, 1]
    _bandplot(
        ax,
        t,
        _get(results, "maxcos_id_q25", n),
        _get(results, "maxcos_id_q75", n),
        id_c,
    )
    _bandplot(
        ax,
        t,
        _get(results, "maxcos_ood_q25", n),
        _get(results, "maxcos_ood_q75", n),
        ood_c,
    )
    _lineplot(ax, t, results["maxcos_id"], label="csID", color=id_c)
    _lineplot(ax, t, results["maxcos_ood"], label="csOOD", color=ood_c)
    ax.set_title("Max cosine alignment (band=IQR)")
    ax.legend()

    # Centroid distance id vs ood
    ax = axes[1, 0]
    _bandplot(
        ax, t, _get(results, "dist_id_q25", n), _get(results, "dist_id_q75", n), id_c
    )
    _bandplot(
        ax, t, _get(results, "dist_ood_q25", n), _get(results, "dist_ood_q75", n), ood_c
    )
    _lineplot(ax, t, results["dist_id"], label="csID", color=id_c, marker="o")
    _lineplot(ax, t, results["dist_ood"], label="csOOD", color=ood_c, marker="o")
    ax.set_title("Distance to nearest centroid (band=IQR)")
    ax.legend()

    # Max confidence ood
    ax = axes[1, 1]
    _bandplot(
        ax, t, _get(results, "conf_ood_q25", n), _get(results, "conf_ood_q75", n), ood_c
    )
    _lineplot(ax, t, results["conf_ood"], color=ood_c, marker="o")
    ax.set_title("Max confidence csOOD (band=IQR)")

    for ax in axes.flat:
        ax.set_xlabel("Step $t$")
    sns.despine(fig=fig2)
    fig2.tight_layout()
    p2 = out_dir / "geometry2.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {p2}")
