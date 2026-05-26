"""Plots for Experiment 3: layer-wise BN affine drift.

x-axis on all plots: layer index, left = closest to input, right = closest to embedding.

WideResNet-40-2 (Hendrycks2020AugMix_WRN) BN layout:
  block1.layer.{0-5}.{bn1,bn2}  →  indices  0-11  (32-ch,  32×32 spatial)
  block2.layer.{0-5}.{bn1,bn2}  →  indices 12-23  (64-ch,  16×16 spatial)
  block3.layer.{0-5}.{bn1,bn2}  →  indices 24-35  (128-ch,  8×8 spatial)
  bn1 (final, pre-AvgPool)       →  index   36     (128-ch,   embedding)

Each BasicBlock: bn1 → ReLU → Conv(3×3) → bn2 → ReLU → Conv(3×3) + skip.
No dropout. Shortcut = bare Conv(1×1), no BN, only on unit 0 of each block.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from src.viz.common import apply_style

_BLOCK_SPANS = [
    (0,  11, "block1\n(32ch, 32×32)",  "#BBDEFB"),
    (12, 23, "block2\n(64ch, 16×16)",  "#C8E6C9"),
    (24, 35, "block3\n(128ch, 8×8)",   "#FFE0B2"),
    (36, 36, "bn_out\n(embedding)",    "#F8BBD0"),
]


def _short(name: str) -> str:
    """block1.layer.2.bn1 → B1·2·1,  bn1 (final) → BN_out."""
    if "." not in name:
        return "BN_out"
    parts = name.split(".")
    block = parts[0][-1]
    unit  = parts[2]
    bn    = parts[3][-1]
    return f"B{block}·{unit}·{bn}"


def _block_of(idx: int) -> tuple[str, str]:
    for lo, hi, label, color in _BLOCK_SPANS:
        if lo <= idx <= hi:
            return label, color
    return "?", "white"


def _add_vertical_block_bands(ax, L: int, heatmap: bool = False) -> None:
    """Vertical colour bands + block labels.

    heatmap=True: sns.heatmap coords where column i occupies [i, i+1].
    heatmap=False: imshow/bar coords where column i is centred at i.
    """
    for lo, hi, label, color in _BLOCK_SPANS:
        if lo >= L:
            break
        hi_clip = min(hi, L - 1)
        if heatmap:
            x0, x1 = lo, hi_clip + 1
        else:
            x0, x1 = lo - 0.5, hi_clip + 0.5
        ax.axvspan(x0, x1, color=color, alpha=0.20, zorder=0)
        ax.text(
            (x0 + x1) / 2, 0.97, label.split("\n")[0],
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7, color="dimgray",
        )


def _input_embedding_arrows(ax) -> None:
    ax.annotate(
        "← input", xy=(0, -0.13), xycoords="axes fraction",
        ha="left", va="top", fontsize=8, color="steelblue", fontweight="bold",
    )
    ax.annotate(
        "embedding →", xy=(1, -0.13), xycoords="axes fraction",
        ha="right", va="top", fontsize=8, color="salmon", fontweight="bold",
    )


def plot(results: dict, out_dir: Path) -> None:
    """Heatmap (layer × t) and profile at t=T, both with input→embedding on x-axis."""
    apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_names = results["layer_names"]
    gamma_drift = np.array(results["gamma_drift"])   # (T+1, L)
    beta_drift  = np.array(results["beta_drift"])
    cv_gamma    = np.array(results["cv_gamma"])
    T = gamma_drift.shape[0] - 1
    L = len(layer_names)

    short      = [_short(n) for n in layer_names]
    bar_colors = [_block_of(i)[1] for i in range(L)]
    xticks     = np.arange(L)

    # ── Heatmaps: x = layer (input→embedding), y = step t ────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "BN affine drift over stream  —  x: layer (input → embedding),  y: step t",
        fontsize=11,
    )

    heatmap_specs = [
        (gamma_drift, axes[0], "‖Δγ‖₂  (gamma drift)",          "rocket"),
        (beta_drift,  axes[1], "‖Δβ‖₂  (beta drift)",           "mako"),
        (cv_gamma,    axes[2], "CV(Δγ)  (structured vs uniform)", "flare"),
    ]

    # y-tick thinning: show at most 10 labels
    ytick_step = max(1, (T + 1) // 10)
    yticklabels = [str(i) if i % ytick_step == 0 else "" for i in range(T + 1)]

    for data, ax, title, cmap in heatmap_specs:
        sns.heatmap(
            data,
            ax=ax,
            cmap=cmap,
            xticklabels=short,
            yticklabels=yticklabels,
            cbar_kws={"shrink": 0.75, "pad": 0.02},
            linewidths=0,
            rasterized=True,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Step $t$")
        ax.set_title(title)
        _add_vertical_block_bands(ax, L, heatmap=True)
        _input_embedding_arrows(ax)

    fig.tight_layout()
    heatmap_path = out_dir / "exp3_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {heatmap_path}")

    # ── Profile at t=T: x = layer (input→embedding) ──────────────────────────
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig2.suptitle(
        f"BN drift at t={T}  —  x: layer (input → embedding)",
        fontsize=11,
    )

    for ax, data, ylabel in [
        (axes2[0], gamma_drift[-1], "‖Δγ‖₂"),
        (axes2[1], beta_drift[-1],  "‖Δβ‖₂"),
    ]:
        ax.bar(xticks, data, color=bar_colors, edgecolor="white",
               linewidth=0.4, width=0.85, alpha=0.90)
        _add_vertical_block_bands(ax, L, heatmap=False)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.5, L - 0.5)

    axes2[1].set_xticks(xticks)
    axes2[1].set_xticklabels(short, rotation=90, fontsize=6)
    axes2[1].set_xlabel("Layer index")
    _input_embedding_arrows(axes2[1])

    patches = [
        mpatches.Patch(color=c, label=lbl.split("\n")[0])
        for lo, _, lbl, c in _BLOCK_SPANS if lo < L
    ]
    axes2[0].legend(handles=patches, fontsize=8, loc="upper left")

    sns.despine(fig=fig2)
    fig2.tight_layout()
    profile_path = out_dir / "exp3_profile.png"
    fig2.savefig(profile_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {profile_path}")
