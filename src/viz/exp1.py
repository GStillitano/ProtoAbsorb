"""Plots for Experiment 1: AUROC + csID accuracy trajectory."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.viz.common import apply_style, METHOD_COLORS


def plot(stream_results: dict[str, dict], out_path: Path) -> None:
    """Overlay AUROC and csID accuracy for one or more streams.

    Args:
        stream_results: {stream_id: results_dict} where results_dict has keys
                        "t", "auroc", "acc_csid".
        out_path: destination PNG path (parent created if needed).
    """
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for sid, r in stream_results.items():
        method = sid.split("/")[0]
        color  = METHOD_COLORS.get(method)
        t      = r["t"]

        auroc_vals = r["auroc"]
        if any(v is not None for v in auroc_vals):
            t_valid = [ti for ti, v in zip(t, auroc_vals) if v is not None]
            a_valid = [v  for v in auroc_vals if v is not None]
            sns.lineplot(x=t_valid, y=a_valid, label=method, color=color,
                         marker="o", markersize=3, ax=ax1)

        sns.lineplot(x=t, y=r["acc_csid"], label=method, color=color,
                     marker="o", markersize=3, ax=ax2)

    ax1.set_xlabel("Step $t$")
    ax1.set_ylabel("AUROC")
    ax1.set_title("AUROC over stream")
    ax1.legend()

    ax2.set_xlabel("Step $t$")
    ax2.set_ylabel("Accuracy (csID)")
    ax2.set_title("csID accuracy over stream")
    ax2.legend()

    sns.despine(fig=fig)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
