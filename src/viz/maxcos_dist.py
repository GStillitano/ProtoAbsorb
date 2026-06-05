"""Distribution plot: per-sample maxcos for csID vs csOOD at a fixed step."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.viz.common import apply_style, POP_COLORS


def plot(results: dict, out_path: Path) -> None:
    """Overlay csID / csOOD maxcos distributions (KDE + histogram)."""
    apply_style()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    maxcos_id  = results["maxcos_id"]
    maxcos_ood = results["maxcos_ood"]
    t          = results.get("t", 0)
    id_c       = POP_COLORS["csID"]
    ood_c      = POP_COLORS["csOOD"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for vals, label, color in [
        (maxcos_id,  "csID",  id_c),
        (maxcos_ood, "csOOD", ood_c),
    ]:
        if not vals:
            continue
        sns.histplot(vals, bins=40, stat="density", color=color,
                     alpha=0.25, edgecolor=None, ax=ax)
        sns.kdeplot(vals, color=color, label=label, ax=ax, linewidth=2.0)
        ax.axvline(sum(vals) / len(vals), color=color, linestyle="--",
                   linewidth=1.2, alpha=0.8)

    ax.set_xlabel("max cosine to class weights")
    ax.set_ylabel("density")
    ax.set_title(f"maxcos distribution  (t={t}, frozen + BN adapt, no grad step)")
    ax.legend()
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
