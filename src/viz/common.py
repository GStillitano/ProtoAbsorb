"""Shared seaborn/matplotlib style for all experiment plots."""
import matplotlib.pyplot as plt
import seaborn as sns

METHOD_COLORS = {
    "tent":     "#E65100",
    "bn_adapt": "#1565C0",
    "nova-tta":  "#2E7D32",
}

POP_COLORS = {
    "csID":  "#1565C0",
    "csOOD": "#C62828",
}


def apply_style() -> None:
    sns.set_theme(
        style="ticks",
        context="notebook",
        font_scale=1.05,
        rc={
            "figure.dpi":      150,
            "axes.grid":       True,
            "grid.alpha":      0.25,
            "grid.linewidth":  0.6,
            "lines.linewidth": 2.0,
            "legend.frameon":  False,
        },
    )
