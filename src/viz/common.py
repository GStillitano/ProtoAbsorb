"""Shared matplotlib style for all experiment plots."""
import matplotlib.pyplot as plt

METHOD_COLORS = {
    "tent":     "#E65100",
    "bn_adapt": "#1565C0",
    "unient":   "#6A1B9A",
    "rosetta":  "#2E7D32",
    "fix":      "#F9A825",
}

POP_COLORS = {
    "csID":  "#1565C0",
    "csOOD": "#C62828",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi":        150,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "font.size":         10,
    })
