"""Consistent Matplotlib styling for figures used in the report."""

import matplotlib.pyplot as plt


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25,
    })
