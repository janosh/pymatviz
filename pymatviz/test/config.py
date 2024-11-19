"""Quick matplotlib config functions."""

from __future__ import annotations

import matplotlib.pyplot as plt


def config_matplotlib() -> None:
    """Set default matplotlib configurations for consistency.
    - Font size: 14 for readability.
    - Savefig: Tight bounding box and 200 DPI for high-quality saved plots.
    - Axes: Title size 16, bold weight for emphasis.
    - Figure: DPI 200, title size 20, bold weight for better visibility.
    - Layout: Enables constrained layout to reduce element overlap.
    """
    plt.rc("font", size=14)
    plt.rc("savefig", bbox="tight", dpi=200)
    plt.rc("axes", titlesize=16, titleweight="bold")
    plt.rc("figure", dpi=200, titlesize=20, titleweight="bold")
    plt.rcParams["figure.constrained_layout.use"] = True
