from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import norm

from pymatviz.utils import NumArray


def qq_gaussian(
    y_true: NumArray,
    y_pred: NumArray,
    y_std: NumArray | dict[str, NumArray],
    ax: Axes = None,
) -> Axes:
    """Plot the Gaussian quantile-quantile (Q-Q) plot of one (passed as array)
    or multiple (passed as dict) sets of uncertainty estimates for a single
    pair of ground truth targets `y_true` and model predictions `y_pred`.

    Overconfidence relative to a Gaussian distribution is visualized as shaded
    areas below the parity line, underconfidence (oversized uncertainties) as
    shaded areas above the parity line.

    The measure of calibration is how well the uncertainty percentiles conform
    to those of a normal distribution.

    Inspired by https://git.io/JufOz.
    Info on Q-Q plots: https://wikipedia.org/wiki/Q-Q_plot

    Args:
        y_true (array): ground truth targets
        y_pred (array): model predictions
        y_std (array | dict[str, array]): model uncertainties
        ax (Axes): matplotlib Axes on which to plot. Defaults to None.

    Returns:
        ax: The plot's matplotlib Axes.
    """
    if ax is None:
        ax = plt.gca()

    if isinstance(y_std, np.ndarray):
        y_std = {"std": y_std}

    res = np.abs(y_pred - y_true)
    resolution = 100

    lines = []  # collect plotted lines to show second legend with miscalibration areas
    for key, std in y_std.items():

        z_scored = (np.array(res) / std).reshape(-1, 1)

        exp_proportions = np.linspace(0, 1, resolution)
        gaussian_upper_bound = norm.ppf(0.5 + exp_proportions / 2)
        obs_proportions = np.mean(z_scored <= gaussian_upper_bound, axis=0)

        [line] = ax.plot(
            exp_proportions, obs_proportions, linewidth=2, alpha=0.8, label=key
        )
        ax.fill_between(
            exp_proportions, y1=obs_proportions, y2=exp_proportions, alpha=0.2
        )
        miscal_area = np.trapz(
            np.abs(obs_proportions - exp_proportions), dx=1 / resolution
        )
        lines.append([line, miscal_area])

    # identity line
    ax.axline((0, 0), (1, 1), alpha=0.5, zorder=0, linestyle="dashed", color="black")

    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set(xlabel="Theoretical Quantile", ylabel="Observed Quantile")

    legend1 = ax.legend(loc="upper left", frameon=False)
    # Multiple legends on the same axes:
    # https://matplotlib.org/3.3.3/tutorials/intermediate/legend_guide.html#multiple-legends-on-the-same-axes
    ax.add_artist(legend1)

    lines, areas = zip(*lines)

    if len(lines) > 1:
        legend2 = ax.legend(
            lines,
            [f"{area:.2f}" for area in areas],
            title="Miscalibration areas",
            loc="lower right",
            ncol=2,
            frameon=False,
        )
        legend2._legend_box.align = "left"  # https://stackoverflow.com/a/44620643
    else:
        ax.legend(
            lines,
            [f"Miscalibration area: {areas[0]:.2f}"],
            loc="lower right",
            frameon=False,
        )

    return ax
