import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from pymatviz.utils import NumArray


def add_dropdown(ax: Axes, percentile: int, err: NumArray) -> None:
    """Add a dashed drop-down line at a given percentile.

    Args:
        ax (Axes): matplotlib Axes on which to add the dropdown.
        percentile (int): Integer in range(100) at which to display dropdown line.
        err (array): Numpy array of errors = abs(preds - targets).
    """
    percent = int(percentile * (len(err) - 1) / 100 + 0.5)
    ax.plot((0, err[percent]), (percentile, percentile), "--", color="grey", alpha=0.4)
    ax.plot(
        (err[percent], err[percent]), (0, percentile), "--", color="grey", alpha=0.4
    )


def cum_res(preds: NumArray, targets: NumArray, ax: Axes = None) -> Axes:
    """Plot the empirical cumulative distribution for the residuals (y - mu).

    Args:
        preds (array): Numpy array of predictions.
        targets (array): Numpy array of targets.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.

    Returns:
        ax: The plot's matplotlib Axes.
    """
    if ax is None:
        ax = plt.gca()

    res = np.sort(preds - targets)

    n_data = len(res)

    # Plot the empirical distribution
    ax.plot(res, np.arange(n_data) / n_data * 100)

    # Fill the 90% coverage region
    # TODO may look better to add drop downs instead
    low = int(0.05 * (n_data - 1) + 0.5)
    up = int(0.95 * (n_data - 1) + 0.5)
    ax.fill_between(res[low:up], (np.arange(n_data) / n_data * 100)[low:up], alpha=0.3)

    # Get robust (and symmetrical) x axis limits
    delta_low = res[low] - res[int(0.97 * low)]
    delta_up = res[int(1.03 * up)] - res[up]
    delta_max = max(delta_low, delta_up)
    lim = max(abs(res[up] + delta_max), abs(res[low] - delta_max))

    ax.set(xlim=(-lim, lim), ylim=(0, 100))

    # Add some visual guidelines
    ax.plot((0, 0), (0, 100), "--", color="grey", alpha=0.4)
    ax.plot((ax.get_xlim()[0], 0), (50, 50), "--", color="grey", alpha=0.4)

    # Label the plot
    ax.set(xlabel="Residual", ylabel="Percentile", title="Cumulative Residual")
    ax.legend(frameon=False)

    return ax


def cum_err(preds: NumArray, targets: NumArray, ax: Axes = None) -> Axes:
    """Plot the empirical cumulative distribution for the absolute errors abs(y - y_hat).

    Args:
        preds (array): Numpy array of predictions.
        targets (array): Numpy array of targets.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.

    Returns:
        ax: The plot's matplotlib Axes.
    """
    if ax is None:
        ax = plt.gca()

    err = np.sort(np.abs(preds - targets))
    n_data = len(err)

    # Plot the empirical distribution
    ax.plot(err, np.arange(n_data) / n_data * 100)

    # Get robust (and symmetrical) x axis limits
    lim = np.percentile(err, 98)
    ax.set(xlim=(0, lim), ylim=(0, 100))

    # Add some visual guidelines
    add_dropdown(ax, 50, err)
    add_dropdown(ax, 75, err)

    # Label the plot
    ax.set(xlabel="Absolute Error", ylabel="Percentile", title="Cumulative Error")
    ax.legend(frameon=False)

    return ax
