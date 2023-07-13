from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np


if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def cumulative_residual(
    res: ArrayLike, ax: plt.Axes | None = None, **kwargs: Any
) -> plt.Axes:
    """Plot the empirical cumulative distribution for the residuals (y - mu).

    Args:
        res (array): Residuals between y_true and y_pred, i.e.
            targets - model predictions.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        **kwargs: Additional keyword arguments passed to ax.fill_between().

    Returns:
        ax: The plot's matplotlib Axes.
    """
    ax = ax or plt.gca()

    res = np.sort(res)
    n_data = len(res)

    # Plot the empirical distribution
    ax.plot(res, np.arange(n_data) / n_data * 100)

    # Fill the 90% coverage region
    # TODO may look better to add drop downs instead
    low = int(0.05 * (n_data - 1) + 0.5)
    up = int(0.95 * (n_data - 1) + 0.5)
    ax.fill_between(
        res[low:up],
        (np.arange(n_data) / n_data * 100)[low:up],
        alpha=kwargs.pop("alpha", 0.3),
        **kwargs,
    )

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

    return ax


def cumulative_error(
    abs_err: ArrayLike, ax: plt.Axes | None = None, **kwargs: Any
) -> plt.Axes:
    """Plot the empirical cumulative distribution of the absolute errors.

    abs(y_true - y_pred).

    Args:
        abs_err (array): Absolute error between y_true and y_pred, i.e.
            abs(targets - model predictions).
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        **kwargs: Additional keyword arguments passed to ax.plot().

    Returns:
        ax: The plot's matplotlib Axes.
    """
    ax = ax or plt.gca()

    errors = np.sort(abs_err)
    n_data = len(errors)

    # Plot the empirical distribution
    ax.plot(errors, np.arange(n_data) / n_data * 100, **kwargs)

    # Get robust (and symmetrical) x-axis limits
    lim = np.percentile(errors, 98)
    ax.set(xlim=(0, lim), ylim=(0, 100))

    line_kwargs = dict(linestyle="--", color="grey", alpha=0.4)
    # Add some visual guidelines
    for percentile in [50, 75]:
        percent = int(percentile * (n_data - 1) / 100 + 0.5)
        ax.plot((0, errors[percent]), (percentile, percentile), **line_kwargs)
        ax.plot((errors[percent], errors[percent]), (0, percentile), **line_kwargs)

    # Label the plot
    ax.set(xlabel="Absolute Error", ylabel="Percentile", title="Cumulative Error")

    return ax
