from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interpn

from pymatviz.utils import NumArray, add_mae_r2_box, with_hist


def hist_density(
    xs: NumArray, ys: NumArray, sort: bool = True, bins: int = 100
) -> tuple[NumArray, NumArray, NumArray]:
    """Return an approximate density of 2d points.

    Args:
        xs (array): x-coordinates of points
        ys (array): y-coordinates of points
        sort (bool, optional): Whether to sort points by density so that densest points
            are plotted last. Defaults to True.
        bins (int, optional): Number of bins (histogram resolution). Defaults to 100.

    Returns:
        tuple[array, array]: x- and y-coordinates (sorted by density) as well as density
            itself
    """
    data, x_e, y_e = np.histogram2d(xs, ys, bins=bins)

    zs = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([xs, ys]).T,
        method="splinef2d",
        bounds_error=False,
    )

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = zs.argsort()
        xs, ys, zs = xs[idx], ys[idx], zs[idx]

    return xs, ys, zs


def density_scatter(
    xs: NumArray,
    ys: NumArray,
    ax: Axes = None,
    color_map: str = "Blues",
    sort: bool = True,
    log: bool = True,
    density_bins: int = 100,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    identity: bool = True,
    stats: bool = True,
    **kwargs: Any,
) -> Axes:
    """Scatter plot colored (and optionally sorted) by density.

    Args:
        xs (array): x values.
        ys (array): y values.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        color_map (str, optional): plt color map or valid string name.
            Defaults to "Blues".
        sort (bool, optional): Whether to sort the data. Defaults to True.
        log (bool, optional): Whether to the color scale. Defaults to True.
        density_bins (int, optional): How many density_bins to use for the density
            histogram, i.e. granularity of the density color scale. Defaults to 100.
        xlabel (str, optional): x-axis label. Defaults to "Actual".
        ylabel (str, optional): y-axis label. Defaults to "Predicted".
        identity (bool, optional): Whether to add an identity/parity line (y = x).
            Defaults to True.
        stats (bool, optional): Whether to display a text box with MAE and R^2.
            Defaults to True.

    Returns:
        ax: The plot's matplotlib Axes.
    """
    if ax is None:
        ax = plt.gca()

    xs, ys, cs = hist_density(xs, ys, sort=sort, bins=density_bins)

    norm = mpl.colors.LogNorm() if log else None

    ax.scatter(xs, ys, c=cs, cmap=color_map, norm=norm, **kwargs)

    if identity:
        ax.axline(
            (0, 0), (1, 1), alpha=0.5, zorder=0, linestyle="dashed", color="black"
        )

    if stats:
        add_mae_r2_box(xs, ys, ax)

    ax.set(xlabel=xlabel, ylabel=ylabel)

    return ax


def scatter_with_err_bar(
    xs: NumArray,
    ys: NumArray,
    xerr: NumArray = None,
    yerr: NumArray = None,
    ax: Axes = None,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    title: str = None,
    **kwargs: Any,
) -> Axes:
    """Scatter plot with optional x- and/or y-error bars. Useful when passing model
    uncertainties as yerr=y_std for checking if uncertainty correlates with error,
    i.e. if points farther from the parity line have larger uncertainty.

    Args:
        xs (array): x-values
        ys (array): y-values
        xerr (array, optional): Horizontal error bars. Defaults to None.
        yerr (array, optional): Vertical error bars. Defaults to None.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        xlabel (str, optional): x-axis label. Defaults to "Actual".
        ylabel (str, optional): y-axis label. Defaults to "Predicted".
        title (str, optional): Plot tile. Defaults to None.

    Returns:
        ax: The plot's matplotlib Axes.
    """
    if ax is None:
        ax = plt.gca()

    styles = dict(markersize=6, fmt="o", ecolor="g", capthick=2, elinewidth=2)
    ax.errorbar(xs, ys, yerr=yerr, xerr=xerr, **kwargs, **styles)

    # identity line
    ax.axline((0, 0), (1, 1), alpha=0.5, zorder=0, linestyle="dashed", color="black")

    add_mae_r2_box(xs, ys, ax)

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    return ax


def density_hexbin(
    xs: NumArray,
    yx: NumArray,
    ax: Axes = None,
    weights: NumArray = None,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    **kwargs: Any,
) -> Axes:
    """Hexagonal-grid scatter plot colored by point density or by density in third
    dimension passed as weights.

    Args:
        xs (array): x values
        yx (array): y values
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        weights (array, optional): If given, these values are accumulated in the bins.
            Otherwise, every point has value 1. Must be of the same length as x and y.
            Defaults to None.
        xlabel (str, optional): x-axis label. Defaults to "Actual".
        ylabel (str, optional): y-axis label. Defaults to "Predicted".

    Returns:
        ax: The plot's matplotlib Axes.
    """
    if ax is None:
        ax = plt.gca()

    # the scatter plot
    hexbin = ax.hexbin(xs, yx, gridsize=75, mincnt=1, bins="log", C=weights, **kwargs)

    cb_ax = ax.inset_axes([0.95, 0.03, 0.03, 0.7])  # [left, bottom, width, height]
    plt.colorbar(hexbin, cax=cb_ax)
    cb_ax.yaxis.set_ticks_position("left")

    # identity line
    ax.axline((0, 0), (1, 1), alpha=0.5, zorder=0, linestyle="dashed", color="black")

    add_mae_r2_box(xs, yx, ax, loc="upper left")

    ax.set(xlabel=xlabel, ylabel=ylabel)

    return ax


def density_scatter_with_hist(
    xs: NumArray,
    ys: NumArray,
    cell: GridSpec = None,
    bins: int = 100,
    **kwargs: Any,
) -> Axes:
    """Scatter plot colored (and optionally sorted) by density
    with histograms along each dimension
    """
    ax_scatter = with_hist(xs, ys, cell, bins)
    ax = density_scatter(xs, ys, ax_scatter, **kwargs)

    return ax


def density_hexbin_with_hist(
    xs: NumArray,
    ys: NumArray,
    cell: GridSpec = None,
    bins: int = 100,
    **kwargs: Any,
) -> Axes:
    """Hexagonal-grid scatter plot colored by density or by third dimension
    passed color_by with histograms along each dimension.
    """
    ax_scatter = with_hist(xs, ys, cell, bins)
    ax = density_hexbin(xs, ys, ax_scatter, **kwargs)

    return ax


def residual_vs_actual(y_true: NumArray, y_pred: NumArray, ax: Axes = None) -> Axes:
    """Plot ground truth targets on the x-axis against residuals
    (y_err = y_true - y_pred) on the y-axis.

    Args:
        y_true (array): Ground truth values
        y_pred (array): Model predictions
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.

    Returns:
        ax: The plot's matplotlib Axes.
    """
    if ax is None:
        ax = plt.gca()

    y_err = y_true - y_pred

    plt.plot(y_true, y_err, "o", alpha=0.5, label=None, mew=1.2, ms=5.2)
    plt.axline(
        [1, 0], [2, 0], linestyle="dashed", color="black", alpha=0.5, label="ideal"
    )

    plt.ylabel(r"Residual ($y_\mathrm{test} - y_\mathrm{pred}$)")
    plt.xlabel("Actual value")
    plt.legend(loc="lower right")

    return ax
