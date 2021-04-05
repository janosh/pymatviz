from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import ndarray as Array
from scipy.interpolate import interpn
from sklearn.metrics import r2_score

from ml_matrics.utils import add_identity, with_hist


def hist_density(
    xs: Array, ys: Array, sort: bool = True, bins: int = 100
) -> Tuple[Array]:
    """Return an approximate density of 2d points.

    Args:
        xs (Array): x-coordinates of points
        ys (Array): y-coordinates of points
        sort (bool, optional): Whether to sort points by density so that densest points
            are plotted last. Defaults to True.
        bins (int, optional): Number of bins (histogram resolution). Defaults to 100.

    Returns:
        tuple[Array]: x- and y-coordinates (sorted by density) as well as density itself.
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


def add_mae_r2_box(xs: Array, ys: Array, ax: Axes, loc: str = "lower right") -> None:

    mae_str = f"$\\mathrm{{MAE}} = {np.abs(xs - ys).mean():.3f}$\n"

    r2_str = f"$R^2 = {r2_score(xs, ys):.3f}$"

    text_box = AnchoredText(mae_str + r2_str, loc=loc, frameon=False)
    ax.add_artist(text_box)


def density_scatter(
    xs: Array,
    ys: Array,
    ax: Axes = None,
    color_map: str = "Blues",
    sort: bool = True,
    log: bool = True,
    density_bins: int = 100,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    identity: bool = True,
    stats: bool = True,
    **kwargs,
) -> Axes:
    """Scatter plot colored (and optionally sorted) by density.

    Args:
        xs (Array): x values.
        ys (Array): y values.
        ax (Axes, optional): plt axes. Defaults to None.
        color_map (str, optional): plt color map or valid string name. Defaults to "Blues".
        sort (bool, optional): Whether to sort the data. Defaults to True.
        log (bool, optional): Whether to the color scale. Defaults to True.
        density_bins (int, optional): How many density_bins to use for the density histogram,
            i.e. granularity of the density color scale. Defaults to 100.
        xlabel (str, optional): x-axis label. Defaults to "Actual".
        ylabel (str, optional): y-axis label. Defaults to "Predicted".
        identity (bool, optional): Whether to add an identity/parity line (y = x).
            Defaults to True.
        stats (bool, optional): Whether to display a text box with MAE and R^2.
            Defaults to True.

    Returns:
        Axes: plt axes with plotted data.
    """
    if ax is None:
        ax = plt.gca()

    xs, ys, cs = hist_density(xs, ys, sort=sort, bins=density_bins)

    norm = mpl.colors.LogNorm() if log else None

    ax.scatter(xs, ys, c=cs, cmap=color_map, norm=norm, **kwargs)
    if identity:
        add_identity(ax, label="ideal")
    if stats:
        add_mae_r2_box(xs, ys, ax)

    ax.set(xlabel=xlabel, ylabel=ylabel)

    return ax


def scatter_with_err_bar(
    xs: Array,
    ys: Array,
    xerr: Array = None,
    yerr: Array = None,
    ax: Axes = None,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    title: str = None,
    **kwargs,
) -> Axes:
    """Scatter plot with optional x- and/or y-error bars. Useful when passing model
    uncertainties as yerr=y_std for checking if uncertainty correlates with error,
    i.e. if points farther from the parity line have larger uncertainty.

    Args:
        xs (Array): x-values
        ys (Array): y-values
        xerr (Array, optional): Horizontal error bars. Defaults to None.
        yerr (Array, optional): Vertical error bars. Defaults to None.
        ax (Axes, optional): plt axes. Defaults to None.
        xlabel (str, optional): x-axis label. Defaults to "Actual".
        ylabel (str, optional): y-axis label. Defaults to "Predicted".
        title (str, optional): Plot tile. Defaults to None.

    Returns:
        Axes: plt axes with plotted data.
    """
    if ax is None:
        ax = plt.gca()

    styles = dict(markersize=6, fmt="o", ecolor="g", capthick=2, elinewidth=2)
    ax.errorbar(xs, ys, yerr=yerr, xerr=xerr, **kwargs, **styles)
    add_identity(ax)
    add_mae_r2_box(xs, ys, ax)

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    return ax


def density_hexbin(
    targets: Array,
    preds: Array,
    ax: Axes = None,
    color_map: Array = None,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
):
    """Hexagonal-grid scatter plot colored by density or by third dimension
    passed color_by"""
    if ax is None:
        ax = plt.gca()

    # the scatter plot
    hexbin = ax.hexbin(targets, preds, gridsize=75, mincnt=1, bins="log", C=color_map)
    cb_ax = inset_axes(ax, width="3%", height="70%", loc="lower right")
    plt.colorbar(hexbin, cax=cb_ax)
    cb_ax.yaxis.set_ticks_position("left")

    add_identity(ax, label="ideal")
    add_mae_r2_box(targets, preds, ax, loc="upper left")

    ax.set(xlabel=xlabel, ylabel=ylabel)


def density_scatter_with_hist(xs, ys, cell=None, bins=100, **kwargs):
    """Scatter plot colored (and optionally sorted) by density
    with histograms along each dimension
    """

    ax_scatter = with_hist(xs, ys, cell, bins)
    density_scatter(xs, ys, ax_scatter, **kwargs)


def density_hexbin_with_hist(xs, ys, cell=None, bins=100, **kwargs):
    """Hexagonal-grid scatter plot colored by density or by third dimension
    passed color_by with histograms along each dimension.
    """

    ax_scatter = with_hist(xs, ys, cell, bins)
    density_hexbin(xs, ys, ax_scatter, **kwargs)


def residual_vs_actual(y_true: Array, y_pred: Array, ax: Axes = None) -> Axes:
    """Plot ground truth targets on the x-axis against residuals
    (y_err = y_true - y_pred) on the y-axis.

    Args:
        y_true (Array): [description]
        y_pred (Array): [description]
        ax (Axes, optional): [description]. Defaults to None.

    Returns:
        Axes: [description]
    """

    if ax is None:
        ax = plt.gca()

    y_err = y_true - y_pred

    xmin = np.min(y_true) * 0.9
    xmax = np.max(y_true) / 0.9

    plt.plot(y_true, y_err, "o", alpha=0.5, label=None, mew=1.2, ms=5.2)
    plt.plot([xmin, xmax], [0, 0], "k--", alpha=0.5, label="ideal")

    plt.ylabel(r"Residual ($y_\mathrm{test} - y_\mathrm{pred}$)")
    plt.xlabel("Actual value")
    plt.legend(loc="lower right")

    return ax
