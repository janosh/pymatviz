import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy import ndarray as Array
from scipy.interpolate import interpn

from mlmatrics.utils import add_identity, with_hist


def hist_density(xs: Array, ys: Array, sort: bool = True, bins: int = 100) -> None:
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


def density_scatter(
    xs: Array,
    ys: Array,
    ax: Axes = None,
    color_map: Array = None,
    sort: bool = True,
    log: bool = True,
    bins: int = 100,
    xlabel: str = "actual",
    ylabel: str = "predicted",
    **kwargs,
):
    """Scatter plot colored (and optionally sorted) by density"""
    if ax is None:
        ax = plt.gca()

    xs, ys, cs = hist_density(xs, ys, sort=sort, bins=bins)

    norm = mpl.colors.LogNorm() if log else None

    ax.scatter(xs, ys, c=cs, cmap=color_map or "Blues", norm=norm, **kwargs)
    add_identity(ax, label="ideal")

    ax.legend(loc="upper left", frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def density_hexbin(
    targets: Array,
    preds: Array,
    ax: Axes = None,
    title: str = None,
    text: str = None,
    color_map: Array = None,
    xlabel: str = "actual",
    ylabel: str = "predicted",
):
    """Hexagonal-grid scatter plot colored by density or by third dimension
    passed color_by"""
    if ax is None:
        ax = plt.gca()

    # the scatter plot
    hexbin = ax.hexbin(targets, preds, gridsize=75, mincnt=1, bins="log", C=color_map)
    cb_ax = inset_axes(ax, width="3%", height="70%", loc=4)
    plt.colorbar(hexbin, cax=cb_ax)
    cb_ax.yaxis.set_ticks_position("left")

    add_identity(ax, label="ideal")

    ax.legend(title=title, frameon=False, loc="upper left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if text:
        ax.annotate(text, xy=(0.04, 0.7), xycoords="axes fraction")


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
