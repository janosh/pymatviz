import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interpn

from mlmatrics.utils import add_identity, with_hist


def hist_density(x, y, sort=True, bins=100):
    """
    return an approximate density of points
    """

    data, x_e, y_e = np.histogram2d(x, y, bins=bins)

    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    return x, y, z


def density_scatter(
    xs,
    ys,
    ax=None,
    colors=None,
    sort=True,
    log=True,
    bins=100,
    xlabel="actual",
    ylabel="predicted",
    text=None,
    **kwargs,
):
    """Scatter plot colored (and optionally sorted) by density"""
    if ax is None:
        ax = plt.gca()

    xs, ys, cs = hist_density(xs, ys, sort=sort, bins=bins)

    norm = mpl.colors.LogNorm() if log else None

    ax.scatter(xs, ys, c=cs, cmap=colors or "Blues", **kwargs, norm=norm)
    add_identity(ax, label="ideal")

    ax.legend(loc="upper left", frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if text:
        ax.annotate(
            text,
            xy=(1, 0),
            xytext=(-15, 15),
            ha="right",
            xycoords="axes fraction",
            textcoords="offset points",
        )

    return ax


def density_hexbin(
    targets,
    preds,
    ax=None,
    title=None,
    text=None,
    color_by=None,
    xlabel="Actual",
    ylabel="Predicted",
):
    """Hexagonal-grid scatter plot colored by density or by third dimension
    passed color_by"""
    if ax is None:
        ax = plt.gca()

    # the scatter plot
    hexbin = ax.hexbin(targets, preds, gridsize=75, mincnt=1, bins="log", C=color_by)
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
