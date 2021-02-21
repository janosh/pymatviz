import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interpn

from mlmatrics.utils import add_identity, with_hist


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

    data, x_edges, y_edges = np.histogram2d(xs, ys, bins)

    cs = interpn(
        (0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])),
        data,
        np.vstack([xs, ys]).T,
        method="splinef2d",
        bounds_error=False,
    )

    if sort:
        # sort points by density so densest points are plotted last
        idx = cs.argsort()
        xs, ys, cs = xs[idx], ys[idx], cs[idx]

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


def density_scatter_hex(
    targets,
    preds,
    ax=None,
    title=None,
    text=None,
    color_by=None,
    xlabel="actual",
    ylabel="predicted",
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


def density_scatter_hex_with_hist(xs, ys, cell=None, bins=100, **kwargs):
    """Hexagonal-grid scatter plot colored by density or by third dimension
    passed color_by with histograms along each dimension.
    """

    ax_scatter = with_hist(xs, ys, cell, bins)
    density_scatter_hex(xs, ys, ax_scatter, **kwargs)
