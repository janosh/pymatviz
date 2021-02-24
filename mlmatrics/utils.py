from os.path import abspath, dirname
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from numpy import ndarray as Array

ROOT: str = dirname(dirname(abspath(__file__)))


def add_identity(ax: Axes = None, **line_kwargs) -> None:
    """Add a parity line (y = x) to the provided axis."""
    if ax is None:
        ax = plt.gca()

    # zorder=0 ensures other plotted data displays on top of line
    default_kwargs = dict(alpha=0.5, zorder=0, linestyle="dashed", color="black")
    (identity,) = ax.plot([], [], **default_kwargs, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(ax)
    # Update identity line when moving the plot in interactive
    # viewing mode to always extend to the plot's edges.
    ax.callbacks.connect("xlim_changed", callback)
    ax.callbacks.connect("ylim_changed", callback)


def with_hist(xs: Array, ys: Array, cell: GridSpec = None, bins: int = 100) -> Axes:
    """Call before creating a plot and use the returned `ax_main` for all
    subsequent plotting ops to create a grid of plots with the main plot in
    the lower left and narrow histograms along its x- and/or y-axes displayed
    above and near the right edge.

    Args:
        xs (Array): x values.
        ys (Array): y values.
        cell (GridSpec, optional): Cell of a plt GridSpec at which to add the
            grid of plots. Defaults to None.
        bins (int, optional): Resolution/bin count of the histograms. Defaults to 100.

    Returns:
        Axes: The axes to be used to for the main plot.
    """
    fig = plt.gcf()

    gs = (cell.subgridspec if cell else fig.add_gridspec)(
        2, 2, width_ratios=(6, 1), height_ratios=(1, 5), wspace=0, hspace=0
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # x_hist
    ax_histx.hist(xs, bins=bins, rwidth=0.8)
    ax_histx.axis("off")

    # y_hist
    ax_histy.hist(ys, bins=bins, rwidth=0.8, orientation="horizontal")
    ax_histy.axis("off")

    return ax_main


def softmax(arr: Array, axis: int = -1) -> Array:
    """ Compute the softmax of an array along an axis. """
    exp = np.exp(arr)
    return exp / exp.sum(axis=axis, keepdims=True)


def one_hot(targets: Array, n_classes: int = None) -> Array:
    """ Get a one-hot encoded version of `targets` containing `n_classes`. """
    if n_classes is None:
        n_classes = np.max(targets) + 1
    return np.eye(n_classes)[targets]


def show_bar_values(
    ax: Axes = None,
    voffset: int = 10,
    hoffset: int = 0,
    labels: List[str] = None,
    fontsize: int = 14,
) -> None:
    """Annotate histograms with a label indicating the height/count of each bar.

    Args:
        ax (matplotlib.axes.Axes): The axes to annotate.
        voffset (int): Vertical offset between the labels and the bars.
        hoffset (int): Horizontal offset between the labels and the bars.
        labels (list[str]): Labels used for annotating bars. Falls back to the
            y-value of each bar if None.
    """
    if labels is None:
        labels = [patch.get_height() for patch in ax.patches]

    for rect, label in zip(ax.patches, labels):

        y_val = rect.get_height()
        x_val = rect.get_x() + rect.get_width() / 2

        # place label at end of the bar and center horizontally
        ax.annotate(
            label, (x_val + hoffset, y_val + voffset), ha="center", fontsize=fontsize
        )
        # ensure enough vertical space to display label above highest bar
        ax.margins(y=0.1)
