from os.path import abspath, dirname
from typing import Any, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from numpy.typing import NDArray


ROOT: str = dirname(dirname(abspath(__file__)))


NumArray = NDArray[Union[np.float64, np.int_]]


def add_identity(ax: Axes = None, **line_kwargs: Any) -> None:
    """Add a parity line (y = x) to the provided axis."""
    if ax is None:
        ax = plt.gca()

    # zorder=0 ensures other plotted data displays on top of line
    default_kwargs = dict(alpha=0.5, zorder=0, linestyle="dashed", color="black")
    (identity,) = ax.plot([], [], **default_kwargs, **line_kwargs)

    def callback(axes: Axes) -> None:
        x_min, x_max = axes.get_xlim()
        y_min, y_max = axes.get_ylim()
        low = max(x_min, y_min)
        high = min(x_max, y_max)
        identity.set_data([low, high], [low, high])

    callback(ax)
    # Register callbacks to update identity line when moving plots in interactive
    # mode to ensure line always extend to plot edges.
    ax.callbacks.connect("xlim_changed", callback)
    ax.callbacks.connect("ylim_changed", callback)


def with_hist(
    xs: NumArray, ys: NumArray, cell: GridSpec = None, bins: int = 100  # type: ignore
) -> Axes:
    """Call before creating a plot and use the returned `ax_main` for all
    subsequent plotting ops to create a grid of plots with the main plot in
    the lower left and narrow histograms along its x- and/or y-axes displayed
    above and near the right edge.

    Args:
        xs (NumArray): x values.
        ys (NumArray): y values.
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


def softmax(arr: NumArray, axis: int = -1) -> NumArray:  # type: ignore
    """Compute the softmax of an array along an axis."""
    exp = np.exp(arr)
    return exp / exp.sum(axis=axis, keepdims=True)


def one_hot(targets: Sequence[int], n_classes: int = None) -> NDArray[np.int_]:
    """Get a one-hot encoded version of `targets` containing `n_classes`."""
    if n_classes is None:
        n_classes = np.max(targets) + 1
    return np.eye(n_classes)[targets]


def annotate_bar_heights(
    ax: Axes = None,
    voffset: int = 10,
    hoffset: int = 0,
    labels: Sequence[Union[str, int, float]] = None,
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
    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = [int(patch.get_height()) for patch in ax.patches]

    for rect, label in zip(ax.patches, labels):

        y_pos = rect.get_height()
        x_pos = rect.get_x() + rect.get_width() / 2 + hoffset

        if ax.get_yscale() == "log":
            y_pos = y_pos + np.log(voffset)
        else:
            y_pos = y_pos + voffset

        # place label at end of the bar and center horizontally
        ax.annotate(label, (x_pos, y_pos), ha="center", fontsize=fontsize)
        # ensure enough vertical space to display label above highest bar
        ax.margins(y=0.1)
