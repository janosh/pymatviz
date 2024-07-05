"""Powerups for matplotlib figures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np


if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.gridspec import GridSpec
    from matplotlib.text import Annotation
    from numpy.typing import ArrayLike


def with_marginal_hist(
    xs: ArrayLike,
    ys: ArrayLike,
    cell: GridSpec | None = None,
    bins: int = 100,
    fig: plt.Figure | plt.Axes | None = None,
) -> plt.Axes:
    """Call before creating a matplotlib figure and use the returned `ax_main` for all
    subsequent plotting ops to create a grid of plots with the main plot in the
    lower left and narrow histograms along its x- and/or y-axes displayed above
    and near the right edge.

    Args:
        xs (array): Marginal histogram values along x-axis.
        ys (array): Marginal histogram values along y-axis.
        cell (GridSpec, optional): Cell of a plt GridSpec at which to add the
            grid of plots. Defaults to None.
        bins (int, optional): Resolution/bin count of the histograms. Defaults to 100.
        fig (Figure, optional): matplotlib Figure or Axes to add the marginal histograms
            to. Defaults to None.

    Returns:
        plt.Axes: The matplotlib Axes to be used for the main plot.
    """
    if fig is None or isinstance(fig, plt.Axes):
        ax_main = fig or plt.gca()
        fig = ax_main.figure

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


def annotate_bars(
    ax: plt.Axes | None = None,
    *,
    v_offset: float = 10,
    h_offset: float = 0,
    labels: Sequence[str | int | float] | None = None,
    fontsize: int = 14,
    y_max_headroom: float = 1.2,
    adjust_test_pos: bool = False,
    **kwargs: Any,
) -> None:
    """Annotate each bar in bar plot with a label.

    Args:
        ax (Axes): The matplotlib axes to annotate.
        v_offset (int): Vertical offset between the labels and the bars.
        h_offset (int): Horizontal offset between the labels and the bars.
        labels (list[str]): Labels used for annotating bars. If not provided, defaults
            to the y-value of each bar.
        fontsize (int): Annotated text size in pts. Defaults to 14.
        y_max_headroom (float): Will be multiplied with the y-value of the tallest bar
            to increase the y-max of the plot, thereby making room for text above all
            bars. Defaults to 1.2.
        adjust_test_pos (bool): If True, use adjustText to prevent overlapping labels.
            Defaults to False.
        **kwargs: Additional arguments (rotation, arrowprops, etc.) are passed to
            ax.annotate().
    """
    ax = ax or plt.gca()

    if labels is None:
        labels = [int(patch.get_height()) for patch in ax.patches]
    elif len(labels) != len(ax.patches):
        raise ValueError(
            f"Got {len(labels)} labels but {len(ax.patches)} bars to annotate"
        )

    y_max: float = 0
    texts: list[Annotation] = []
    for rect, label in zip(ax.patches, labels):
        y_pos = rect.get_height()
        x_pos = rect.get_x() + rect.get_width() / 2 + h_offset

        if ax.get_yscale() == "log":
            y_pos += np.log(max(1, v_offset))
        else:
            y_pos += v_offset

        y_max = max(y_max, y_pos)

        txt = f"{label:,}" if isinstance(label, (int, float)) else label
        # place label at end of the bar and center horizontally
        anno = ax.annotate(
            txt, (x_pos, y_pos), ha="center", fontsize=fontsize, **kwargs
        )
        texts.append(anno)

    # ensure enough vertical space to display label above highest bar
    ax.set(ylim=(None, y_max * y_max_headroom))
    if adjust_test_pos:
        try:
            from adjustText import adjust_text

            adjust_text(texts, ax=ax)
        except ImportError as exc:
            raise ImportError(
                "adjustText not installed, falling back to default matplotlib "
                "label placement. Use pip install adjustText."
            ) from exc
