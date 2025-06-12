"""Histograms."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from pymatviz.bar import spacegroup_bar
from pymatviz.enums import ElemCountMode
from pymatviz.process_data import count_elements
from pymatviz.typing import BACKENDS, MATPLOTLIB, PLOTLY, Backend


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from pymatviz.typing import ElemValues


def spacegroup_hist(*args: Any, **kwargs: Any) -> plt.Axes | go.Figure:
    """Alias for spacegroup_bar."""
    warnings.warn(
        "spacegroup_hist() is deprecated and will be removed in a future version. "
        "use pymatviz.bar.spacegroup_bar() instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return spacegroup_bar(*args, **kwargs)


def elements_hist(
    formulas: ElemValues,
    *,
    count_mode: ElemCountMode = ElemCountMode.composition,
    log_y: bool = False,
    keep_top: int | None = None,
    show_values: Literal["percent", "count"] | None = "percent",
    bar_width: float = 0.7,
    opacity: float = 0.8,
    fig_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Plot a histogram of elements (e.g. to show occurrence in a dataset) using Plotly.

    Args:
        formulas (list[str]): compositional strings, e.g. ["Fe2O3", "Bi2Te3"].
        count_mode ("composition" | "fractional_composition" | "reduced_composition"):
            Reduce or normalize compositions before counting. See `count_elements` for
            details. Only used when formulas is list of composition strings/objects.
        log_y (bool, optional): Whether y-axis is log or linear. Default = False.
        keep_top (int | None): Display only the top n elements by prevalence.
        show_values ("percent"|"count"|None): "percent" (default) shows percentage
            labels on bars. "count" shows count values. None removes labels.
        bar_width (float): Width of bars as fraction of available space. Default = 0.7.
        opacity (float): Bar opacity between 0 and 1. Default = 0.8.
        fig_kwargs (dict | None): Additional arguments passed to go.Figure().
        **kwargs: Additional keyword arguments passed to go.Bar().

    Returns:
        go.Figure: Plotly figure object
    """
    elem_counts = count_elements(formulas, count_mode)
    non_zero = elem_counts[elem_counts > 0].sort_values(ascending=False)

    if keep_top is not None:
        non_zero = non_zero.head(keep_top)

    # Prepare text labels for bars
    text_labels = None
    if show_values is not None:
        if show_values == "percent":
            sum_elements = non_zero.sum()
            text_labels = [f"{el / sum_elements:.0%}" for el in non_zero.values]
        else:
            text_labels = [str(int(val)) for val in non_zero.values]

    fig = go.Figure(**fig_kwargs or {})
    fig.add_bar(
        x=non_zero.index,
        y=non_zero.values,
        text=text_labels,
        textposition="outside",
        opacity=opacity,
        marker_line_width=1,
        marker_line_color="black",
        width=bar_width,
        **kwargs,
    )

    # Set y-axis scale and labels
    y_title = "log(Element Count)" if log_y else "Element Count"
    fig.update_yaxes(type="log" if log_y else "linear", title=y_title)
    fig.update_xaxes(title="Element")
    fig.layout.showlegend = False

    return fig


def histogram(
    values: Sequence[float] | dict[str, Sequence[float]],
    *,
    bins: int | Sequence[float] | str = 200,
    x_range: tuple[float | None, float | None] | None = None,
    density: bool = False,
    bin_width: float = 1.2,
    log_y: bool = False,
    backend: Backend = PLOTLY,
    fig_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> plt.Figure | go.Figure:
    """Get a histogram with plotly (default) or matplotlib backend but using fast numpy
    pre-processing before handing the data off to the plot function.

    Very common use case when dealing with large datasets so worth having a dedicated
    function for it. Two advantages over the matplotlib/plotly native histograms are
    much faster and much smaller file sizes (when saving plotly figs as HTML since
    plotly saves a complete copy of the data to disk from which it recomputes the
    histogram on the fly to render the figure).
    Speedup example:

        gaussian = np.random.normal(0, 1, 1_000_000_000)
        plot_histogram(gaussian)  # takes 17s
        px.histogram(gaussian)  # ran for 3m45s before crashing the Jupyter kernel

    Args:
        values (Sequence[float] or dict[str, Sequence[float]]): The values to plot as a
            histogram. If a dict is provided, the keys are used as legend labels.
        bins (int or sequence, optional): The number of bins or the bin edges to use for
            the histogram. If not provided, a default value will be used.
        x_range (tuple, optional): The range of values to include in the histogram. If
            not provided, the whole range of values will be used. Defaults to None.
        density (bool, optional): Whether to normalize the histogram. Defaults to False.
        bin_width (float, optional): The width of the histogram bins as a fraction of
            distance between bin edges. Defaults to 1.2 (20% overlap).
        log_y (bool, optional): Whether to log scale the y-axis. Defaults to False.
        backend (str, optional): The plotting backend to use. Can be either 'matplotlib'
            or 'plotly'. Defaults to 'plotly'.
        fig_kwargs (dict, optional): Additional keyword arguments to pass to the figure
            creation function (plt.figure for Matplotlib or go.Figure for Plotly).
        **kwargs: Additional keyword arguments to pass to the plotting function
            (plt.bar for Matplotlib or go.Figure.add_bar for Plotly).

    Returns:
        plt.Figure | go.Figure: The figure object containing the histogram.
    """
    fig_kwargs = fig_kwargs or {}

    # if values was a Series, extract the name attribute to use as legend label
    x_axis_title = getattr(values, "name", "Value")
    data = values if isinstance(values, dict) else {x_axis_title: values}

    # Calculate the maximum data range across all datasets
    all_values = np.concatenate(list(data.values()))
    global_min = np.min(all_values)
    global_max = np.max(all_values)

    if x_range is None:
        x_range = (global_min, global_max)
    else:
        x_range = (x_range[0] or global_min, x_range[1] or global_max)

    # Calculate bin edges
    if isinstance(bins, int):
        bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    elif isinstance(bins, str):
        bin_edges = np.histogram_bin_edges(all_values, bins=bins, range=x_range)
    else:
        bin_edges = np.asarray(bins)

    if backend == MATPLOTLIB:
        fig = plt.figure(**fig_kwargs)
        for label, vals in data.items():
            hist_vals, _ = np.histogram(vals, bins=bin_edges, density=density)
            plt.bar(
                bin_edges[:-1],
                hist_vals,
                label=label,
                alpha=0.7,
                width=bin_width * (bin_edges[1] - bin_edges[0]),
                align="edge",
                **kwargs,
            )

        plt.yscale("log" if log_y else "linear")
        plt.ylabel("Density" if density else "Count")
        plt.xlabel(x_axis_title)

        if len(data) > 1:
            plt.legend()

    elif backend == PLOTLY:
        fig = go.Figure(**fig_kwargs)
        for label, vals in data.items():
            hist_vals, _ = np.histogram(vals, bins=bin_edges, density=density)
            fig.add_bar(
                x=bin_edges[:-1],
                y=hist_vals,
                name=label,
                opacity=0.7,
                width=bin_width * (bin_edges[1] - bin_edges[0]),
                marker_line_width=0,
            )

        y_title = "Density" if density else "Count"
        fig.update_yaxes(type="log" if log_y else "linear", title=y_title)
        fig.update_xaxes(title=x_axis_title)

    else:
        raise ValueError(f"Unsupported {backend=}. Must be one of {BACKENDS}")

    return fig
