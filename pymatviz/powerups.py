"""Powerups/enhancements such as parity lines, annotations and marginals for matplotlib
and plotly figures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score

from pymatviz.utils import (
    MPL_BACKEND,
    PLOTLY_BACKEND,
    VALID_BACKENDS,
    VALID_FIG_NAMES,
    VALID_FIG_TYPES,
    AxOrFig,
    Backend,
    annotate,
    get_fig_xy_range,
    pretty_label,
    validate_fig,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.gridspec import GridSpec
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.text import Annotation
    from numpy.typing import ArrayLike
    from plotly.basedatatypes import BaseTraceType


def with_marginal_hist(
    xs: ArrayLike,
    ys: ArrayLike,
    cell: GridSpec | None = None,
    bins: int = 100,
    fig: plt.Figure | plt.Axes | None = None,
) -> plt.Axes:
    """Call before creating a plot and use the returned `ax_main` for all
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


@validate_fig
def annotate_metrics(
    xs: ArrayLike,
    ys: ArrayLike,
    fig: AxOrFig | None = None,
    metrics: dict[str, float] | Sequence[str] = ("MAE", "R2"),
    prefix: str = "",
    suffix: str = "",
    fmt: str = ".3",
    **kwargs: Any,
) -> AnchoredText:
    """Provide a set of x and y values of equal length and an optional Axes
    object on which to print the values' mean absolute error and R^2
    coefficient of determination.

    Args:
        xs (array): x values.
        ys (array): y values.
        fig (plt.Axes | plt.Figure | go.Figure | None, optional): matplotlib Axes or
            Figure or plotly Figure on which to add the annotation. Defaults to None.
        metrics (dict[str, float] | Sequence[str], optional): Metrics to show. Can be a
            subset of recognized keys MAE, R2, R2_adj, RMSE, MSE, MAPE or the names of
            sklearn.metrics.regression functions or any dict of metric names and values.
            Defaults to ("MAE", "R2").
        prefix (str, optional): Title or other string to prepend to metrics.
            Defaults to "".
        suffix (str, optional): Text to append after metrics. Defaults to "".
        fmt (str, optional): f-string float format for metrics. Defaults to '.3'.
        **kwargs: Additional arguments to pass to annotate().

    Returns:
        plt.Axes | plt.Figure | go.Figure: The annotated figure.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    if not isinstance(metrics, (dict, list, tuple, set)):
        raise TypeError(
            f"metrics must be dict|list|tuple|set, not {type(metrics).__name__}"
        )

    backend: Backend = PLOTLY_BACKEND if isinstance(fig, go.Figure) else MPL_BACKEND

    funcs = {
        "MAE": lambda x, y: np.abs(x - y).mean(),
        "RMSE": lambda x, y: (((x - y) ** 2).mean()) ** 0.5,
        "MSE": lambda x, y: ((x - y) ** 2).mean(),
        "MAPE": mape,
        "R2": r2_score,
        "R2_adj": lambda x, y: 1 - (1 - r2_score(x, y)) * (len(x) - 1) / (len(x) - 2),
    }
    for key in set(metrics) - set(funcs):
        if func := getattr(sklearn.metrics, key, None):
            funcs[key] = func
    if bad_keys := set(metrics) - set(funcs):
        raise ValueError(f"Unrecognized metrics: {bad_keys}")

    nan_mask = np.isnan(xs) | np.isnan(ys)
    xs, ys = xs[~nan_mask], ys[~nan_mask]

    text = prefix
    newline = "\n" if backend == MPL_BACKEND else "<br>"

    if isinstance(metrics, dict):
        for key, val in metrics.items():
            label = pretty_label(key, backend)
            text += f"{label} = {val:{fmt}}{newline}"
    else:
        for key in metrics:
            value = funcs[key](xs, ys)
            label = pretty_label(key, backend)
            text += f"{label} = {value:{fmt}}{newline}"
    text += suffix

    return annotate(text, fig, **kwargs)


def add_identity_line(
    fig: go.Figure | plt.Figure | plt.Axes,
    *,
    line_kwds: dict[str, Any] | None = None,
    trace_idx: int = 0,
    retain_xy_limits: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """Add a line shape to the background layer of a plotly figure spanning
    from smallest to largest x/y values in the trace specified by trace_idx.

    Args:
        fig (go.Figure | plt.Figure | plt.Axes): plotly/matplotlib figure or axes to
            add the identity line to.
        line_kwds (dict[str, Any], optional): Keyword arguments for customizing the line
            shape will be passed to fig.add_shape(line=line_kwds). Defaults to
            dict(color="gray", width=1, dash="dash").
        trace_idx (int, optional): Index of the trace to use for measuring x/y limits.
            Defaults to 0. Unused if kaleido package is installed and the figure's
            actual x/y-range can be obtained from fig.full_figure_for_development().
            Applies only to plotly figures.
        retain_xy_limits (bool, optional): If True, the x/y-axis limits will be retained
            after adding the identity line. Defaults to False.
        **kwargs: Additional arguments are passed to fig.add_shape().

    Raises:
        TypeError: If fig is neither a plotly nor a matplotlib figure or axes.
        ValueError: If fig is a plotly figure and kaleido is not installed and trace_idx
            is out of range.

    Returns:
        Figure: Figure with added identity line.
    """
    (x_min, x_max), (y_min, y_max) = get_fig_xy_range(fig=fig, trace_idx=trace_idx)

    if isinstance(fig, (plt.Figure, plt.Axes)):  # handle matplotlib
        ax = fig if isinstance(fig, plt.Axes) else fig.gca()

        line_defaults = dict(alpha=0.5, zorder=0, linestyle="dashed", color="black")
        ax.axline((x_min, x_min), (x_max, x_max), **line_defaults | (line_kwds or {}))
        return fig

    if isinstance(fig, go.Figure):
        xy_min_min = min(x_min, y_min)
        xy_max_min = min(x_max, y_max)

        if fig._grid_ref is not None:  # noqa: SLF001
            kwargs.setdefault("row", "all")
            kwargs.setdefault("col", "all")

        line_defaults = dict(color="gray", width=1, dash="dash")
        fig.add_shape(
            type="line",
            **dict(x0=xy_min_min, y0=xy_min_min, x1=xy_max_min, y1=xy_max_min),
            layer="below",
            line=line_defaults | (line_kwds or {}),
            **kwargs,
        )
        if retain_xy_limits:
            fig.update_xaxes(range=[x_min, x_max])
            fig.update_yaxes(range=[y_min, y_max])

        return fig

    raise TypeError(f"{fig=} must be instance of {VALID_FIG_NAMES}")


def add_best_fit_line(
    fig: go.Figure | plt.Figure | plt.Axes,
    *,
    xs: ArrayLike = (),
    ys: ArrayLike = (),
    trace_idx: int = 0,
    line_kwds: dict[str, Any] | None = None,
    annotate_params: bool | dict[str, Any] = True,
    **kwargs: Any,
) -> go.Figure:
    """Add line of best fit according to least squares to a plotly or matplotlib figure.

    Args:
        fig (go.Figure | plt.Figure | plt.Axes): plotly/matplotlib figure or axes to
            add the best fit line to.
        xs (array, optional): x-values to use for fitting. Defaults to () which
            means use the x-values of trace at trace_idx in fig.
        ys (array, optional): y-values to use for fitting. Defaults to () which
            means use the y-values of trace at trace_idx in fig.
        trace_idx (int, optional): Index of the trace to use for measuring x/y values
            for fitting if xs and ys are not provided. Defaults to 0.
        line_kwds (dict[str, Any], optional): Keyword arguments for customizing the line
            shape. For plotly, will be passed to fig.add_shape(line=line_kwds).
            For matplotlib, will be passed to ax.plot(). Defaults to None.
        annotate_params (dict[str, Any], optional): Pass dict to customize
            the annotation of the best fit line. Set to False to disable annotation.
            Defaults to True.
        **kwargs: Additional arguments are passed to fig.add_shape() for plotly or
            ax.plot() for matplotlib.

    Raises:
        TypeError: If fig is neither a plotly nor a matplotlib figure or axes.
        ValueError: If fig is a plotly figure and xs and ys are not provided and
            trace_idx is out of range.

    Returns:
        Figure: Figure with added best fit line.
    """
    if not isinstance(fig, VALID_FIG_TYPES):
        raise TypeError(f"{fig=} must be instance of {VALID_FIG_NAMES}")

    backend = PLOTLY_BACKEND if isinstance(fig, go.Figure) else MPL_BACKEND
    # default to navy color but let annotate_params override
    line_color = kwargs.setdefault(
        "color",
        annotate_params.get("color", "navy")
        if isinstance(annotate_params, dict)
        else "navy",
    )

    if 0 in {len(xs), len(ys)}:
        if isinstance(fig, go.Figure):
            if not len(xs) or not len(ys):
                trace = fig.data[trace_idx]
                xs, ys = trace.x, trace.y
        else:
            ax = fig if isinstance(fig, plt.Axes) else fig.gca()
            if not len(xs) or not len(ys):
                # get scatter data
                artist = ax.get_children()[trace_idx]
                if isinstance(artist, plt.Line2D):
                    xs, ys = artist.get_xdata(), artist.get_ydata()
                else:
                    xs, ys = artist.get_offsets().T
    slope, intercept = np.polyfit(xs, ys, 1)

    (x_min, x_max), _ = get_fig_xy_range(fig, trace_idx=trace_idx)

    x0, x1 = x_min, x_max
    y0, y1 = slope * x0 + intercept, slope * x1 + intercept

    if annotate_params:
        if backend == MPL_BACKEND:
            defaults = dict(loc="lower right", color=line_color)
        else:
            defaults = dict(
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.02,
                showarrow=False,
                font_color=line_color,
            )
        if isinstance(annotate_params, dict):
            defaults |= annotate_params
        annotate(f"LS fit: y = {slope:.2}x + {intercept:.2}", fig=fig, **defaults)

    if backend == MPL_BACKEND:
        ax = fig if isinstance(fig, plt.Axes) else fig.gca()

        defaults = dict(alpha=0.7, linestyle="--", zorder=1)
        ax.axline((x0, y0), (x1, y1), **(defaults | (line_kwds or {})) | kwargs)

        return fig
    if backend == PLOTLY_BACKEND:
        if fig._grid_ref is not None:  # noqa: SLF001
            for key in ("row", "col"):
                kwargs.setdefault(key, "all")

        line_kwds = dict(
            color=kwargs.pop("color"), width=2, dash="dash", **(line_kwds or {})
        )
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=line_kwds, **kwargs)

        return fig

    raise ValueError(f"Unsupported {backend=}. Must be one of {VALID_BACKENDS}")


def add_ecdf_line(
    fig: go.Figure,
    values: ArrayLike = (),
    trace_idx: int = 0,
    trace_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Add an empirical cumulative distribution function (ECDF) line to a plotly figure.

    Support for matplotlib planned but not implemented. PRs welcome.

    Args:
        fig (go.Figure): plotly figure to add the ECDF line to.
        values (array, optional): Values to compute the ECDF from. Defaults to () which
            means use the x-values of trace at trace_idx in fig.
        trace_idx (int, optional): Index of the trace whose x-values to use for
            computing the ECDF. Defaults to 0. Unused if values is not empty.
        trace_kwargs (dict[str, Any], optional): Passed to trace_ecdf.update().
            Defaults to None. Use e.g. to set trace name (default "Cumulative") or
            line_color (default "gray").
        **kwargs: Passed to fig.add_trace().

    Returns:
        Figure: Figure with added ECDF line.
    """
    valid_fig_types = (go.Figure,)
    if not isinstance(fig, valid_fig_types):
        type_names = " | ".join(
            f"{t.__module__}.{t.__qualname__}" for t in valid_fig_types
        )
        raise TypeError(f"{fig=} must be instance of {type_names}")

    if values == ():
        target_trace: BaseTraceType = fig.data[trace_idx]
        if isinstance(target_trace, (go.Histogram, go.Scatter)):
            values = target_trace.x
        elif isinstance(target_trace, go.Bar):
            xs, ys = target_trace.x, target_trace.y
            values = np.repeat(xs[:-1], ys)

        else:
            cls = type(target_trace)
            qual_name = cls.__module__ + "." + cls.__qualname__
            raise ValueError(
                f"Cannot auto-determine x-values for ECDF from {qual_name}, "
                "pass values explicitly. Currently only Histogram, Scatter, Box, "
                "and Violin traces are supported and may well need more testing. "
                "Please report issues at https://github.com/janosh/pymatviz/issues."
            )

    ecdf_trace = px.ecdf(values).data[0]

    # if fig has facets, add ECDF to all subplots
    add_trace_defaults = {} if fig._grid_ref is None else dict(row="all", col="all")  # noqa: SLF001

    fig.add_trace(ecdf_trace, **add_trace_defaults | kwargs)
    # move ECDF line to secondary y-axis
    # set color to darkened version of primary y-axis color
    trace_defaults = dict(yaxis="y2", name="Cumulative", line=dict(color="gray"))
    trace_kwargs = trace_defaults | (trace_kwargs or {})
    fig.data[-1].update(**trace_kwargs)

    color = trace_kwargs.get("line_color", trace_kwargs.get("line", {}).get("color"))

    yaxis_defaults = dict(
        title=trace_kwargs["name"],
        side="right",
        overlaying="y",
        range=(0, 1),
        showgrid=False,
        color=color,
        linecolor=color,
    )
    # make secondary ECDF y-axis inherit primary y-axis styles
    fig.layout.yaxis2 = yaxis_defaults | getattr(fig.layout, "yaxis2", {})

    return fig
