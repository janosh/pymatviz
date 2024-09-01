"""Powerups that can be applied to both matplotlib and plotly figures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import sklearn
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score

from pymatviz.utils import (
    BACKENDS,
    MATPLOTLIB,
    PLOTLY,
    VALID_FIG_NAMES,
    VALID_FIG_TYPES,
    AxOrFig,
    Backend,
    annotate,
    get_fig_xy_range,
    get_font_color,
    luminance,
    pretty_label,
    validate_fig,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.offsetbox import AnchoredText
    from numpy.typing import ArrayLike


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
    if not isinstance(metrics, dict | list | tuple | set):
        raise TypeError(
            f"metrics must be dict|list|tuple|set, not {type(metrics).__name__}"
        )

    backend: Backend = PLOTLY if isinstance(fig, go.Figure) else MATPLOTLIB

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
    newline = "\n" if backend == MATPLOTLIB else "<br>"

    def calculate_metrics(xs: ArrayLike, ys: ArrayLike) -> str:
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        if xs.shape != ys.shape:
            raise ValueError(
                f"xs and ys must have the same shape. Got {xs.shape} and {ys.shape}"
            )
        nan_mask = np.isnan(xs) | np.isnan(ys)
        xs, ys = xs[~nan_mask], ys[~nan_mask]
        text = prefix
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
        return text

    if (
        backend == PLOTLY
        and isinstance(fig, go.Figure)
        and any(getattr(trace, "xaxis", None) not in ("x", None) for trace in fig.data)
    ):
        # Handle faceted Plotly figure
        texts = []
        for trace in fig.data:
            trace_xs, trace_ys = trace.x, trace.y
            texts.append(calculate_metrics(trace_xs, trace_ys))
        return annotate(texts, fig, **kwargs)

    # Handle non-faceted figures or matplotlib
    text = calculate_metrics(xs, ys)
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

    if isinstance(fig, plt.Figure | plt.Axes):  # handle matplotlib
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

        line_defaults = dict(color="gray", width=1, dash="dash") | kwargs.pop(
            "line", {}
        )
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
    trace_idx: int | None = None,
    line_kwds: dict[str, Any] | None = None,
    annotate_params: bool | dict[str, Any] = True,
    warn: bool = True,
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
        warn (bool, optional): If True, print a warning if trace_idx is unspecified
            and the figure has multiple traces. Defaults to True.
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

    backend = PLOTLY if isinstance(fig, go.Figure) else MATPLOTLIB
    default_color = "navy" if luminance(get_font_color(fig)) < 0.7 else "lightskyblue"
    # let annotate_params override line_color (also used for annotation)
    line_color = kwargs.setdefault(
        "color",
        annotate_params.get("color", default_color)
        if isinstance(annotate_params, dict)
        else default_color,
    )

    is_faceted = backend == PLOTLY and any(
        getattr(trace, "xaxis", None) not in ("x", None) for trace in fig.data
    )
    if trace_idx is None:
        n_traces = (
            len(fig.data) if isinstance(fig, go.Figure) else len(fig.get_children())
        )
        if n_traces > 1 and warn and not is_faceted:
            # for a faceted plot, multiple traces are expected so doesn't make sense to
            # warn even if user may have wanted to specify trace_idx. we can't know
            print(  # noqa: T201
                f"add_best_fit_line Warning: {trace_idx=} but figure has {n_traces}"
                " traces, defaulting to trace_idx=0. Check fig.data[0] to make sure"
                " this is the expected trace."
            )
        trace_idx = 0

    use_custom_data = len(xs) > 0 and len(ys) > 0

    if backend == MATPLOTLIB:
        ax = fig if isinstance(fig, plt.Axes) else fig.gca()

        if not use_custom_data:
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
            defaults = dict(loc="lower right", color=line_color)
            if isinstance(annotate_params, dict):
                defaults |= annotate_params
            sign = "+" if intercept >= 0 else "-"
            annotate(
                f"LS fit: y = {slope:.2g}x {sign} {abs(intercept):.2g}",
                fig=fig,
                **defaults,
            )

        defaults = dict(alpha=0.7, linestyle="--", zorder=1)
        ax.axline((x0, y0), (x1, y1), **(defaults | (line_kwds or {})) | kwargs)

        return fig

    if backend == PLOTLY:
        # if is_faceted, add line to all subplots
        traces = fig.data if is_faceted else [fig.data[trace_idx]]
        annotation_texts = []

        for trace in traces:
            subplot = trace.xaxis[1:] if trace.xaxis else ""  # 'x2' -> '2', 'x' -> ''
            xref = f"x{subplot}" if subplot else "x"
            yref = f"y{subplot}" if subplot else "y"

            _xs, _ys = (xs, ys) if use_custom_data else (trace.x, trace.y)

            slope, intercept = np.polyfit(_xs, _ys, 1)

            x_min, x_max = min(_xs), max(_xs)
            x0, x1 = x_min, x_max
            y0, y1 = slope * x0 + intercept, slope * x1 + intercept

            line_kwds = (
                (line_kwds or {})
                | dict(color=line_color, width=2, dash="dash")
                | kwargs.pop("line", {})
            )
            invalid_kwargs = ("color",)  # for fig.add_shape

            fig.add_shape(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                xref=xref,
                yref=yref,
                line=line_kwds,
                **{k: v for k, v in kwargs.items() if k not in invalid_kwargs},
            )

            if annotate_params:
                sign = "+" if intercept >= 0 else "-"
                annotation_texts.append(
                    f"LS fit: y = {slope:.2g}x {sign} {abs(intercept):.2g}"
                )

        if annotate_params:
            defaults = dict(
                x=0.98,
                y=0.02,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(color=line_color),
            )
            if isinstance(annotate_params, dict):
                defaults |= annotate_params

            # Use a single string for non-faceted plots, list for faceted plots
            annotation_text = (
                annotation_texts[0] if len(annotation_texts) == 1 else annotation_texts
            )
            annotate(annotation_text, fig=fig, **defaults)

        return fig

    raise ValueError(f"Unsupported {backend=}. Must be one of {BACKENDS}")
