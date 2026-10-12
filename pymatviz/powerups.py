"""Powerups for plotly figures."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence, Sized
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score

from pymatviz.utils import annotate, get_fig_xy_range, get_font_color, luminance
from pymatviz.utils.plotting import PRETTY_LABELS


if TYPE_CHECKING:
    from numpy.typing import ArrayLike

TracePredicate = Callable[[go.Scatter], bool]
TraceSelector = int | slice | Sequence[int] | TracePredicate
AnnotationMode = Literal["per_trace", "combined", "none"]


def _is_faceted(fig: go.Figure) -> bool:
    """Whether any trace sits on a subplot other than the primary x-axis."""
    return any(getattr(trace, "xaxis", None) not in (None, "x") for trace in fig.data)


def annotate_metrics(
    xs: ArrayLike | Sequence[ArrayLike],
    ys: ArrayLike | Sequence[ArrayLike],
    fig: go.Figure | None = None,
    metrics: Mapping[str, float | str] | Sequence[str] = ("MAE", "R2"),
    prefix: str = "",
    suffix: str = "",
    fmt: str = ".3",
    **kwargs: Any,
) -> go.Figure:
    """Provide a set of x and y values of equal length and an optional Figure
    object on which to print the values' mean absolute error and R^2
    coefficient of determination.

    Args:
        xs (array): x values. For faceted figures, either one array per subplot or
            empty to use each subplot's trace data.
        ys (array): y values, same shape as xs.
        fig (go.Figure | None, optional): plotly Figure on which to add the annotation.
            Defaults to None.
        metrics (Mapping[str, float | str] | Sequence[str], optional): Metrics to
            show. A sequence of metric names (built-in MAE, RMSE, MSE, MAPE, R2,
            R2_adj or any sklearn.metrics function name) computed from xs and ys and
            labeled by name. A mapping of labels to either precomputed numbers
            (shown as is) or metric names (computed), e.g.
            {"Mean abs. error": "MAE", "Custom": 1.23}. Defaults to ("MAE", "R2").
        prefix (str, optional): Title or other string to prepend to metrics.
            Defaults to "".
        suffix (str, optional): Text to append after metrics. Defaults to "".
        fmt (str, optional): f-string float format for metrics. Defaults to '.3'.
        **kwargs: Additional arguments to pass to annotate().

    Returns:
        go.Figure: The annotated figure.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    if not isinstance(fig, go.Figure):
        raise TypeError(f"{fig=} must be instance of go.Figure")
    if not isinstance(metrics, (Mapping, list, tuple, set)):
        raise TypeError(
            f"metrics must be dict|list|tuple|set, not {type(metrics).__name__}"
        )

    funcs = {
        "MAE": lambda x, y: np.abs(x - y).mean(),
        "RMSE": lambda x, y: (((x - y) ** 2).mean()) ** 0.5,
        "MSE": lambda x, y: ((x - y) ** 2).mean(),
        "MAPE": mape,
        "R2": r2_score,
        # adjusted R2 is undefined for n <= 2 samples (zero/negative denominator)
        "R2_adj": lambda x, y: (
            1 - (1 - r2_score(x, y)) * (len(x) - 1) / (len(x) - 2)
            if len(x) > 2
            else float("nan")
        ),
    }
    # label -> precomputed number or name of the metric to compute
    label_to_metric: dict[str, float | str] = (
        {str(label): val for label, val in metrics.items()}
        if isinstance(metrics, Mapping)
        else {str(name): str(name) for name in metrics}
    )
    if bad_vals := {
        label: val
        for label, val in label_to_metric.items()
        if not isinstance(val, (Real, str))
    }:
        raise TypeError(
            f"metrics values must be numbers or metric names, got {bad_vals}"
        )
    metric_names = {val for val in label_to_metric.values() if isinstance(val, str)}
    for name in metric_names - set(funcs):
        if func := getattr(sklearn.metrics, name, None):
            funcs[name] = func
    if bad_names := metric_names - set(funcs):
        raise ValueError(f"Unrecognized metrics: {bad_names}")

    def calculate_metrics(
        xs: ArrayLike | Sequence[ArrayLike], ys: ArrayLike | Sequence[ArrayLike]
    ) -> str:
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        if xs.shape != ys.shape:
            raise ValueError(
                f"xs and ys must have the same shape. Got {xs.shape} and {ys.shape}"
            )
        nan_mask = np.isnan(xs) | np.isnan(ys)
        xs, ys = xs[~nan_mask], ys[~nan_mask]
        text = prefix
        for label, metric in label_to_metric.items():
            value = funcs[metric](xs, ys) if isinstance(metric, str) else metric
            text += f"{PRETTY_LABELS.get(label, label)} = {float(value):{fmt}}<br>"
        return text + suffix

    if _is_faceted(fig):  # one annotation per subplot
        n_subplots = len(fig.data)
        if (  # one array per subplot
            isinstance(xs, (list, tuple))
            and isinstance(ys, (list, tuple))
            and len(xs) == len(ys) == n_subplots
            and all(np.ndim(sub_xs) == 1 for sub_xs in xs)
        ):
            subplot_data = list(zip(xs, ys, strict=True))
        elif all(isinstance(vals, Sized) and len(vals) == 0 for vals in (xs, ys)):
            # empty xs/ys: use each subplot's trace data
            subplot_data = [(trace.x, trace.y) for trace in fig.data]
        else:
            raise ValueError(
                f"Faceted figures need one xs/ys array per subplot ({n_subplots} "
                f"traces) or empty xs/ys to use trace data, got {type(xs).__name__} xs"
            )
        texts = [calculate_metrics(sub_xs, sub_ys) for sub_xs, sub_ys in subplot_data]
        return annotate(texts, fig, **kwargs)

    return annotate(calculate_metrics(xs, ys), fig, **kwargs)


def add_identity_line(
    fig: go.Figure,
    *,
    line_kwargs: dict[str, Any] | None = None,
    traces: TraceSelector = lambda _: True,
    retain_xy_limits: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """Add a y=x line shape to the background layer of a plotly figure spanning
    the overlap of the x and y ranges of the traces specified by traces.

    Args:
        fig (go.Figure): plotly figure to add the identity line to.
        line_kwargs (dict[str, Any], optional): Keyword arguments for customizing the
            line shape will be passed to fig.add_shape(line=line_kwargs). Defaults to
            dict(color="gray", width=1, dash="dash").
        traces (TraceSelector, optional): Which trace(s) to use for determining
            the x/y range. Can be int, slice, sequence of ints, or a function that takes
            a trace and returns True/False. By default, applies to all traces.
        retain_xy_limits (bool, optional): If True, the x/y-axis limits will be retained
            after adding the identity line. Defaults to False.
        **kwargs: Additional arguments are passed to fig.add_shape().

    Raises:
        TypeError: If fig is not a plotly figure.
        ValueError: If fig is a plotly figure and no valid traces are found.

    Returns:
        go.Figure: Figure with added identity line.
    """
    if not isinstance(fig, go.Figure):
        raise TypeError(f"{fig=} must be instance of go.Figure")

    (x_min, x_max), (y_min, y_max) = get_fig_xy_range(fig=fig, traces=traces)

    # y=x is only visible where the x and y ranges overlap; spanning beyond would
    # stretch the shorter axis
    line_min, line_max = max(x_min, y_min), min(x_max, y_max)
    if line_min >= line_max:
        warnings.warn(
            f"Skipping identity line: x range {(x_min, x_max)} and y range "
            f"{(y_min, y_max)} don't overlap",
            UserWarning,
            stacklevel=2,
        )
        return fig

    if fig._grid_ref is not None:
        kwargs.setdefault("row", "all")
        kwargs.setdefault("col", "all")

    # Prepare line properties
    line_defaults = dict(color="gray", width=1, dash="dash")
    if "line" in kwargs:
        line_defaults.update(kwargs.pop("line"))
    if line_kwargs:
        line_defaults.update(line_kwargs)

    fig.add_shape(
        type="line",
        x0=line_min,
        y0=line_min,
        x1=line_max,
        y1=line_max,
        layer="below",
        line=line_defaults,
        **kwargs,
    )
    if retain_xy_limits:  # plotly expects log axis ranges in log10 units
        for (low, high), axis, update_axes in (
            ((x_min, x_max), fig.layout.xaxis, fig.update_xaxes),
            ((y_min, y_max), fig.layout.yaxis, fig.update_yaxes),
        ):
            is_log = axis.type == "log"
            update_axes(range=np.log10([low, high]) if is_log else [low, high])

    return fig


def _has_xy_data(trace: go.Scatter) -> bool:
    """Check if a trace has non-empty x and y data."""
    return (
        hasattr(trace, "x")
        and hasattr(trace, "y")
        and trace.x is not None
        and trace.y is not None
        and len(trace.x) > 0
        and len(trace.y) > 0
    )


def _get_valid_traces(
    fig: go.Figure,
    traces: TraceSelector,
    validate_trace: Callable[[go.Scatter], bool] = _has_xy_data,
) -> list[int]:
    """Helper to get valid trace indices from a trace selector.

    Args:
        fig (go.Figure): The figure containing traces to filter
        traces (TraceSelector): Trace selector (int, slice, sequence, or callable)
        validate_trace (Callable[[go.Scatter], bool]): Function to validate if a
            trace is usable. Defaults to requiring both x and y data.

    Returns:
        list[int]: valid trace indices
    """
    # Select traces
    selected_traces: list[int] = []
    if isinstance(traces, int):
        if 0 <= traces < len(fig.data):
            selected_traces = [traces]
        else:
            raise ValueError(
                f"Trace index {traces} out of bounds (0-{len(fig.data) - 1})"
            )
    elif isinstance(traces, slice):
        selected_traces = list(range(*traces.indices(len(fig.data))))
    elif isinstance(traces, (list, tuple)):
        # At runtime, we validate indices are integers and in bounds
        selected_traces = [
            idx for idx in traces if isinstance(idx, int) and 0 <= idx < len(fig.data)
        ]
        if not selected_traces:
            valid_range = f"0-{len(fig.data) - 1}"
            raise ValueError(f"No valid trace indices in {traces}, {valid_range=}")
    elif callable(traces):
        trace_predicate = cast("TracePredicate", traces)
        selected_traces = [
            idx for idx, trace in enumerate(fig.data) if trace_predicate(trace)
        ]
        if not selected_traces:
            raise ValueError("No traces matched the filtering function")
    else:
        actual = type(traces).__name__
        raise TypeError(
            f"traces must be int, slice, sequence, or callable, got {actual}"
        )

    # Filter for traces with valid data according to validator
    valid_traces = [idx for idx in selected_traces if validate_trace(fig.data[idx])]

    if not valid_traces:
        raise ValueError("No valid traces with required data found")

    return valid_traces


def _get_trace_color(trace: go.Scatter, default_color: str = "navy") -> str:
    """Extract color from trace, first checking marker then line."""
    marker_color = getattr(getattr(trace, "marker", None), "color", None)
    if isinstance(marker_color, str):  # skip per-point color arrays
        return marker_color
    line_color = getattr(getattr(trace, "line", None), "color", None)
    return line_color if isinstance(line_color, str) else default_color


def add_best_fit_line(
    fig: go.Figure,
    *,
    xs: ArrayLike = (),
    ys: ArrayLike = (),
    traces: TraceSelector = lambda _: True,
    line_kwargs: dict[str, Any] | None = None,
    annotate_params: bool | dict[str, Any] = True,
    annotation_mode: AnnotationMode = "per_trace",
    xref: str = "x",
    yref: str = "y",
    **kwargs: Any,
) -> go.Figure:
    """Add line of best fit according to least squares to a plotly figure.

    Args:
        fig (go.Figure): plotly figure to add the best fit line to.
        xs (array, optional): x-values to use for fitting. Defaults to () which
            means use the x-values of traces selected by the traces parameter.
        ys (array, optional): y-values to use for fitting. Defaults to () which
            means use the y-values of traces selected by the traces parameter.
        traces (TraceSelector, optional): Which trace(s) to use. Can be int, slice,
            sequence of ints, or a function that takes a trace and returns True/False.
            By default, applies to all traces. Only used when xs and ys not provided.
        line_kwargs (dict[str, Any], optional): Keyword arguments for customizing the
            line shape. Will be passed to fig.add_shape(line=line_kwargs).
            Defaults to None.
        annotate_params (bool | dict[str, Any], optional): Pass dict to customize
            the annotation of the best fit line. Set to False to disable annotation.
            Defaults to True.
        annotation_mode (AnnotationMode, optional): How to fit trace data. Options:
            - "per_trace": Each selected trace gets its own fit line and annotation
            - "combined": Selected traces in the same subplot share one fit line and
              annotation, replacing previous fit annotations
            - "none": Fit lines per trace without annotations
            Faceted figures get separate fits per subplot in all modes.
            Defaults to "per_trace".
        xref (str, optional): x-axis of the subplot to draw directly passed xs/ys
            on, e.g. "x2". Defaults to "x".
        yref (str, optional): y-axis matching xref, e.g. "y2". Defaults to "y".
        **kwargs: Additional arguments are passed to fig.add_shape(). color sets
            the fit line color for directly passed xs/ys and traces without a color.

    Raises:
        TypeError: If fig is not a plotly figure.
        ValueError: If fig is a plotly figure and xs and ys are not provided and
            no valid traces are found.

    Returns:
        go.Figure: Figure with added best fit line.
    """
    if not isinstance(fig, go.Figure):
        raise TypeError(f"{fig=} must be instance of go.Figure")

    default_color = "navy" if luminance(get_font_color(fig)) < 0.7 else "lightskyblue"
    line_color = kwargs.pop(
        "color",
        annotate_params.get("color", default_color)
        if isinstance(annotate_params, dict)
        else default_color,
    )

    # Continue annotation offsets per subplot; combined mode replaces old fit labels.
    def is_fit_label(anno: go.layout.Annotation) -> bool:
        """Whether an annotation was added by a previous add_best_fit_line call."""
        return "LS fit: y =" in (anno.text or "")

    if annotation_mode == "combined":
        fig.layout.annotations = [
            anno for anno in fig.layout.annotations if not is_fit_label(anno)
        ]
    labels_per_subplot: dict[str, int] = {}
    for anno in fig.layout.annotations:
        if is_fit_label(anno):
            anno_x = (anno.xref or "x").removesuffix(" domain").replace("paper", "x")
            labels_per_subplot[anno_x] = labels_per_subplot.get(anno_x, 0) + 1

    def add_fit_line(
        data_xs: ArrayLike,
        data_ys: ArrayLike,
        color: str,
        xref: str = "x",
        yref: str = "y",
    ) -> None:
        """Fit the finite data points and add the fit line and optional label."""
        arr_xs, arr_ys = np.asarray(data_xs, dtype=float), np.asarray(data_ys, float)
        finite = np.isfinite(arr_xs) & np.isfinite(arr_ys)
        if (n_finite := int(finite.sum())) < 2:
            warnings.warn(
                f"Skipping best fit line: needs >= 2 finite points, got {n_finite} "
                f"of {len(arr_xs)} on {xref=}",
                UserWarning,
                stacklevel=3,
            )
            return
        arr_xs, arr_ys = arr_xs[finite], arr_ys[finite]
        slope, intercept = np.polyfit(arr_xs, arr_ys, 1)
        x_min, x_max = float(arr_xs.min()), float(arr_xs.max())
        y0, y1 = slope * x_min + intercept, slope * x_max + intercept

        plotly_line_defaults = dict(color=color, width=2, dash="dash")
        if line_kwargs:
            plotly_line_defaults.update(line_kwargs)
        fig.add_shape(
            **dict(
                type="line",
                x0=x_min,
                y0=y0,
                x1=x_max,
                y1=y1,
                xref=xref,
                yref=yref,
                line=plotly_line_defaults,
            )
            | kwargs
        )

        if not annotate_params or annotation_mode == "none":
            return

        sign = "+" if intercept >= 0 else "-"
        text = f"LS fit: y = {slope:.2g}x {sign} {abs(intercept):.2g}"

        n_labels = labels_per_subplot.get(xref, 0)
        labels_per_subplot[xref] = n_labels + 1

        plotly_anno_defaults: dict[str, Any] = dict(
            x=0.98,
            y=0.02 + 0.05 * n_labels,
            xref=f"{xref} domain",
            yref=f"{yref} domain",
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font=dict(color=color),
        )

        if isinstance(annotate_params, dict):
            anno_params = dict(annotate_params)
            anno_color = anno_params.pop("color", None)
            plotly_anno_defaults.update(anno_params)
            if anno_color is not None:
                font = plotly_anno_defaults["font"]
                if not isinstance(font, dict):  # e.g. a plotly Font object
                    font = font.to_plotly_json()
                plotly_anno_defaults["font"] = font | {"color": anno_color}

        fig.add_annotation(text=text, **plotly_anno_defaults)

    # CASE 1: Custom data provided directly
    xs_arr, ys_arr = np.asarray(xs), np.asarray(ys)
    if len(xs_arr) > 0 and len(ys_arr) > 0:
        add_fit_line(xs_arr, ys_arr, line_color, xref, yref)
        return fig

    # CASE 2: Fit trace data, grouped by subplot (a single group if not faceted).
    # Combined mode fits all selected traces of a subplot together, per_trace (and
    # none) fits each trace with >= 2 points separately.
    subplot_traces: dict[tuple[str, str], list[Any]] = {}
    for trace_idx in dict.fromkeys(_get_valid_traces(fig, traces)):
        trace = fig.data[trace_idx]
        refs = (getattr(trace, "xaxis", None) or "x", getattr(trace, "yaxis", None))
        subplot_traces.setdefault((refs[0], refs[1] or "y"), []).append(trace)

    for (sub_xref, sub_yref), group in subplot_traces.items():
        if annotation_mode == "combined":
            all_xs = np.concatenate([trace.x for trace in group])
            all_ys = np.concatenate([trace.y for trace in group])
            color = _get_trace_color(group[0], line_color)
            add_fit_line(all_xs, all_ys, color, sub_xref, sub_yref)
            continue
        for trace in group:
            if len(trace.x) < 2 or len(trace.y) < 2:
                continue
            color = _get_trace_color(trace, line_color)
            add_fit_line(trace.x, trace.y, color, sub_xref, sub_yref)

    return fig


def enhance_parity_plot(
    fig: go.Figure | None = None,
    xs: ArrayLike = (),
    ys: ArrayLike = (),
    *,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] | None = None,
    stats: bool | dict[str, Any] | None = True,
    traces: TraceSelector = lambda _: True,
    annotation_mode: AnnotationMode = "combined",
) -> go.Figure | None:
    """Add parity plot powerups to a plotly figure, including
    identity line (y=x), best-fit line, and pred vs ref statistics (MAE, R², ...).

    Args:
        xs (array): x-values to use for fitting best-fit line and computing stats.
        ys (array): y-values to use for fitting best-fit line and computing stats.
        fig (go.Figure | None): plotly Figure to add powerups to. Defaults to None.
        identity_line (bool | dict[str, Any], optional): Whether to add a parity line
            (y=x). Pass a dict to customize line properties. Defaults to True.
        best_fit_line (bool | dict[str, Any] | None, optional): Whether to add a
            best-fit line. Pass a dict to customize line properties. If None (default),
            will be enabled if R² > 0.3.
        stats (bool | dict[str, Any] | None, optional): Whether to display text box(es)
            with metrics (MAE, R², etc). Pass a dict to customize text box properties,
            or None to disable. Defaults to True.
        traces (TraceSelector, optional): Specifies which trace(s) to use. Can be int,
            slice, sequence of ints, or a function that takes a trace and returns bool.
            By default, applies to all traces. Only used when xs and ys not provided.
        annotation_mode (AnnotationMode, optional): How to display metric annotations.
            Defaults to "combined". Options:
            - "per_trace": Each selected trace gets its own annotation
            - "combined": All selected traces are combined into a single annotation
            - "none": Only add identity line and best-fit line, don't show annotations

    Returns:
        go.Figure: The enhanced figure.

    Raises:
        ValueError: If xs and ys are not provided and fig is not a plotly figure where
            the data can be directly accessed.
    """
    # Convert to numpy arrays early for array operations
    xs_arr, ys_arr = np.asarray(xs), np.asarray(ys)

    # Add identity line if requested
    if identity_line and fig is not None:
        identity_kwargs = identity_line if isinstance(identity_line, dict) else {}
        add_identity_line(fig, traces=traces, **identity_kwargs)

    # Early return if no stats or best-fit line needed and manual data not provided
    if (
        not stats
        and best_fit_line in (False, None)
        and len(xs_arr) == 0
        and len(ys_arr) == 0
    ):
        return fig

    has_data = len(xs_arr) > 0 and len(ys_arr) > 0
    if not has_data:
        if not isinstance(fig, go.Figure):
            raise TypeError(
                "this powerup can only get x/y data from the figure directly "
                "for plotly figures."
            )
        valid_traces = _get_valid_traces(fig, traces)
        if annotation_mode == "combined":
            xs_arr = np.concatenate([fig.data[idx].x for idx in valid_traces])
            ys_arr = np.concatenate([fig.data[idx].y for idx in valid_traces])

    if has_data or annotation_mode == "combined":
        if best_fit_line is None:
            best_fit_line = r2_score(xs_arr, ys_arr) > 0.3
        # trace-derived data on faceted figures is fitted/annotated per subplot
        per_subplot = not has_data and fig is not None and _is_faceted(fig)
        if best_fit_line and fig is not None:
            fit_kwargs = {} if isinstance(best_fit_line, bool) else best_fit_line
            if per_subplot:
                add_best_fit_line(
                    fig, traces=traces, annotation_mode="combined", **fit_kwargs
                )
            else:
                add_best_fit_line(fig, xs=xs_arr, ys=ys_arr, **fit_kwargs)
        if stats and annotation_mode != "none" and fig is not None:
            stats_kwargs = {} if isinstance(stats, bool) else stats
            stat_xs, stat_ys = ((), ()) if per_subplot else (xs_arr, ys_arr)
            annotate_metrics(stat_xs, stat_ys, fig=fig, **stats_kwargs)
        return fig

    if annotation_mode == "per_trace" and fig is not None:
        for trace_idx in dict.fromkeys(valid_traces):
            # Get data for this trace
            trace = fig.data[trace_idx]
            trace_xs = np.array(trace.x)
            trace_ys = np.array(trace.y)

            # Skip if insufficient data
            if len(trace_xs) < 2 or len(trace_ys) < 2:
                continue

            # Get color from trace
            trace_color = _get_trace_color(trace, "navy")

            # Add best fit line if requested
            if best_fit_line is not False:
                line_style: dict[str, Any] = {"color": trace_color}
                if isinstance(best_fit_line, dict):
                    line_style |= best_fit_line
                add_best_fit_line(fig, xs=trace_xs, ys=trace_ys, line_kwargs=line_style)

            # Add stats annotation if requested
            if stats:
                per_trace_stats_kwargs = {} if isinstance(stats, bool) else stats
                annotate_metrics(trace_xs, trace_ys, fig=fig, **per_trace_stats_kwargs)

        return fig

    if annotation_mode != "none":
        raise ValueError(
            f"Unknown {annotation_mode=}. Must be one of {get_args(AnnotationMode)}"
        )

    return fig


def add_ecdf_line(
    fig: go.Figure,
    values: ArrayLike = (),
    traces: TraceSelector = lambda _: True,
    trace_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Add an empirical cumulative distribution function (ECDF) line to a plotly figure.

    Args:
        fig (go.Figure): plotly figure to add the ECDF line to.
        values (array, optional): Values to compute the ECDF from. Defaults to () which
            means use the x-values of traces selected by the traces parameter.
        traces (TraceSelector, optional): Specifies which trace(s) to use. Can be int,
            slice, sequence of ints, or a function that takes a trace and returns bool.
            By default, applies to all traces. Only used when values is not provided.
        trace_kwargs (dict[str, Any], optional): Passed to trace_ecdf.update().
            Defaults to None. Use e.g. to set trace name (default "Cumulative") or
            line_color (default "gray"). Applied to every added ECDF trace.
        **kwargs: Passed to fig.add_trace().

    Returns:
        go.Figure: Figure with added ECDF line(s).
    """
    trace_kwargs = trace_kwargs or {}
    if not isinstance(fig, go.Figure):
        raise TypeError(f"{fig=} must be instance of go.Figure")

    # Check if we have a subplot structure and get base name for traces
    has_subplots = fig._grid_ref is not None
    base_name = trace_kwargs.get("name", "Cumulative")
    xlabel = fig.layout.xaxis.title.text or "x"
    hovertemplate = f"{xlabel}: %{{x}}<br>Percent: %{{y:.2%}}<extra></extra>"
    yaxis_defaults = dict(
        title=base_name, side="right", overlaying="y", range=(0, 1), showgrid=False
    )

    # Convert to numpy array for array operations
    values_arr = np.asarray(values) if values is not None else np.array([])

    # If explicit values are provided, just add one ECDF line
    if len(values_arr) > 0:
        ecdf_trace = px.ecdf(values_arr).data[0]
        fig.add_trace(ecdf_trace, **kwargs)

        fig.data[-1].hovertemplate = hovertemplate

        # Create unique name
        name = base_name
        trace_names = [trace.name for trace in fig.data[:-1]]
        name_suffix = 1
        while name in trace_names:
            name_suffix += 1
            name = f"{base_name} {name_suffix}"

        # Apply user overrides to trace defaults
        line_defaults = dict(
            color=trace_kwargs.get("line_color", "gray"),
            dash=trace_kwargs.get("line", {}).get("dash", "solid"),
        )
        current_trace_kwargs: dict[str, Any] = {
            "yaxis": "y2",
            "name": name,
            "line": line_defaults,
        } | trace_kwargs
        if isinstance(line_kwargs := trace_kwargs.get("line"), dict):
            current_trace_kwargs["line"] = line_defaults | line_kwargs

        fig.data[-1].update(**current_trace_kwargs)

        yaxis_defaults["title"] = name
        line_dict = current_trace_kwargs.get("line", {})
        if isinstance(line_dict, dict) and (color := line_dict.get("color")):
            yaxis_defaults["color"] = color

        # Set up yaxis2 properly
        if not hasattr(fig.layout, "yaxis2"):
            fig.layout.yaxis2 = yaxis_defaults
        else:
            for key, value in yaxis_defaults.items():
                setattr(fig.layout.yaxis2, key, value)

        return fig

    # ECDF validation function - histograms only need x data
    def is_horizontal(trace: Any) -> bool:
        """Bars/histograms with values on the y-axis (histograms with only y data)."""
        if getattr(trace, "orientation", None) == "h":
            return True
        return isinstance(trace, go.Histogram) and trace.x is None

    def validate_ecdf_trace(trace: go.Scatter) -> bool:
        if isinstance(trace, go.Histogram):
            values = trace.y if is_horizontal(trace) else trace.x
            return values is not None and len(values) > 0
        return (
            hasattr(trace, "x")
            and hasattr(trace, "y")
            and trace.x is not None
            and trace.y is not None
            and len(trace.x) > 0
            and len(trace.y) > 0
        )

    # Get valid traces using the helper function with custom validation
    selected_traces = _get_valid_traces(fig, traces, validate_ecdf_trace)
    is_single_trace = len(selected_traces) == 1

    # Set up secondary y-axis if it doesn't exist yet
    if not hasattr(fig.layout, "yaxis2"):
        fig.layout.yaxis2 = yaxis_defaults

    # Process each selected trace
    for trace_idx in selected_traces:
        target_trace = fig.data[trace_idx]

        # Extract values from the trace
        # horizontal bars/histograms have their values on the y-axis, so their ECDF
        # runs along a secondary top x-axis instead of the right y-axis
        horizontal = is_horizontal(target_trace)
        if isinstance(target_trace, (go.Histogram, go.Scatter, go.Scattergl)):
            ecdf_trace = (
                px.ecdf(y=target_trace.y, orientation="h")
                if horizontal
                else px.ecdf(target_trace.x)
            ).data[0]
        elif isinstance(target_trace, go.Bar):  # bar lengths weight their positions
            positions, lengths = target_trace.x, target_trace.y
            if horizontal:
                positions, lengths = lengths, positions
            if len(positions) == len(lengths) + 1:  # bin edges, bars at left edges
                positions = positions[:-1]
            ecdf_trace = (
                px.ecdf(x=lengths, y=positions, orientation="h")
                if horizontal
                else px.ecdf(x=positions, y=lengths)
            ).data[0]
        else:
            cls = type(target_trace)
            qual_name = f"{cls.__module__}.{cls.__qualname__}"
            raise TypeError(
                f"Cannot auto-determine x-values for ECDF from {qual_name}. "
                "Pass values explicitly or use supported trace types."
            )

        ecdf_trace.hovertemplate = (
            f"{fig.layout.yaxis.title.text or 'y'}: %{{y}}<br>"
            "Percent: %{x:.2%}<extra></extra>"
            if horizontal
            else hovertemplate
        )

        colorway = fig.layout.colorway or px.colors.qualitative.Plotly
        default_color = colorway[trace_idx % len(colorway)]
        target_color = _get_trace_color(target_trace, default_color)

        legendgroup = target_trace.legendgroup or str(trace_idx)

        # Create name for ECDF trace
        trace_name = base_name
        if not is_single_trace and getattr(target_trace, "name", None):
            trace_name = f"{base_name} ({target_trace.name})"

        # Build the trace using the determined target_color
        line_opts = {
            "color": target_color,
            "dash": trace_kwargs.get("line", {}).get("dash", "solid"),
        }

        axis_kwarg = dict(xaxis="x2") if horizontal else dict(yaxis="y2")
        ecdf_trace.update(
            dict(name=trace_name, legendgroup=legendgroup, line=line_opts) | axis_kwarg
        )
        if horizontal and not hasattr(fig.layout, "xaxis2"):
            fig.layout.xaxis2 = dict(
                title=base_name,
                side="top",
                overlaying="x",
                range=(0, 1),
                showgrid=False,
            )

        # For subplots, we just need to ensure we use the correct xaxis
        # If trace has a specific xaxis, copy it to maintain subplot structure
        if has_subplots and not horizontal and hasattr(target_trace, "xaxis"):
            ecdf_trace["xaxis"] = target_trace.xaxis

        # Add to figure with defaults
        fig.add_trace(ecdf_trace, **kwargs)

        # Apply user overrides, merging line dicts into the existing line style
        current_trace_kwargs = dict(trace_kwargs)
        if isinstance(line_kwargs := trace_kwargs.get("line"), dict):
            line = ecdf_trace.line
            current_line = dict(color=line.color, dash=line.dash, width=line.width)
            current_trace_kwargs["line"] = current_line | line_kwargs

        if current_trace_kwargs:
            fig.data[-1].update(**current_trace_kwargs)

    # Make sure yaxis2 has color set if specified in trace_kwargs
    line_settings = trace_kwargs.get("line")
    yaxis2_color = (
        line_settings.get("color") if isinstance(line_settings, dict) else None
    )
    if not isinstance(yaxis2_color, str):
        yaxis2_color = trace_kwargs.get("line_color")
    if isinstance(yaxis2_color, str):
        fig.layout.yaxis2.update(color=yaxis2_color)

    return fig


_common_update_menu = dict(
    pad={"r": 10, "t": 10}, showactive=True, x=1, xanchor="right", y=1, yanchor="top"
)

# buttons to toggle log/linear y-axis. apply to a plotly figure like this:
# fig.layout.updatemenus = [toggle_log_linear_y_axis]
# use toggle_log_linear_y_axis | dict(x=1, y=0, ...) to customize
toggle_log_linear_y_axis = dict(
    type="buttons",
    direction="left",
    buttons=[
        dict(args=[{"yaxis.type": "linear"}], label="Linear Y", method="relayout"),
        dict(args=[{"yaxis.type": "log"}], label="Log Y", method="relayout"),
    ],
    **_common_update_menu,
)


toggle_log_linear_x_axis = dict(
    type="buttons",
    direction="left",
    buttons=[
        dict(args=[{"xaxis.type": "linear"}], label="Linear X", method="relayout"),
        dict(args=[{"xaxis.type": "log"}], label="Log X", method="relayout"),
    ],
    **_common_update_menu,
)

# Toggle grid visibility
toggle_grid = dict(
    type="buttons",
    direction="left",
    buttons=[
        dict(
            args=[{"xaxis.showgrid": True, "yaxis.showgrid": True}],
            label="Show Grid",
            method="relayout",
        ),
        dict(
            args=[{"xaxis.showgrid": False, "yaxis.showgrid": False}],
            label="Hide Grid",
            method="relayout",
        ),
    ],
    **_common_update_menu,
)

# Toggle between different color scales
select_colorscale = dict(
    type="buttons",
    direction="down",
    buttons=[
        dict(args=[{"colorscale": "Viridis"}], label="Viridis", method="restyle"),
        dict(args=[{"colorscale": "Plasma"}], label="Plasma", method="restyle"),
        dict(args=[{"colorscale": "Inferno"}], label="Inferno", method="restyle"),
        dict(args=[{"colorscale": "Magma"}], label="Magma", method="restyle"),
    ],
    **_common_update_menu,
)

# Toggle between different plot types (e.g. scatter, line)
select_marker_mode = dict(
    type="buttons",
    direction="down",
    buttons=[
        dict(
            args=[{"type": "scatter", "mode": "markers"}],
            label="Scatter",
            method="restyle",
        ),
        dict(
            args=[{"type": "scatter", "mode": "lines"}], label="Line", method="restyle"
        ),
        dict(
            args=[{"type": "scatter", "mode": "lines+markers"}],
            label="Line+Markers",
            method="restyle",
        ),
    ],
    **_common_update_menu,
)
