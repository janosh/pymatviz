"""Powerups for plotly figures."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, get_args

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


def annotate_metrics(
    xs: ArrayLike,
    ys: ArrayLike,
    fig: go.Figure | None = None,
    metrics: dict[str, float] | Sequence[str] = ("MAE", "R2"),
    prefix: str = "",
    suffix: str = "",
    fmt: str = ".3",
    **kwargs: Any,
) -> go.Figure:
    """Provide a set of x and y values of equal length and an optional Figure
    object on which to print the values' mean absolute error and R^2
    coefficient of determination.

    Args:
        xs (array): x values.
        ys (array): y values.
        fig (go.Figure | None, optional): plotly Figure on which to add the annotation.
            Defaults to None.
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
        go.Figure: The annotated figure.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    if not isinstance(fig, go.Figure):
        raise TypeError(f"{fig=} must be instance of go.Figure")
    if not isinstance(metrics, (dict, list, tuple, set)):
        raise TypeError(
            f"metrics must be dict|list|tuple|set, not {type(metrics).__name__}"
        )

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
                label = PRETTY_LABELS.get(key, key)
                text += f"{label} = {val:{fmt}}<br>"
        else:
            for key in metrics:
                value = funcs[key](xs, ys)
                label = str(PRETTY_LABELS.get(key, key))
                text += f"{label} = {value:{fmt}}<br>"
        text += suffix
        return text

    if isinstance(fig, go.Figure) and any(
        getattr(trace, "xaxis", None) not in ("x", None) for trace in fig.data
    ):
        # Handle faceted Plotly figure
        texts = []
        for trace in fig.data:
            trace_xs, trace_ys = trace.x, trace.y
            texts.append(calculate_metrics(trace_xs, trace_ys))
        return annotate(texts, fig, **kwargs)

    # Handle non-faceted figures
    text = calculate_metrics(xs, ys)
    return annotate(text, fig, **kwargs)


def add_identity_line(
    fig: go.Figure,
    *,
    line_kwargs: dict[str, Any] | None = None,
    traces: TraceSelector = lambda _: True,
    retain_xy_limits: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """Add a line shape to the background layer of a plotly figure spanning
    from smallest to largest x/y values in the trace specified by traces.

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

    xy_min_min = min(x_min, y_min)
    xy_max_min = min(x_max, y_max)

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
        x0=xy_min_min,
        y0=xy_min_min,
        x1=xy_max_min,
        y1=xy_max_min,
        layer="below",
        line=line_defaults,
        **kwargs,
    )
    if retain_xy_limits:
        fig.update_xaxes(range=[x_min, x_max])
        fig.update_yaxes(range=[y_min, y_max])

    return fig


def _get_valid_traces(
    fig: go.Figure,
    traces: TraceSelector,
    validate_trace: Callable[[go.Scatter], bool] | None = None,
) -> list[int]:
    """Helper to get valid trace indices from a trace selector.

    Args:
        fig (go.Figure): The figure containing traces to filter
        traces (TraceSelector): Trace selector (int, slice, sequence, or callable)
        validate_trace (Callable[[go.Scatter], bool] | None): Optional function to
            validate if a trace is usable.
            If not provided, requires both x and y data.

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
        selected_traces = [idx for idx in traces if 0 <= idx < len(fig.data)]
        if not selected_traces:
            valid_range = f"0-{len(fig.data) - 1}"
            raise ValueError(f"No valid trace indices in {traces}, {valid_range=}")
    elif callable(traces):
        selected_traces = [idx for idx, trace in enumerate(fig.data) if traces(trace)]
        if not selected_traces:
            raise ValueError("No traces matched the filtering function")
    else:
        actual = type(traces).__name__
        raise TypeError(
            f"traces must be int, slice, sequence, or callable, got {actual}"
        )

    # Default validator: trace needs both x and y
    if validate_trace is None:
        validate_trace = lambda trace: (
            hasattr(trace, "x")
            and hasattr(trace, "y")
            and trace.x is not None
            and trace.y is not None
            and len(trace.x) > 0
            and len(trace.y) > 0
        )

    # Filter for traces with valid data according to validator
    valid_traces = [idx for idx in selected_traces if validate_trace(fig.data[idx])]

    if not valid_traces:
        raise ValueError("No valid traces with required data found")

    return valid_traces


def _get_trace_color(trace: go.Scatter, default_color: str = "navy") -> str:
    """Extract color from trace, first checking marker then line."""
    if (
        hasattr(trace, "marker")
        and hasattr(trace.marker, "color")
        and not isinstance(trace.marker.color, list)
    ):
        return trace.marker.color
    if hasattr(trace, "line") and hasattr(trace.line, "color"):
        return trace.line.color
    return default_color


def add_best_fit_line(
    fig: go.Figure,
    *,
    xs: ArrayLike = (),
    ys: ArrayLike = (),
    traces: TraceSelector = lambda _: True,
    line_kwargs: dict[str, Any] | None = None,
    annotate_params: bool | dict[str, Any] = True,
    annotation_mode: AnnotationMode = "per_trace",
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
        annotation_mode (AnnotationMode, optional): How to display annotations. Options:
            - "per_trace": Each selected trace gets its own annotation
            - "combined": All selected traces get a combined annotation
            - "none": No annotations shown
            Defaults to "per_trace".
        **kwargs: Additional arguments are passed to fig.add_shape().

    Raises:
        TypeError: If fig is not a plotly figure.
        ValueError: If fig is a plotly figure and xs and ys are not provided and
            no valid traces are found.

    Returns:
        go.Figure: Figure with added best fit line.
    """
    if not isinstance(fig, go.Figure):
        raise TypeError(f"{fig=} must be instance of go.Figure")

    # Determine styling
    default_color = "navy" if luminance(get_font_color(fig)) < 0.7 else "lightskyblue"
    line_color = kwargs.pop(
        "color",
        annotate_params.get("color", default_color)
        if isinstance(annotate_params, dict)
        else default_color,
    )

    # Clear existing LS fit annotations
    annotation_count = 0
    if (
        hasattr(fig.layout, "annotations")
        and fig.layout.annotations
        and any("LS fit: y =" in anno.text for anno in fig.layout.annotations)
        and annotation_mode != "none"
    ):
        # Count existing annotations
        annotation_count = sum(
            "LS fit: y =" in anno.text for anno in fig.layout.annotations
        )
        if annotation_mode == "combined":
            # Remove all existing annotations when in combined mode
            fig.layout.annotations = [
                anno
                for anno in fig.layout.annotations
                if "LS fit: y =" not in anno.text
            ]

    # Function to add best fit line with annotation
    def add_fit_line(
        data_xs: ArrayLike,
        data_ys: ArrayLike,
        color: str,
        xref: str = "x",
        yref: str = "y",
    ) -> None:
        # Calculate line parameters
        slope, intercept = np.polyfit(data_xs, data_ys, 1)
        x_min, x_max = min(data_xs), max(data_xs)
        y0, y1 = slope * x_min + intercept, slope * x_max + intercept

        # Add line
        plotly_line_defaults = dict(color=color, width=2, dash="dash")
        if line_kwargs:
            plotly_line_defaults.update(line_kwargs)
        fig.add_shape(
            type="line",
            x0=x_min,
            y0=y0,
            x1=x_max,
            y1=y1,
            xref=xref,
            yref=yref,
            line=plotly_line_defaults,
        )

        # Add annotation if requested
        if not annotate_params or annotation_mode == "none":
            return

        sign = "+" if intercept >= 0 else "-"
        text = f"LS fit: y = {slope:.2g}x {sign} {abs(intercept):.2g}"

        # Calculate annotation position
        nonlocal annotation_count  # Use outer variable to track annotation count
        y_offset = 0.05 * annotation_count
        annotation_count += 1  # Increment for the next annotation

        plotly_anno_defaults = dict(
            x=0.98,
            y=0.02 + y_offset,  # Use cumulative offset
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font=dict(color=color),
        )

        # For faceted plots, set proper references
        if xref != "x" or yref != "y":
            plotly_anno_defaults["xref"] = f"{xref} domain"
            plotly_anno_defaults["yref"] = f"{yref} domain"

        if isinstance(annotate_params, dict):
            plotly_anno_defaults.update(annotate_params)
            # Update y-position after applying custom parameters
            if "y" not in annotate_params:
                plotly_anno_defaults["y"] = 0.02 + y_offset

            # Ensure font color is properly set
            if "color" in annotate_params:
                if "font" not in plotly_anno_defaults:
                    plotly_anno_defaults["font"] = dict()
                plotly_anno_defaults["font"]["color"] = annotate_params["color"]  # type: ignore[index]

        annotate(text, fig=fig, **plotly_anno_defaults)

    # CASE 1: Custom data provided directly
    if len(xs) > 0 and len(ys) > 0:
        add_fit_line(xs, ys, line_color)
        return fig

    # Get valid traces for plotly
    valid_traces = _get_valid_traces(fig, traces, None)

    # Check if this is a faceted plot
    is_faceted = any(
        hasattr(trace, "xaxis") and trace.xaxis != "x" for trace in fig.data
    )

    # CASE 2A: Faceted plotly plot
    if is_faceted:
        # Group traces by subplot
        subplot_groups: dict[str, list[int]] = {}
        for idx in valid_traces:
            trace = fig.data[idx]
            subplot = trace.xaxis if hasattr(trace, "xaxis") and trace.xaxis else "x"
            if subplot not in subplot_groups:
                subplot_groups[subplot] = []
            subplot_groups[subplot].append(idx)

        # Process each subplot
        for subplot, traces_in_subplot in subplot_groups.items():
            trace = fig.data[traces_in_subplot[0]]
            subplot_idx_str = subplot[1:] if len(subplot) > 1 else ""
            xref, yref = subplot, f"y{subplot_idx_str}" if subplot_idx_str else "y"
            color = _get_trace_color(trace, "navy")
            add_fit_line(trace.x, trace.y, color, xref, yref)

        return fig

    # CASE 2B: Combined annotation mode for plotly
    if annotation_mode == "combined":
        all_xs = np.concatenate([fig.data[idx].x for idx in valid_traces])
        all_ys = np.concatenate([fig.data[idx].y for idx in valid_traces])
        color = (
            _get_trace_color(fig.data[valid_traces[0]], line_color)
            if valid_traces
            else line_color
        )
        add_fit_line(all_xs, all_ys, color)
        return fig

    # CASE 2C: Per-trace annotation mode for plotly
    processed_traces: set[int] = set()
    for trace_idx in valid_traces:
        if trace_idx in processed_traces:
            continue
        processed_traces.add(trace_idx)

        trace = fig.data[trace_idx]
        if len(trace.x) < 2 or len(trace.y) < 2:
            continue

        color = _get_trace_color(trace, line_color)
        add_fit_line(trace.x, trace.y, color, "x", "y")

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
    # Add identity line if requested
    if identity_line and fig is not None:
        identity_kwargs = identity_line if isinstance(identity_line, dict) else {}
        add_identity_line(fig, traces=traces, **identity_kwargs)

    # Early return if no stats or best-fit line needed and manual data not provided
    if not stats and best_fit_line is not True and len(xs) == 0 and len(ys) == 0:
        return fig

    # Case 1: Data provided directly
    if len(xs) > 0 and len(ys) > 0:
        # If best_fit_line is None, determine whether to add it based on R²
        if best_fit_line is None:
            r2 = r2_score(xs, ys)
            best_fit_line = r2 > 0.3

        # Add best fit line if requested
        if best_fit_line and fig is not None:
            direct_best_fit_kwargs = (
                {} if isinstance(best_fit_line, bool) else best_fit_line
            )
            add_best_fit_line(fig, xs=xs, ys=ys, **direct_best_fit_kwargs)

        # Add stats annotation if requested
        if stats and annotation_mode != "none" and fig is not None:
            direct_stats_kwargs = {} if isinstance(stats, bool) else stats
            annotate_metrics(xs, ys, fig=fig, **direct_stats_kwargs)

        return fig

    # Case 2: Need to extract data from figure
    if not isinstance(fig, go.Figure):
        raise TypeError(
            "this powerup can only get x/y data from the figure directly for plotly "
            "figures."
        )

    # Get valid traces
    valid_traces = _get_valid_traces(fig, traces, None)

    # Handle different annotation modes
    if annotation_mode == "combined":
        # Combine data from all selected traces
        all_xs = np.concatenate([fig.data[idx].x for idx in valid_traces])
        all_ys = np.concatenate([fig.data[idx].y for idx in valid_traces])

        # Add overall best-fit line if requested
        if best_fit_line is None:
            r2 = r2_score(all_xs, all_ys)
            best_fit_line = r2 > 0.3

        if best_fit_line:
            combined_best_fit_kwargs = (
                {} if isinstance(best_fit_line, bool) else best_fit_line
            )
            add_best_fit_line(fig, xs=all_xs, ys=all_ys, **combined_best_fit_kwargs)

        # Add combined stats annotation if requested
        if stats and annotation_mode != "none":
            combined_stats_kwargs = {} if isinstance(stats, bool) else stats
            annotate_metrics(all_xs, all_ys, fig=fig, **combined_stats_kwargs)

        return fig

    if annotation_mode == "per_trace":
        all_traced_processed: set[int] = (
            set()
        )  # Keep track of processed traces to avoid duplicates

        for trace_idx in valid_traces:
            if trace_idx in all_traced_processed:
                continue
            all_traced_processed.add(trace_idx)

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
                # Prepare line style
                if isinstance(best_fit_line, dict):
                    per_trace_line_style = dict(color=trace_color, width=2, dash="dash")
                    per_trace_line_style.update(best_fit_line)
                    add_best_fit_line(
                        fig, xs=trace_xs, ys=trace_ys, line_kwargs=per_trace_line_style
                    )
                else:
                    add_best_fit_line(
                        fig,
                        xs=trace_xs,
                        ys=trace_ys,
                        line_kwargs={"color": trace_color},
                    )

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
            line_color (default "gray").
            You can pass 'offset_size' to adjust spacing between multiple annotations.
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

    # If explicit values are provided, just add one ECDF line
    if values is not None and len(values) > 0:
        ecdf_trace = px.ecdf(values).data[0]
        fig.add_trace(ecdf_trace, **kwargs)

        # Set hover template
        xlabel = fig.layout.xaxis.title.text or "x"
        fig.data[
            -1
        ].hovertemplate = f"{xlabel}: %{{x}}<br>Percent: %{{y:.2%}}<extra></extra>"

        # Create unique name
        name = base_name
        trace_names = [trace.name for trace in fig.data[:-1]]
        name_suffix = 1
        while name in trace_names:
            name_suffix += 1
            name = f"{base_name} {name_suffix}"

        # Set up trace defaults and apply
        line_color = trace_kwargs.get("line_color", "gray")
        line_dash = trace_kwargs.get("line", {}).get("dash", "solid")

        trace_defaults = dict(
            yaxis="y2",
            name=name,
            line=dict(
                color=line_color,
                dash=line_dash,
            ),
        )

        # Apply user overrides
        current_trace_kwargs = trace_defaults.copy()
        for key, value in trace_kwargs.items():
            if key == "line" and isinstance(value, dict):
                current_trace_kwargs["line"] = {**trace_defaults["line"], **value}
            else:
                current_trace_kwargs[key] = value

        fig.data[-1].update(**current_trace_kwargs)

        # Set up y-axis
        yaxis_defaults = dict(
            title=name,
            side="right",
            overlaying="y",
            range=(0, 1),
            showgrid=False,
        )

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
    def validate_ecdf_trace(trace: go.Scatter) -> bool:
        if isinstance(trace, (go.Histogram, go.Bar)):
            return hasattr(trace, "x") and trace.x is not None and len(trace.x) > 0
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
        fig.layout.yaxis2 = dict(
            title=base_name,
            side="right",
            overlaying="y",
            range=(0, 1),
            showgrid=False,
        )

    # Process each selected trace
    for cnt, trace_idx in enumerate(selected_traces):
        target_trace = fig.data[trace_idx]

        # Extract values from the trace
        if isinstance(target_trace, (go.Histogram, go.Scatter, go.Scattergl)):
            trace_values = target_trace.x
        elif isinstance(target_trace, go.Bar):
            xs, ys = target_trace.x, target_trace.y
            if len(xs) + 1 == len(ys):  # if xs are bin edges
                xs = xs[:-1]
            trace_values = np.repeat(xs, ys)
        else:
            cls = type(target_trace)
            qual_name = f"{cls.__module__}.{cls.__qualname__}"
            raise TypeError(
                f"Cannot auto-determine x-values for ECDF from {qual_name}. "
                "Pass values explicitly or use supported trace types."
            )

        # Create ECDF trace
        ecdf_trace = px.ecdf(trace_values).data[0]

        # Set hover template
        xlabel = fig.layout.xaxis.title.text or "x"
        ecdf_trace.hovertemplate = (
            f"{xlabel}: %{{x}}<br>Percent: %{{y:.2%}}<extra></extra>"
        )

        # Try full figure development view first for potentially resolved color
        # Use try-except block for robustness if full_fig fails
        full_fig = fig.full_figure_for_development(warn=False)
        if trace_idx < len(full_fig.data):
            full_trace = full_fig.data[trace_idx]
            # Try getting color from full_trace, don't provide a default yet
            target_color = _get_trace_color(full_trace)
        else:
            mod_trace_idx = trace_idx % len(px.colors.DEFAULT_PLOTLY_COLORS)
            target_color = px.colors.DEFAULT_PLOTLY_COLORS[mod_trace_idx]

        # Set legendgroup
        legendgroup = str(trace_idx)
        if getattr(target_trace, "legendgroup", None):
            legendgroup = target_trace.legendgroup

        # Create name for ECDF trace
        trace_name = base_name
        if not is_single_trace and getattr(target_trace, "name", None):
            trace_name = f"{base_name} ({target_trace.name})"

        # Build the trace using the determined target_color
        line_opts = {
            "color": target_color,
            "dash": trace_kwargs.get("line", {}).get("dash", "solid"),
        }

        ecdf_trace.update(
            dict(
                yaxis="y2",
                name=trace_name,
                legendgroup=legendgroup,
                line=line_opts,
            )
        )

        # For subplots, we just need to ensure we use the correct xaxis
        # If trace has a specific xaxis, copy it to maintain subplot structure
        if has_subplots and hasattr(target_trace, "xaxis"):
            ecdf_trace["xaxis"] = target_trace.xaxis

        # Add to figure with defaults
        fig.add_trace(ecdf_trace, **kwargs)

        # Apply user overrides
        current_trace_kwargs = {}
        for key, value in trace_kwargs.items():
            if key == "line" and isinstance(value, dict):
                # Get existing line properties as dict
                current_line = {}
                if hasattr(ecdf_trace, "line"):
                    if hasattr(ecdf_trace.line, "color"):
                        current_line["color"] = ecdf_trace.line.color
                    if hasattr(ecdf_trace.line, "dash"):
                        current_line["dash"] = ecdf_trace.line.dash
                    if hasattr(ecdf_trace.line, "width"):
                        current_line["width"] = ecdf_trace.line.width
                current_trace_kwargs["line"] = {**current_line, **value}
            else:
                current_trace_kwargs[key] = value

        # Add vertical offset for annotation text if multiple traces
        if (
            len(selected_traces) > 1
            and not has_subplots
            and "text" in current_trace_kwargs
            and "y" not in current_trace_kwargs
        ):
            # Get font size and custom offset if provided
            font_size = 12  # Default Plotly font size
            offset_size = current_trace_kwargs.pop("offset_size", None)

            # Calculate offset based on 1.5x font size in normalized coordinates
            default_offset = 1.5 * font_size / 400
            y_offset = offset_size if offset_size is not None else default_offset
            y_offset = cnt * y_offset

            current_trace_kwargs["y"] = 0.02 + y_offset
            current_trace_kwargs["yanchor"] = "bottom"

            # Add annotation to figure layout for tests to find
            fig.add_annotation(
                text=current_trace_kwargs["text"],
                x=0.98,
                y=0.02 + y_offset,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
            )

        if current_trace_kwargs:
            fig.data[-1].update(**current_trace_kwargs)

    # Make sure yaxis2 has color set if specified in trace_kwargs
    if "line" in trace_kwargs and "color" in trace_kwargs["line"]:
        fig.layout.yaxis2.color = trace_kwargs["line"]["color"]
    elif "line_color" in trace_kwargs:
        fig.layout.yaxis2.color = trace_kwargs["line_color"]

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
