"""Powerups that can be applied to both matplotlib and plotly figures."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, get_args

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import sklearn
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score

from pymatviz.typing import (
    MATPLOTLIB,
    PLOTLY,
    VALID_FIG_NAMES,
    VALID_FIG_TYPES,
    AxOrFig,
    Backend,
)
from pymatviz.utils import (
    annotate,
    get_fig_xy_range,
    get_font_color,
    luminance,
    pretty_label,
    validate_fig,
)


if TYPE_CHECKING:
    from matplotlib.offsetbox import AnchoredText
    from numpy.typing import ArrayLike

TracePredicate = Callable[[go.Scatter], bool]
TraceSelector = int | slice | Sequence[int] | TracePredicate


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
    line_kwargs: dict[str, Any] | None = None,
    traces: TraceSelector = lambda _: True,
    retain_xy_limits: bool = False,
    **kwargs: Any,
) -> go.Figure | plt.Figure | plt.Axes:
    """Add a line shape to the background layer of a plotly figure spanning
    from smallest to largest x/y values in the trace specified by traces.

    Args:
        fig (go.Figure | plt.Figure | plt.Axes): plotly/matplotlib figure or axes to
            add the identity line to.
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
        TypeError: If fig is neither a plotly nor a matplotlib figure or axes.
        ValueError: If fig is a plotly figure and no valid traces are found.

    Returns:
        Figure: Figure with added identity line.
    """
    (x_min, x_max), (y_min, y_max) = get_fig_xy_range(fig=fig, traces=traces)

    if isinstance(fig, plt.Figure | plt.Axes):  # handle matplotlib
        ax = fig if isinstance(fig, plt.Axes) else fig.gca()

        line_defaults = dict(alpha=0.5, zorder=0, linestyle="dashed", color="black")
        ax.axline((x_min, x_min), (x_max, x_max), **line_defaults | (line_kwargs or {}))
        return fig

    if isinstance(fig, go.Figure):
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

    raise TypeError(f"{fig=} must be instance of {VALID_FIG_NAMES}")


AnnotationMode = Literal["per_trace", "combined", "none"]


def _get_valid_traces(
    fig: go.Figure,
    traces: TraceSelector,
    validate_trace: Callable[[go.Scatter], bool] | None = None,
) -> list[int]:
    """Helper to get valid trace indices from a trace selector.

    Args:
        fig: The figure containing traces to filter
        traces: Trace selector (int, slice, sequence, or callable)
        validate_trace: Optional function to validate if a trace is usable.
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
    fig: go.Figure | plt.Figure | plt.Axes,
    *,
    xs: ArrayLike = (),
    ys: ArrayLike = (),
    traces: TraceSelector = lambda _: True,
    line_kwargs: dict[str, Any] | None = None,
    annotate_params: bool | dict[str, Any] = True,
    annotation_mode: AnnotationMode = "per_trace",
    **kwargs: Any,
) -> go.Figure | plt.Figure | plt.Axes:
    """Add line of best fit according to least squares to a plotly or matplotlib figure.

    Args:
        fig (go.Figure | plt.Figure | plt.Axes): plotly/matplotlib figure or axes to
            add the best fit line to.
        xs (array, optional): x-values to use for fitting. Defaults to () which
            means use the x-values of traces selected by the traces parameter.
        ys (array, optional): y-values to use for fitting. Defaults to () which
            means use the y-values of traces selected by the traces parameter.
        traces (TraceSelector, optional): Which trace(s) to use. Can be int, slice,
            sequence of ints, or a function that takes a trace and returns True/False.
            By default, applies to all traces. Only used when xs and ys not provided.
        line_kwargs (dict[str, Any], optional): Keyword arguments for customizing the
            line shape. For plotly, will be passed to fig.add_shape(line=line_kwargs).
            For matplotlib, will be passed to ax.plot(). Defaults to None.
        annotate_params (bool | dict[str, Any], optional): Pass dict to customize
            the annotation of the best fit line. Set to False to disable annotation.
            Defaults to True.
        annotation_mode (AnnotationMode, optional): How to display annotations. Options:
            - "per_trace": Each selected trace gets its own annotation
            - "combined": All selected traces get a combined annotation
            - "none": No annotations shown
            Defaults to "per_trace".
        **kwargs: Additional arguments are passed to fig.add_shape() for plotly or
            ax.plot() for matplotlib.

    Raises:
        TypeError: If fig is neither a plotly nor a matplotlib figure or axes.
        ValueError: If fig is a plotly figure and xs and ys are not provided and
            no valid traces are found.

    Returns:
        Figure: Figure with added best fit line.
    """
    if not isinstance(fig, VALID_FIG_TYPES):
        raise TypeError(f"{fig=} must be instance of {VALID_FIG_NAMES}")

    # Determine backend and set up basic styling
    backend: Backend = PLOTLY if isinstance(fig, go.Figure) else MATPLOTLIB
    default_color = "navy" if luminance(get_font_color(fig)) < 0.7 else "lightskyblue"
    line_color = kwargs.pop(
        "color",
        annotate_params.get("color", default_color)
        if isinstance(annotate_params, dict)
        else default_color,
    )

    # Clear existing LS fit annotations for plotly
    annotation_count = 0
    if (
        backend == PLOTLY
        and hasattr(fig.layout, "annotations")
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

        # Add line based on backend
        if backend == MATPLOTLIB:
            ax = fig if isinstance(fig, plt.Axes) else fig.gca()
            mpl_line_defaults = dict(alpha=0.7, linestyle="--", zorder=1, color=color)
            if line_kwargs:
                mpl_line_defaults.update(line_kwargs)
            if kwargs:
                mpl_line_defaults.update(kwargs)
            ax.axline((x_min, y0), (x_max, y1), **mpl_line_defaults)
        else:  # Plotly
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

        if backend == MATPLOTLIB:
            mpl_anno_defaults = dict(loc="lower right", color=color)
            if isinstance(annotate_params, dict):
                mpl_anno_defaults.update(annotate_params)
            annotate(text, fig=fig, **mpl_anno_defaults)
        else:  # Plotly
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

    # CASE 2: Matplotlib - extract data from the selected trace
    if backend == MATPLOTLIB:
        ax = fig if isinstance(fig, plt.Axes) else fig.gca()
        trace_idx = 0 if not isinstance(traces, int) else traces
        artist = ax.get_children()[trace_idx]

        if isinstance(artist, plt.Line2D):
            xs, ys = artist.get_xdata(), artist.get_ydata()
        else:
            xs, ys = artist.get_offsets().T

        add_fit_line(xs, ys, line_color)
        return fig

    # Get valid traces for plotly
    valid_traces = _get_valid_traces(fig, traces, None)

    # Check if this is a faceted plot
    is_faceted = any(
        hasattr(trace, "xaxis") and trace.xaxis != "x" for trace in fig.data
    )

    # CASE 3A: Faceted plotly plot
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

    # CASE 3B: Combined annotation mode for plotly
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

    # CASE 3C: Per-trace annotation mode for plotly
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
    fig: AxOrFig | None = None,
    xs: ArrayLike = (),
    ys: ArrayLike = (),
    *,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] | None = None,
    stats: bool | dict[str, Any] | None = True,
    traces: TraceSelector = lambda _: True,
    annotation_mode: AnnotationMode = "combined",
) -> AxOrFig:
    """Add parity plot powerups to either a plotly or matplotlib figure, including
    identity line (y=x), best-fit line, and pred vs ref statistics (MAE, R², ...).

    Args:
        xs (array): x-values to use for fitting best-fit line and computing stats.
        ys (array): y-values to use for fitting best-fit line and computing stats.
        fig (plt.Axes | plt.Figure | go.Figure | None): matplotlib Axes or Figure or
            plotly Figure to add powerups to. Defaults to None.
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
        plt.Axes | plt.Figure | go.Figure: The enhanced figure.

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
            "figures. for matplotlib, pass xs and ys explicitly."
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
