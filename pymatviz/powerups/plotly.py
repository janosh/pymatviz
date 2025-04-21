"""Powerups for plotly figures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pymatviz.powerups.both import TraceSelector, _get_trace_color, _get_valid_traces


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike


def add_ecdf_line(
    fig: go.Figure,
    values: ArrayLike = (),
    traces: TraceSelector = lambda _: True,
    trace_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Add an empirical cumulative distribution function (ECDF) line to a plotly figure.

    Support for matplotlib planned but not implemented. PRs welcome.

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
        Figure: Figure with added ECDF line(s).
    """
    trace_kwargs = trace_kwargs or {}
    valid_fig_types = (go.Figure,)
    if not isinstance(fig, valid_fig_types):
        type_names = " | ".join(
            f"{t.__module__}.{t.__qualname__}" for t in valid_fig_types
        )
        raise TypeError(f"{fig=} must be instance of {type_names}")

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

        if color := current_trace_kwargs.get("line", {}).get("color"):
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
