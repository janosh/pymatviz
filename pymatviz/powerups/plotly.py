"""Powerups for plotly figures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from plotly.basedatatypes import BaseTraceType


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
    trace_kwargs = trace_kwargs or {}
    valid_fig_types = (go.Figure,)
    if not isinstance(fig, valid_fig_types):
        type_names = " | ".join(
            f"{t.__module__}.{t.__qualname__}" for t in valid_fig_types
        )
        raise TypeError(f"{fig=} must be instance of {type_names}")

    target_trace: BaseTraceType = fig.data[trace_idx]
    if values is None or len(values) == 0:
        if isinstance(target_trace, (go.Histogram, go.Scatter, go.Scattergl)):
            values = target_trace.x
        elif isinstance(target_trace, go.Bar):
            xs, ys = target_trace.x, target_trace.y
            # if xs are bin edges, drop last and interpret as bin centers
            if len(xs) + 1 == len(ys):
                xs = xs[:-1]
            values = np.repeat(xs, ys)
        else:
            cls = type(target_trace)
            qual_name = f"{cls.__module__}.{cls.__qualname__}"
            raise ValueError(
                f"Cannot auto-determine x-values for ECDF from {qual_name}, "
                "pass values explicitly. Currently only Histogram, Scatter, Box, "
                "and Violin traces are supported and may well need more testing. "
                "Please report issues at https://github.com/janosh/pymatviz/issues."
            )

    ecdf_trace = px.ecdf(values).data[0]

    # if fig has facets, add ECDF to all subplots
    fig_add_trace_defaults = {} if fig._grid_ref is None else dict(row="all", col="all")  # noqa: SLF001
    fig.add_trace(ecdf_trace, **fig_add_trace_defaults | kwargs)

    # get xlabel falling back on 'x' if not set
    xlabel = fig.layout.xaxis.title.text or "x"
    tooltip_template = f"{xlabel}: %{{x}}<br>Percent: %{{y:.2%}}<extra></extra>"
    fig.data[-1].hovertemplate = tooltip_template

    # move ECDF line to secondary y-axis
    # set color to darkened version of primary y-axis color
    name = uniq_name = trace_kwargs.get("name", "Cumulative")
    trace_names = [trace.name for trace in fig.data]
    name_suffix = 1
    while uniq_name in trace_names:
        name_suffix += 1
        uniq_name = f"{name} {name_suffix}"

    if "marker" in target_trace and target_trace.marker["color"] is None:
        dev_trace = fig.full_figure_for_development(warn=False).data[trace_idx]
        # NOTE unsure if this has any downstream effects, should just be applying
        # the color that plotly would have later assigned to the trace anyway
        target_trace.marker = dev_trace.marker

    # set default legendgroup of target trace if not already set
    legendgroup = target_trace.legendgroup or target_trace.name
    target_trace.legendgroup = legendgroup

    # set line color to be the same as the target trace's marker color
    target_color = target_trace.marker.color if "marker" in target_trace else "gray"
    trace_defaults = dict(
        yaxis="y2",
        name=uniq_name,
        line=dict(
            color=target_color,
            dash="solid",
        ),
        legendgroup=legendgroup,
    )
    trace_kwargs = trace_defaults | trace_kwargs
    fig.data[-1].update(**trace_kwargs)

    # line_color becomes target_color via trace_defaults if a different color was not
    # already set in trace_kwargs
    line_color = trace_kwargs.get(
        "line_color", trace_kwargs.get("line", {}).get("color")
    )

    yaxis_defaults = dict(
        title=uniq_name,
        side="right",
        overlaying="y",
        range=(0, 1),
        showgrid=False,
        color=line_color,
    )
    # make secondary ECDF y-axis inherit primary y-axis styles
    yaxis2_layout = getattr(fig.layout, "yaxis2", {})
    if yaxis2_layout:
        # convert to dict
        yaxis2_layout = yaxis2_layout._props  # type: ignore[union-attr] # noqa: SLF001
    fig.layout.yaxis2 = yaxis_defaults | yaxis2_layout

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

# Toggle between different plot types (e.g., scatter, line)
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
