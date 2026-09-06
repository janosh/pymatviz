"""Scatter plot visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json, normalize_plot_series
from pymatviz.widgets._traits import PlotControlsTraits
from pymatviz.widgets.matterviz import MatterVizWidget


class ScatterPlotWidget(PlotControlsTraits, MatterVizWidget):
    """MatterViz widget wrapper for 2D scatter/line plots.

    The payload follows matterviz `ScatterPlot` props with a Python-friendly API.
    """

    series = tl.List(allow_none=True).tag(sync=True)
    x_axis = tl.Dict(allow_none=True).tag(sync=True)
    y_axis = tl.Dict(allow_none=True).tag(sync=True)
    y2_axis = tl.Dict(allow_none=True).tag(sync=True)
    display = tl.Dict(allow_none=True).tag(sync=True)
    legend = tl.Dict(allow_none=True).tag(sync=True)
    styles = tl.Dict(allow_none=True).tag(sync=True)
    color_scale = tl.Dict(allow_none=True).tag(sync=True)
    size_scale = tl.Dict(allow_none=True).tag(sync=True)
    ref_lines = tl.List(allow_none=True).tag(sync=True)
    fill_regions = tl.List(allow_none=True).tag(sync=True)
    error_bands = tl.List(allow_none=True).tag(sync=True)
    show_legend = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    # auto picks canvas above the dense-point threshold, svg below
    marker_renderer = tl.CaselessStrEnum(
        values=["auto", "svg", "canvas"], allow_none=True, default_value=None
    ).tag(sync=True)
    padding = tl.Dict(allow_none=True).tag(sync=True)
    range_padding = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    x2_axis = tl.Dict(allow_none=True).tag(sync=True)
    x_range = tl.List(allow_none=True).tag(sync=True)
    x2_range = tl.List(allow_none=True).tag(sync=True)
    y_range = tl.List(allow_none=True).tag(sync=True)
    y2_range = tl.List(allow_none=True).tag(sync=True)
    color_bar = tl.Dict(allow_none=True).tag(sync=True)
    hover_config = tl.Dict(allow_none=True).tag(sync=True)
    label_placement_config = tl.Dict(allow_none=True).tag(sync=True)
    point_tween = tl.Dict(allow_none=True).tag(sync=True)
    line_tween = tl.Dict(allow_none=True).tag(sync=True)

    # Interaction state (two-way synced with the frontend for ipywidgets linking).
    # selected_point drives a highlight from Python and takes
    # {"series_idx", "point_idx"}. active_point/hovered_point are
    # populated on user click/hover (observe them to drive other widgets).
    # hovered_point is {"series_idx", "point_idx", "x", "y"} or None.
    # active_point is the last clicked point or None; it adds an
    # event_id (monotonically increasing per widget instance) so re-clicking
    # the same point is still observed as a distinct event:
    # {"series_idx", "point_idx", "x", "y", "event_id"}.
    selected_point = tl.Dict(allow_none=True, default_value=None).tag(sync=True)
    active_point = tl.Dict(allow_none=True, default_value=None).tag(sync=True)
    hovered_point = tl.Dict(allow_none=True, default_value=None).tag(sync=True)

    def __init__(
        self,
        series: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        *,
        x_axis: dict[str, Any] | None = None,
        y_axis: dict[str, Any] | None = None,
        y2_axis: dict[str, Any] | None = None,
        display: dict[str, Any] | None = None,
        legend: dict[str, Any] | None = None,
        styles: dict[str, Any] | None = None,
        color_scale: dict[str, Any] | None = None,
        size_scale: dict[str, Any] | None = None,
        ref_lines: list[dict[str, Any]] | None = None,
        fill_regions: list[dict[str, Any]] | None = None,
        error_bands: list[dict[str, Any]] | None = None,
        show_legend: bool | None = None,
        marker_renderer: str | None = None,
        padding: dict[str, int] | None = None,
        range_padding: float | None = None,
        x2_axis: dict[str, Any] | None = None,
        x_range: list[float | None] | None = None,
        x2_range: list[float | None] | None = None,
        y_range: list[float | None] | None = None,
        y2_range: list[float | None] | None = None,
        color_bar: dict[str, Any] | None = None,
        hover_config: dict[str, Any] | None = None,
        label_placement_config: dict[str, Any] | None = None,
        point_tween: dict[str, Any] | None = None,
        line_tween: dict[str, Any] | None = None,
        selected_point: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a scatter plot widget.

        Args:
            series: Plot series with required ``x`` and ``y`` arrays.
            x_axis: X axis configuration.
            y_axis: Primary Y axis configuration.
            y2_axis: Secondary Y axis configuration.
            display: Grid/zero-line display options.
            legend: Legend configuration.
            styles: Point and line style override configuration.
            color_scale: Color scaling configuration.
            size_scale: Marker size scaling configuration.
            ref_lines: Reference line definitions.
            fill_regions: Filled region definitions.
            error_bands: Error-band definitions.
            show_legend: Whether to show the legend (frontend default: auto).
            marker_renderer: ``"auto"`` (default), ``"svg"`` or ``"canvas"`` --
                how points are drawn; auto switches to canvas for dense data.
            padding: Plot area padding ``{"t": N, "b": N, "l": N, "r": N}``
                in pixels.
            range_padding: Fraction of data range to add as padding.
                Python default is ``None``; the frontend uses ``0.05``.
            x2_axis: Secondary X axis configuration.
            x_range: Fixed X axis range ``[min, max]``.
            x2_range: Fixed secondary X axis range ``[min, max]``.
            y_range: Fixed Y axis range ``[min, max]``.
            y2_range: Fixed secondary Y axis range ``[min, max]``.
            color_bar: Color bar / continuous legend configuration.
            hover_config: Tooltip hover behavior configuration.
            label_placement_config: Data label positioning configuration.
            point_tween: Point animation configuration.
            line_tween: Line animation configuration.
            selected_point: Point to highlight from Python, as
                ``{"series_idx", "point_idx"}``. Use ``observe("active_point")``
                to react to clicks and link this widget to others.
            **kwargs: Additional base widget keyword arguments.
        """
        super().__init__(
            widget_type="scatter_plot",
            series=normalize_plot_series(series, component_name="ScatterPlot"),
            **normalize_plot_json(
                dict(
                    x_axis=x_axis,
                    y_axis=y_axis,
                    y2_axis=y2_axis,
                    display=display,
                    legend=legend,
                    styles=styles,
                    color_scale=color_scale,
                    size_scale=size_scale,
                    ref_lines=ref_lines,
                    fill_regions=fill_regions,
                    error_bands=error_bands,
                    padding=padding,
                    x2_axis=x2_axis,
                    x_range=x_range,
                    x2_range=x2_range,
                    y_range=y_range,
                    y2_range=y2_range,
                    color_bar=color_bar,
                    hover_config=hover_config,
                    label_placement_config=label_placement_config,
                    point_tween=point_tween,
                    line_tween=line_tween,
                ),
                "ScatterPlot",
            ),
            show_legend=show_legend,
            marker_renderer=marker_renderer,
            range_padding=range_padding,
            selected_point=selected_point,
            **kwargs,
        )
