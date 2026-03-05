"""Scatter plot visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json, normalize_plot_series
from pymatviz.widgets.matterviz import MatterVizWidget


class ScatterPlotWidget(MatterVizWidget):
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
    controls = tl.Dict(allow_none=True).tag(sync=True)
    padding = tl.Dict(allow_none=True).tag(sync=True)
    range_padding = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    show_legend = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
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
    point_events = tl.Unicode(allow_none=True).tag(sync=True)

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
        controls: dict[str, Any] | None = None,
        padding: dict[str, int] | None = None,
        range_padding: float | None = None,
        show_legend: bool | None = None,
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
        point_events: str | None = None,
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
            controls: Control pane configuration.
            padding: Plot area padding ``{"t": N, "b": N, "l": N, "r": N}``
                in pixels.
            range_padding: Fraction of data range to add as padding.
                Python default is ``None``; the frontend uses ``0.05``.
            show_legend: Whether to show the legend.
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
            point_events: CSS ``pointer-events`` value for data points.
            **kwargs: Additional base widget keyword arguments.
        """
        super().__init__(
            widget_type="scatter_plot",
            series=normalize_plot_series(series, component_name="ScatterPlot"),
            x_axis=normalize_plot_json(x_axis, "ScatterPlot.x_axis"),
            y_axis=normalize_plot_json(y_axis, "ScatterPlot.y_axis"),
            y2_axis=normalize_plot_json(y2_axis, "ScatterPlot.y2_axis"),
            display=normalize_plot_json(display, "ScatterPlot.display"),
            legend=normalize_plot_json(legend, "ScatterPlot.legend"),
            styles=normalize_plot_json(styles, "ScatterPlot.styles"),
            color_scale=normalize_plot_json(color_scale, "ScatterPlot.color_scale"),
            size_scale=normalize_plot_json(size_scale, "ScatterPlot.size_scale"),
            ref_lines=normalize_plot_json(ref_lines, "ScatterPlot.ref_lines"),
            fill_regions=normalize_plot_json(fill_regions, "ScatterPlot.fill_regions"),
            error_bands=normalize_plot_json(error_bands, "ScatterPlot.error_bands"),
            controls=normalize_plot_json(controls, "ScatterPlot.controls"),
            padding=normalize_plot_json(padding, "ScatterPlot.padding"),
            range_padding=range_padding,
            show_legend=show_legend,
            x2_axis=normalize_plot_json(x2_axis, "ScatterPlot.x2_axis"),
            x_range=x_range,
            x2_range=x2_range,
            y_range=y_range,
            y2_range=y2_range,
            color_bar=normalize_plot_json(color_bar, "ScatterPlot.color_bar"),
            hover_config=normalize_plot_json(hover_config, "ScatterPlot.hover_config"),
            label_placement_config=normalize_plot_json(
                label_placement_config, "ScatterPlot.label_placement_config"
            ),
            point_tween=normalize_plot_json(point_tween, "ScatterPlot.point_tween"),
            line_tween=normalize_plot_json(line_tween, "ScatterPlot.line_tween"),
            point_events=point_events,
            **kwargs,
        )
