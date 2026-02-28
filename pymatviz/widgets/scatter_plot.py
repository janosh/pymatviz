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
            **kwargs,
        )
