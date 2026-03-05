"""Bar plot visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json, normalize_plot_series
from pymatviz.widgets.matterviz import MatterVizWidget


class BarPlotWidget(MatterVizWidget):
    """MatterViz widget wrapper for grouped/stacked/overlay bar plots."""

    series = tl.List(allow_none=True).tag(sync=True)
    orientation = tl.CaselessStrEnum(
        values=["vertical", "horizontal"], default_value="vertical"
    ).tag(sync=True)
    mode = tl.CaselessStrEnum(
        values=["overlay", "stacked", "grouped"], default_value="overlay"
    ).tag(sync=True)
    x_axis = tl.Dict(allow_none=True).tag(sync=True)
    y_axis = tl.Dict(allow_none=True).tag(sync=True)
    y2_axis = tl.Dict(allow_none=True).tag(sync=True)
    display = tl.Dict(allow_none=True).tag(sync=True)
    legend = tl.Dict(allow_none=True).tag(sync=True)
    bar = tl.Dict(allow_none=True).tag(sync=True)
    line = tl.Dict(allow_none=True).tag(sync=True)
    ref_lines = tl.List(allow_none=True).tag(sync=True)
    controls = tl.Dict(allow_none=True).tag(sync=True)
    padding = tl.Dict(allow_none=True).tag(sync=True)
    range_padding = tl.Float(allow_none=True, default_value=None).tag(sync=True)
    show_legend = tl.Bool(allow_none=True, default_value=None).tag(sync=True)
    x2_axis = tl.Dict(allow_none=True).tag(sync=True)
    x_range = tl.List(allow_none=True).tag(sync=True)
    x2_range = tl.List(allow_none=True).tag(sync=True)
    y_range = tl.List(allow_none=True).tag(sync=True)
    y2_range = tl.List(allow_none=True).tag(sync=True)
    color_scale = tl.Dict(allow_none=True).tag(sync=True)
    size_scale = tl.Dict(allow_none=True).tag(sync=True)
    point_tween = tl.Dict(allow_none=True).tag(sync=True)

    def __init__(
        self,
        series: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        *,
        orientation: str = "vertical",
        mode: str = "overlay",
        x_axis: dict[str, Any] | None = None,
        y_axis: dict[str, Any] | None = None,
        y2_axis: dict[str, Any] | None = None,
        display: dict[str, Any] | None = None,
        legend: dict[str, Any] | None = None,
        bar: dict[str, Any] | None = None,
        line: dict[str, Any] | None = None,
        ref_lines: list[dict[str, Any]] | None = None,
        controls: dict[str, Any] | None = None,
        padding: dict[str, int] | None = None,
        range_padding: float | None = None,
        show_legend: bool | None = None,
        x2_axis: dict[str, Any] | None = None,
        x_range: list[float | None] | None = None,
        x2_range: list[float | None] | None = None,
        y_range: list[float | None] | None = None,
        y2_range: list[float | None] | None = None,
        color_scale: dict[str, Any] | None = None,
        size_scale: dict[str, Any] | None = None,
        point_tween: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a bar plot widget.

        Args:
            series: Plot series with required ``x`` and ``y`` arrays.
            orientation: Bar orientation (``vertical`` or ``horizontal``).
            mode: Bar mode (``overlay``, ``stacked``, or ``grouped``).
            x_axis: X axis configuration.
            y_axis: Primary Y axis configuration.
            y2_axis: Secondary Y axis configuration.
            display: Grid/zero-line display options.
            legend: Legend configuration.
            bar: Bar-style configuration.
            line: Line-style configuration for line-rendered series.
            ref_lines: Reference line definitions.
            controls: Control pane configuration.
            padding: Plot area padding ``{"t": N, "b": N, "l": N, "r": N}``
                in pixels.
            range_padding: Fraction of data range to add as padding.
            show_legend: Whether to show the legend.
            x2_axis: Secondary X axis configuration.
            x_range: Fixed X axis range ``[min, max]``.
            x2_range: Fixed secondary X axis range ``[min, max]``.
            y_range: Fixed Y axis range ``[min, max]``.
            y2_range: Fixed secondary Y axis range ``[min, max]``.
            color_scale: Color scaling configuration for bar colors.
            size_scale: Marker size scaling (for line markers on bars).
            point_tween: Point/marker animation configuration.
            **kwargs: Additional base widget keyword arguments.
        """
        super().__init__(
            widget_type="bar_plot",
            series=normalize_plot_series(series, component_name="BarPlot"),
            orientation=orientation,
            mode=mode,
            x_axis=normalize_plot_json(x_axis, "BarPlot.x_axis"),
            y_axis=normalize_plot_json(y_axis, "BarPlot.y_axis"),
            y2_axis=normalize_plot_json(y2_axis, "BarPlot.y2_axis"),
            display=normalize_plot_json(display, "BarPlot.display"),
            legend=normalize_plot_json(legend, "BarPlot.legend"),
            bar=normalize_plot_json(bar, "BarPlot.bar"),
            line=normalize_plot_json(line, "BarPlot.line"),
            ref_lines=normalize_plot_json(ref_lines, "BarPlot.ref_lines"),
            controls=normalize_plot_json(controls, "BarPlot.controls"),
            padding=normalize_plot_json(padding, "BarPlot.padding"),
            range_padding=range_padding,
            show_legend=show_legend,
            x2_axis=normalize_plot_json(x2_axis, "BarPlot.x2_axis"),
            x_range=x_range,
            x2_range=x2_range,
            y_range=y_range,
            y2_range=y2_range,
            color_scale=normalize_plot_json(color_scale, "BarPlot.color_scale"),
            size_scale=normalize_plot_json(size_scale, "BarPlot.size_scale"),
            point_tween=normalize_plot_json(point_tween, "BarPlot.point_tween"),
            **kwargs,
        )
