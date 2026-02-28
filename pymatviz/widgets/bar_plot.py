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
            **kwargs,
        )
