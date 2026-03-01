"""Histogram visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json, normalize_plot_series
from pymatviz.widgets.matterviz import MatterVizWidget


class HistogramWidget(MatterVizWidget):
    """MatterViz widget wrapper for histogram visualizations."""

    series = tl.List(allow_none=True).tag(sync=True)
    bins = tl.Int(default_value=100).tag(sync=True)
    mode = tl.CaselessStrEnum(values=["single", "overlay"], default_value="single").tag(
        sync=True
    )
    selected_property = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    show_legend = tl.Bool(default_value=True).tag(sync=True)
    x_axis = tl.Dict(allow_none=True).tag(sync=True)
    y_axis = tl.Dict(allow_none=True).tag(sync=True)
    y2_axis = tl.Dict(allow_none=True).tag(sync=True)
    display = tl.Dict(allow_none=True).tag(sync=True)
    legend = tl.Dict(allow_none=True).tag(sync=True)
    bar = tl.Dict(allow_none=True).tag(sync=True)
    ref_lines = tl.List(allow_none=True).tag(sync=True)
    controls = tl.Dict(allow_none=True).tag(sync=True)

    def __init__(
        self,
        series: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        *,
        bins: int = 100,
        mode: str = "single",
        selected_property: str | None = None,
        show_legend: bool = True,
        x_axis: dict[str, Any] | None = None,
        y_axis: dict[str, Any] | None = None,
        y2_axis: dict[str, Any] | None = None,
        display: dict[str, Any] | None = None,
        legend: dict[str, Any] | None = None,
        bar: dict[str, Any] | None = None,
        ref_lines: list[dict[str, Any]] | None = None,
        controls: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a histogram widget.

        Args:
            series: Plot series with required ``x`` and ``y`` arrays (``y`` values are
                used for histogram binning by the frontend component).
            bins: Number of bins.
            mode: Histogram mode (``single`` or ``overlay``).
            selected_property: Active series label in single-mode controls.
            show_legend: Whether to render the legend.
            x_axis: X axis configuration.
            y_axis: Primary Y axis configuration.
            y2_axis: Secondary Y axis configuration.
            display: Grid/zero-line display options.
            legend: Legend configuration.
            bar: Bar-style configuration.
            ref_lines: Reference line definitions.
            controls: Control pane configuration.
            **kwargs: Additional base widget keyword arguments.
        """
        super().__init__(
            widget_type="histogram",
            series=normalize_plot_series(series, component_name="Histogram"),
            bins=bins,
            mode=mode,
            selected_property=selected_property,
            show_legend=show_legend,
            x_axis=normalize_plot_json(x_axis, "Histogram.x_axis"),
            y_axis=normalize_plot_json(y_axis, "Histogram.y_axis"),
            y2_axis=normalize_plot_json(y2_axis, "Histogram.y2_axis"),
            display=normalize_plot_json(display, "Histogram.display"),
            legend=normalize_plot_json(legend, "Histogram.legend"),
            bar=normalize_plot_json(bar, "Histogram.bar"),
            ref_lines=normalize_plot_json(ref_lines, "Histogram.ref_lines"),
            controls=normalize_plot_json(controls, "Histogram.controls"),
            **kwargs,
        )
