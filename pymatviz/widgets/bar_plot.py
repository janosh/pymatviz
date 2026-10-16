"""Bar plot visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_json_kwargs, normalize_plot_series
from pymatviz.widgets._traits import CartesianPlotTraits, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class BarPlotWidget(CartesianPlotTraits, MatterVizWidget):
    """MatterViz widget wrapper for grouped/stacked/overlay bar plots."""

    orientation = tl.CaselessStrEnum(
        values=["vertical", "horizontal"], default_value="vertical"
    ).tag(sync=True)
    mode = tl.CaselessStrEnum(
        values=["overlay", "stacked", "grouped"], default_value="overlay"
    ).tag(sync=True)
    bar = optional_trait(tl.Dict)  # bar-style config
    line = optional_trait(tl.Dict)  # style of line-rendered series
    color_scale = optional_trait(tl.Dict)
    size_scale = optional_trait(tl.Dict)  # line-marker size scaling

    def __init__(
        self,
        series: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a bar plot widget.

        Args:
            series: Plot series with required ``x`` and ``y`` arrays.
            **kwargs: Widget traits, e.g. ``orientation``, ``mode`` (``overlay``,
                ``stacked`` or ``grouped``), axis/range configs (see
                ``CartesianPlotTraits``), ``bar``, ``line``, ``color_scale``.
        """
        json_keys = (*self._JSON_KWARGS, "bar", "line", "color_scale", "size_scale")
        normalize_json_kwargs(kwargs, json_keys, "BarPlot")
        super().__init__(
            widget_type="bar_plot",
            series=normalize_plot_series(series, component_name="BarPlot"),
            **kwargs,
        )
