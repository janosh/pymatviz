"""Scatter plot visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_json_kwargs
from pymatviz.widgets._schema_traits import ScatterPlotTraits
from pymatviz.widgets._traits import (
    CartesianPlotTraits,
    color_bar_trait,
    color_scale_trait,
    optional_trait,
)
from pymatviz.widgets.matterviz import MatterVizWidget


class ScatterPlotWidget(CartesianPlotTraits, ScatterPlotTraits, MatterVizWidget):
    """MatterViz widget wrapper for 2D scatter/line plots.

    The payload follows matterviz `ScatterPlot` props with a Python-friendly API.
    """

    color_scale = color_scale_trait()
    color_bar = color_bar_trait()
    # Size legend props; {} shows it when markers vary in size, None hides it
    size_legend = tl.Dict(default_value={}, allow_none=True).tag(sync=True)
    # Density contours per series: True for defaults or a dict of levels, bandwidth,
    # filled, fill_opacity, stroke_width
    density_contours = optional_trait(tl.Union, trait_types=[tl.Bool(), tl.Dict()])

    _SCATTER_JSON_KWARGS = (
        "styles", "color_scale", "size_scale", "fill_regions", "error_bands",
        "color_bar", "size_legend", "density_contours", "hover_config",
        "label_placement_config", "point_tween", "line_tween",
    )  # fmt: skip

    def __init__(
        self,
        series: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a scatter plot widget.

        Args:
            series: Plot series with required ``x`` and ``y`` arrays.
            **kwargs: Widget traits, e.g. axis/range configs (see
                ``CartesianPlotTraits``), ``styles``, ``color_scale``,
                ``fill_regions``, ``error_bands``, ``marker_renderer`` (``"auto"``,
                ``"svg"`` or ``"canvas"``) or ``selected_point``
                (``{"series_idx", "point_idx"}`` to highlight; observe
                ``active_point`` to react to clicks).
        """
        json_keys = (*self._JSON_KWARGS, *self._SCATTER_JSON_KWARGS)
        normalize_json_kwargs(kwargs, json_keys, "ScatterPlot")
        super().__init__(
            series=series,
            **kwargs,
        )

    @tl.validate("series")
    def _validate_series(self, proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Normalize series on init and reassignment, binary when large."""
        return self._binary_series(proposal["value"], "ScatterPlot")
