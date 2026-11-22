"""Scatter plot visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._schema_traits import ScatterPlotTraits
from pymatviz.widgets._traits import (
    JsonDict,
    PlotControlsTraits,
    color_bar_trait,
    color_scale_trait,
    df_series,
    optional_trait,
    validator,
)
from pymatviz.widgets.matterviz import MatterVizWidget


class ScatterPlotWidget(PlotControlsTraits, ScatterPlotTraits, MatterVizWidget):
    """MatterViz widget wrapper for 2D scatter/line plots.

    The payload follows matterviz `ScatterPlot` props with a Python-friendly API.
    """

    color_scale = color_scale_trait()
    color_bar = color_bar_trait()
    # Size legend props; {} shows it when markers vary in size, None hides it
    size_legend = JsonDict(default_value={}, allow_none=True).tag(sync=True)
    # Density contours per series: True for defaults or a dict of levels, bandwidth,
    # filled, fill_opacity, stroke_width
    density_contours = optional_trait(tl.Union, trait_types=[tl.Bool(), tl.Dict()])

    def __init__(
        self,
        series: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any = None,
        *,
        x: str | None = None,
        y: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a scatter plot widget.

        Args:
            series: Plot series dicts with ``x`` and ``y`` arrays, or a DataFrame whose
                columns ``x``, ``y`` and ``color`` name.
            x: DataFrame column of the x values.
            y: DataFrame column of the y values.
            color: DataFrame column to split into one series per value.
            **kwargs: Widget traits, e.g. axis/range configs (``x_axis``
                dicts, ``[min, max]`` ``x_range`` lists), ``styles``, ``color_scale``,
                ``fill_regions``, ``error_bands``, ``marker_renderer`` (``"auto"``,
                ``"svg"`` or ``"canvas"``) or ``selected_point``
                (``{"series_idx", "point_idx"}`` to highlight; observe
                ``active_point`` to react to clicks).
        """
        series = df_series(series, color, kwargs, x=x, y=y)
        super().__init__(series=series, **kwargs)

    _raw_json_traits = frozenset({"series"})  # _normalize_series converts them

    @validator("series")
    def _validate_series(self, proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Normalize series on init and reassignment, binary when large."""
        return self._normalize_series(proposal["value"], "ScatterPlot")
