"""Scatter plot visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._binary import series_column
from pymatviz.widgets._schema_traits import ScatterPlotTraits
from pymatviz.widgets._traits import (
    JsonDict,
    PlotControlsTraits,
    color_bar_trait,
    color_scale_trait,
    df_series,
    from_frontend,
    index_error,
    optional_trait,
    validator,
)
from pymatviz.widgets.matterviz import MatterVizWidget


_POINT_KEYS = ("series_idx", "point_idx")


def _n_points(series_entry: Any) -> int | None:
    """Point count of a series entry (JSON or binary), None when unknown (x missing
    or an iterator mid-batch, before the series validator ran).
    """
    if not isinstance(series_entry, dict):
        return None
    x_vals = series_column(series_entry, "x")
    return len(x_vals) if hasattr(x_vals, "__len__") else None


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

    def _point_error(self, point: dict[str, Any]) -> IndexError | None:
        """IndexError if ``point`` addresses no point of the current series."""
        series = self.series or []
        series_idx, point_idx = (point[key] for key in _POINT_KEYS)
        if not 0 <= series_idx < len(series):
            return index_error(
                self, "selected_point.series_idx", series_idx, len(series), "series"
            )
        n_points = _n_points(series[series_idx])
        if point_idx < 0 or (n_points is not None and point_idx >= n_points):
            return index_error(
                self, "selected_point.point_idx", point_idx, n_points, "points"
            )
        return None

    @validator("selected_point")
    def _check_selected_point(self, proposal: dict[str, Any]) -> Any:
        """Require int ``series_idx`` and ``point_idx`` addressing a point of the
        series. A frontend selection out of range resets to None.
        """
        if (point := proposal["value"]) is None:
            return None
        if bad := [
            key
            for key in _POINT_KEYS
            if isinstance(point.get(key), bool) or not isinstance(point.get(key), int)
        ]:
            raise TypeError(
                f"selected_point needs int {' and '.join(_POINT_KEYS)}, got {point}: "
                f"bad {bad}"
            )
        if (error := self._point_error(point)) is None:
            return point
        if from_frontend(self, "selected_point"):
            return None
        raise error

    @tl.observe("series")
    def _reset_selected_point(self, _change: dict[str, Any]) -> None:
        """Clear a selected point the new series no longer have."""
        if self.selected_point is not None and self._point_error(self.selected_point):
            self.selected_point = None

    @validator("series")
    def _validate_series(self, proposal: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Normalize series on init and reassignment, binary when large."""
        return self._normalize_series(proposal["value"], "ScatterPlot")
