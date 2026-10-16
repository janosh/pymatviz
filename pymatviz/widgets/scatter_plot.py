"""Scatter plot visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_json_kwargs, normalize_plot_series
from pymatviz.widgets._traits import CartesianPlotTraits, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class ScatterPlotWidget(CartesianPlotTraits, MatterVizWidget):
    """MatterViz widget wrapper for 2D scatter/line plots.

    The payload follows matterviz `ScatterPlot` props with a Python-friendly API.
    """

    styles = optional_trait(tl.Dict)  # point/line style overrides
    color_scale = optional_trait(tl.Dict)
    size_scale = optional_trait(tl.Dict)
    fill_regions = optional_trait(tl.List)
    error_bands = optional_trait(tl.List)
    # auto picks canvas above the dense-point threshold, svg below
    marker_renderer = optional_trait(
        tl.CaselessStrEnum, values=["auto", "svg", "canvas"]
    )
    color_bar = optional_trait(tl.Dict)
    hover_config = optional_trait(tl.Dict)
    label_placement_config = optional_trait(tl.Dict)
    point_tween = optional_trait(tl.Dict)
    line_tween = optional_trait(tl.Dict)

    # Interaction state (two-way synced with the frontend for ipywidgets linking).
    # selected_point drives a highlight from Python and takes
    # {"series_idx", "point_idx"}. active_point/hovered_point are
    # populated on user click/hover (observe them to drive other widgets).
    # hovered_point is {"series_idx", "point_idx", "x", "y"} or None.
    # active_point is the last clicked point or None; it adds an
    # event_id (monotonically increasing per widget instance) so re-clicking
    # the same point is still observed as a distinct event:
    # {"series_idx", "point_idx", "x", "y", "event_id"}.
    selected_point = optional_trait(tl.Dict)
    active_point = optional_trait(tl.Dict)
    hovered_point = optional_trait(tl.Dict)

    _SCATTER_JSON_KWARGS = (
        "styles", "color_scale", "size_scale", "fill_regions", "error_bands",
        "color_bar", "hover_config", "label_placement_config", "point_tween",
        "line_tween",
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
            widget_type="scatter_plot",
            series=normalize_plot_series(series, component_name="ScatterPlot"),
            **kwargs,
        )
