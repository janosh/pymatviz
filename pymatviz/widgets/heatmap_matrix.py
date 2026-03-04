"""Heatmap matrix widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json
from pymatviz.widgets.matterviz import MatterVizWidget


class HeatmapMatrixWidget(MatterVizWidget):
    """MatterViz widget for 2D heatmap/matrix visualizations.

    Supports element-vs-element matrices, confusion matrices, and general
    labeled grids with color scaling.

    Examples:
        >>> from pymatviz import HeatmapMatrixWidget
        >>> widget = HeatmapMatrixWidget(
        ...     x_items=["Fe", "O", "Li"],
        ...     y_items=["Fe", "O", "Li"],
        ...     values=[[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]],
        ... )
    """

    x_items = tl.List().tag(sync=True)
    y_items = tl.List().tag(sync=True)
    values = tl.Any(allow_none=True).tag(sync=True)
    color_scale = tl.Unicode(default_value="interpolateViridis").tag(sync=True)
    color_scale_range = tl.List(allow_none=True).tag(sync=True)
    log_scale = tl.Bool(default_value=False).tag(sync=True)
    missing_color = tl.Unicode(default_value="transparent").tag(sync=True)
    x_axis = tl.Dict(allow_none=True).tag(sync=True)
    y_axis = tl.Dict(allow_none=True).tag(sync=True)
    tile_size = tl.Unicode(default_value="6px").tag(sync=True)
    gap = tl.Unicode(default_value="0px").tag(sync=True)
    show_values = tl.Any(default_value=False).tag(sync=True)

    def __init__(
        self,
        x_items: list[str | dict[str, Any]] | None = None,
        y_items: list[str | dict[str, Any]] | None = None,
        values: list[list[float]] | dict[str, dict[str, float]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a heatmap matrix widget.

        Args:
            x_items: Column labels (strings or ``{key, label}`` dicts).
            y_items: Row labels (strings or ``{key, label}`` dicts).
            values: 2D list of values or nested dict keyed by item keys.
            **kwargs: Additional widget properties.
        """
        super().__init__(
            widget_type="heatmap_matrix",
            x_items=normalize_plot_json(x_items or [], "HeatmapMatrix.x_items"),
            y_items=normalize_plot_json(y_items or [], "HeatmapMatrix.y_items"),
            values=normalize_plot_json(values, "HeatmapMatrix.values"),
            **kwargs,
        )
