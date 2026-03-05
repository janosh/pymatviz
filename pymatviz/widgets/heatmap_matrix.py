"""Heatmap matrix widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json
from pymatviz.widgets.matterviz import MatterVizWidget


def _normalize_axis_items(
    items: list[str | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert axis items to the ``{key, label}`` dicts expected by the JS component.

    Plain strings (and other non-dict scalars) are wrapped as
    ``{"key": s, "label": s}``. Dicts are passed through after JSON normalization.
    """
    return [
        normalize_plot_json(item, "HeatmapMatrix.axis_item")
        if isinstance(item, dict)
        else {"key": str(item), "label": str(item)}
        for item in items
    ]


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
    tile_size = tl.Unicode(default_value="50px").tag(sync=True)
    gap = tl.Unicode(default_value="0px").tag(sync=True)
    show_values = tl.Any(default_value=True).tag(sync=True)
    label_style = tl.Unicode(default_value="font-size: 14px;").tag(sync=True)

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
                Plain strings are auto-converted to ``{"key": s, "label": s}``.
            y_items: Row labels (strings or ``{key, label}`` dicts).
                Plain strings are auto-converted to ``{"key": s, "label": s}``.
            values: 2D list of values or nested dict keyed by item keys.
            **kwargs: Additional widget properties.
        """
        super().__init__(
            widget_type="heatmap_matrix",
            x_items=_normalize_axis_items(x_items or []),
            y_items=_normalize_axis_items(y_items or []),
            values=normalize_plot_json(values, "HeatmapMatrix.values"),
            **kwargs,
        )
