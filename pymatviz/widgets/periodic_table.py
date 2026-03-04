"""Interactive periodic table widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json
from pymatviz.widgets.matterviz import MatterVizWidget


class PeriodicTableWidget(MatterVizWidget):
    """MatterViz widget for interactive periodic table visualizations.

    Supports element heatmaps, custom color scales, labels, and click interaction.

    Examples:
        Heatmap by element counts:
        >>> from pymatviz import PeriodicTableWidget
        >>> widget = PeriodicTableWidget(
        ...     heatmap_values={"Fe": 42, "O": 100, "Li": 15},
        ...     color_scale="interpolateViridis",
        ... )
    """

    heatmap_values = tl.Any(allow_none=True).tag(sync=True)
    color_scale = tl.Unicode(default_value="interpolateViridis").tag(sync=True)
    color_scale_range = tl.List(allow_none=True).tag(sync=True)
    color_overrides = tl.Dict(allow_none=True).tag(sync=True)
    labels = tl.Dict(allow_none=True).tag(sync=True)
    log_scale = tl.Bool(default_value=False).tag(sync=True)
    show_color_bar = tl.Bool(default_value=True).tag(sync=True)
    gap = tl.Unicode(default_value="0.3cqw").tag(sync=True)
    missing_color = tl.Unicode(default_value="element-category").tag(sync=True)

    _DEFAULT_STYLE = "width: 100%; max-height: 400px; aspect-ratio: 18/10;"

    def __init__(
        self,
        heatmap_values: dict[str, float] | list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PeriodicTableWidget.

        Args:
            heatmap_values: Element-keyed dict or list of values to color-map
                onto the periodic table.
            **kwargs: Additional widget properties.
        """
        kwargs.setdefault("style", self._DEFAULT_STYLE)
        super().__init__(
            widget_type="periodic_table",
            heatmap_values=normalize_plot_json(
                heatmap_values, "PeriodicTable.heatmap_values"
            ),
            **kwargs,
        )
