"""Sankey flow diagram widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_json_kwargs, normalize_plot_json
from pymatviz.widgets._traits import PlotControlsTraits, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class SankeyWidget(PlotControlsTraits, MatterVizWidget):
    """MatterViz widget for Sankey flow diagrams.

    ``data`` is ``{"nodes": [{"id", "label", "color"}, ...], "links": [{"source",
    "target", "value"}, ...]}`` where link ends are node ids or node indices.

    Examples:
        >>> from pymatviz.widgets import SankeyWidget
        >>> SankeyWidget(
        ...     {
        ...         "nodes": [{"label": "a"}, {"label": "b"}],
        ...         "links": [{"source": 0, "target": 1, "value": 3}],
        ...     }
        ... )
    """

    data = optional_trait(tl.Dict)
    orientation = optional_trait(tl.CaselessStrEnum, values=["horizontal", "vertical"])
    node_width = optional_trait(tl.Float)  # px thickness of node rectangles
    node_padding = optional_trait(tl.Float)  # px gap between nodes in a column
    node_align = optional_trait(
        tl.CaselessStrEnum, values=["left", "right", "center", "justify"]
    )
    # fold a node's outgoing links below this outflow fraction into "Other" (0 = off)
    min_fraction = optional_trait(tl.Float)
    max_links = optional_trait(tl.Int, min=0)  # outgoing links per node (0 = all)
    iterations = optional_trait(tl.Int, min=0)  # d3-sankey relaxation passes
    link_opacity = optional_trait(tl.Float)
    link_color_mode = optional_trait(
        tl.CaselessStrEnum, values=["source", "target", "gradient", "static"]
    )
    show_node_labels = optional_trait(tl.Bool)
    value_format = optional_trait(tl.Unicode)  # d3-format string for values
    padding = optional_trait(tl.Dict)  # {"t", "b", "l", "r"} in pixels
    legend = optional_trait(tl.Dict)
    show_legend = optional_trait(tl.Bool)
    fullscreen_toggle = optional_trait(tl.Bool)

    @tl.validate("data")
    def _validate_data(self, proposal: dict[str, Any]) -> dict[str, Any] | None:
        """Require ``nodes`` and ``links`` lists and normalize numpy values."""
        data = proposal["value"]
        if data is None:
            return None
        if not all(isinstance(data.get(key), list) for key in ("nodes", "links")):
            raise tl.TraitError(
                f"Sankey data needs 'nodes' and 'links' lists, got keys={sorted(data)}"
            )
        try:
            return normalize_plot_json(data, "Sankey.data")
        except (TypeError, ValueError) as exc:
            raise tl.TraitError(str(exc)) from exc

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize a Sankey widget.

        Args:
            data: ``{"nodes": [...], "links": [...]}`` flow data.
            **kwargs: Widget traits, e.g. ``orientation``, ``node_align``,
                ``link_color_mode``, ``value_format``.
        """
        normalize_json_kwargs(kwargs, ("padding", "legend"), "Sankey")
        super().__init__(widget_type="sankey", data=data, **kwargs)
