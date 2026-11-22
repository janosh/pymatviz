"""Sankey flow diagram widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json
from pymatviz.widgets._schema_traits import SankeyTraits
from pymatviz.widgets._traits import PlotControlsTraits, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class SankeyWidget(PlotControlsTraits, SankeyTraits, MatterVizWidget):
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

    max_links = optional_trait(tl.Int, min=0)  # outgoing links per node (0 = all)
    iterations = optional_trait(tl.Int, min=0)  # d3-sankey relaxation passes

    @tl.validate("data")
    def _validate_data(self, proposal: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize numpy values (tuples to lists) and require ``nodes`` and
        ``links`` lists.
        """
        try:
            data = normalize_plot_json(proposal["value"], "Sankey.data")
        except (TypeError, ValueError) as exc:
            raise tl.TraitError(str(exc)) from exc
        if data is not None and not all(
            isinstance(data.get(key), list) for key in ("nodes", "links")
        ):
            raise tl.TraitError(
                f"Sankey data needs 'nodes' and 'links' lists, got keys={sorted(data)}"
            )
        return data

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize a Sankey widget.

        Args:
            data: ``{"nodes": [...], "links": [...]}`` flow data.
            **kwargs: Widget traits, e.g. ``orientation``, ``node_align``,
                ``link_color_mode``, ``value_format``.
        """
        super().__init__(data=data, **kwargs)
