"""Sankey flow diagram widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._schema_traits import SankeyTraits
from pymatviz.widgets._traits import PlotControlsTraits, optional_trait, validator
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

    @validator("data")
    def _validate_data(self, proposal: dict[str, Any]) -> dict[str, Any] | None:
        """Require ``nodes`` and ``links`` lists (tuples are lists after JSON
        normalization).
        """
        data = proposal["value"]
        if data is not None and not all(
            isinstance(data.get(key), list) for key in ("nodes", "links")
        ):
            raise ValueError(
                f"Sankey data needs 'nodes' and 'links' lists, got keys={sorted(data)}"
            )
        return data

    _positional_traits = ("data",)
