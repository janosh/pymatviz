"""Chemical potential diagram widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_plot_json
from pymatviz.widgets.matterviz import MatterVizWidget


class ChemPotDiagramWidget(MatterVizWidget):
    """MatterViz widget for chemical potential diagrams.

    Visualizes thermodynamic stability regions in chemical potential space
    from phase diagram entry data.

    Examples:
        >>> from pymatviz import ChemPotDiagramWidget
        >>> widget = ChemPotDiagramWidget(
        ...     entries=[
        ...         {"name": "Li2O", "energy": -14.3, "composition": {"Li": 2, "O": 1}},
        ...     ],
        ... )
    """

    entries = tl.List(allow_none=True).tag(sync=True)
    config = tl.Dict(allow_none=True).tag(sync=True)
    temperature = tl.Float(allow_none=True, default_value=None).tag(sync=True)

    def __init__(
        self,
        entries: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a chemical potential diagram widget.

        Args:
            entries: Phase data entries with ``name``, ``energy``, and
                ``composition`` fields.
            **kwargs: Additional widget properties including ``config`` (dict
                for diagram settings) and ``temperature`` (float, Kelvin).
        """
        super().__init__(
            widget_type="chem_pot_diagram",
            entries=normalize_plot_json(entries or [], "ChemPotDiagram.entries"),
            **kwargs,
        )
