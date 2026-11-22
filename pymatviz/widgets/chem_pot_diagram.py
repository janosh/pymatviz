"""Chemical potential diagram widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import (
    _normalize_entry_compositions,
    normalize_plot_json,
)
from pymatviz.widgets._schema_traits import ChemPotDiagramTraits
from pymatviz.widgets._traits import normalized_trait, optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


def _chem_pot_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """JSON entries with formula string compositions parsed to dicts."""
    normalized = _normalize_entry_compositions(entries)
    return normalize_plot_json(normalized, "ChemPotDiagram.entries")


class ChemPotDiagramWidget(ChemPotDiagramTraits, MatterVizWidget):
    """MatterViz widget for chemical potential diagrams.

    Visualizes thermodynamic stability regions in chemical potential space
    from phase diagram entry data.

    Examples:
        >>> import pymatviz as pmv
        >>> widget = pmv.chem_pot_diagram(
        ...     entries=[
        ...         {"name": "Li2O", "energy": -14.3, "composition": {"Li": 2, "O": 1}},
        ...     ],
        ... )
    """

    entries = normalized_trait(_chem_pot_entries, tl.List())  # on every assignment
    width = optional_trait(tl.Int, min=1)
    height = optional_trait(tl.Int, min=1)

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
                for diagram settings), ``temperature`` (float, Kelvin) and
                ``width``/``height`` (px, the aspect ratio the diagram keeps as it fills
                the available width).
        """
        super().__init__(entries=[] if entries is None else entries, **kwargs)
