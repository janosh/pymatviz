"""Binary phase diagram visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

from pymatviz.widgets._schema_traits import PhaseDiagramTraits
from pymatviz.widgets.matterviz import MatterVizWidget


class PhaseDiagramWidget(PhaseDiagramTraits, MatterVizWidget):
    """MatterViz widget for visualizing isobaric binary phase diagrams.

    Renders temperature-composition phase diagrams with regions, boundaries,
    and special points (eutectic, peritectic, etc.).

    Note: This is NOT for pymatgen's PhaseDiagram class (use ConvexHullWidget for
    that). This widget renders binary phase diagrams with temperature-dependent
    phase regions.

    Examples:
        From a phase diagram data dict:
        >>> import pymatviz as pmv
        >>> data = {
        ...     "components": ["Fe", "C"],
        ...     "temperature_range": [300, 2000],
        ...     "regions": [...],
        ...     "boundaries": [...],
        ... }
        >>> widget = pmv.phase_diagram(data=data)
        >>> widget
    """

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize the PhaseDiagramWidget.

        Args:
            data: Phase diagram data dict with components, temperature_range,
                regions, and boundaries.
            **kwargs: Additional widget properties.
        """
        super().__init__(data=data, **kwargs)
