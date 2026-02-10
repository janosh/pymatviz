"""Binary phase diagram visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets.matterviz import MatterVizWidget


class PhaseDiagramWidget(MatterVizWidget):
    """MatterViz widget for visualizing isobaric binary phase diagrams.

    Renders temperature-composition phase diagrams with regions, boundaries,
    and special points (eutectic, peritectic, etc.).

    Note: This is NOT for pymatgen's PhaseDiagram class (use ConvexHullWidget for
    that). This widget renders binary phase diagrams with temperature-dependent
    phase regions.

    Examples:
        From a phase diagram data dict:
        >>> from pymatviz import PhaseDiagramWidget
        >>> data = {
        ...     "components": ["Fe", "C"],
        ...     "temperature_range": [300, 2000],
        ...     "regions": [...],
        ...     "boundaries": [...],
        ... }
        >>> widget = PhaseDiagramWidget(data=data)
        >>> widget
    """

    data = tl.Dict(allow_none=True).tag(sync=True)

    def __init__(self, data: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize the PhaseDiagramWidget.

        Args:
            data: Phase diagram data dict with components, temperature_range,
                regions, and boundaries.
            **kwargs: Additional widget properties.
        """
        super().__init__(widget_type="phase_diagram", data=data, **kwargs)
