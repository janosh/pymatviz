"""Convex hull phase stability visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._traits import optional_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class ConvexHullWidget(MatterVizWidget):
    """MatterViz widget for visualizing convex hull phase stability diagrams.

    Supports binary, ternary, and quaternary convex hulls. Accepts either a
    pymatgen PhaseDiagram object or a list of entry dicts with composition and
    energy data.

    Examples:
        From a pymatgen PhaseDiagram:
        >>> from pymatviz import ConvexHullWidget
        >>> from pymatgen.analysis.phase_diagram import PhaseDiagram
        >>> pd = PhaseDiagram(entries)
        >>> widget = ConvexHullWidget(entries=pd)
        >>> widget

        From a list of entry dicts:
        >>> entries = [
        ...     {"composition": {"Li": 1}, "energy_per_atom": -1.5},
        ...     {"composition": {"Li": 1, "O": 2}, "energy_per_atom": -3.2},
        ... ]
        >>> widget = ConvexHullWidget(entries=entries)

        With custom visualization options:
        >>> widget = ConvexHullWidget(
        ...     entries=pd,
        ...     show_unstable_labels=True,
        ...     temperature=300,
        ...     style="height: 600px;",
        ... )
    """

    entries = tl.List(allow_none=True).tag(sync=True)

    # Display options
    show_stable = tl.Bool(default_value=True).tag(sync=True)
    show_unstable = tl.Bool(default_value=True).tag(sync=True)
    show_hull_faces = tl.Bool(default_value=True).tag(sync=True)
    hull_face_opacity = optional_trait(tl.Float)
    show_stable_labels = optional_trait(tl.Bool)
    show_unstable_labels = optional_trait(tl.Bool)
    max_hull_dist_show_labels = optional_trait(tl.Float)
    max_hull_dist_show_phases = optional_trait(tl.Float)

    # Temperature
    temperature = optional_trait(tl.Float)

    def __init__(self, entries: Any | None = None, **kwargs: Any) -> None:
        """Initialize the ConvexHullWidget.

        Args:
            entries: Convex hull data -- a pymatgen PhaseDiagram, or a list of dicts
                with composition and energy keys.
            **kwargs: Additional widget properties.
        """
        from pymatviz.widgets._normalize import normalize_convex_hull_entries

        super().__init__(
            widget_type="convex_hull",
            entries=normalize_convex_hull_entries(entries),
            **kwargs,
        )
