"""Convex hull phase stability visualization widget for Jupyter notebooks."""

from __future__ import annotations

from typing import Any

import traitlets as tl

from pymatviz.widgets._normalize import normalize_convex_hull_entries
from pymatviz.widgets._schema_traits import ConvexHullTraits
from pymatviz.widgets._traits import normalized_trait
from pymatviz.widgets.matterviz import MatterVizWidget


class ConvexHullWidget(ConvexHullTraits, MatterVizWidget):
    """MatterViz widget for visualizing convex hull phase stability diagrams.

    Supports binary, ternary, and quaternary convex hulls. Accepts a matscikit or
    pymatgen PhaseDiagram, pymatgen entries, or a list of entry dicts with
    composition and energy data.

    Examples:
        From a pymatgen PhaseDiagram:
        >>> import pymatviz as pmv
        >>> from pymatgen.analysis.phase_diagram import PhaseDiagram
        >>> pd = PhaseDiagram(entries)
        >>> widget = pmv.convex_hull(entries=pd)
        >>> widget

        From a list of entry dicts:
        >>> entries = [
        ...     {"composition": {"Li": 1}, "energy_per_atom": -1.5},
        ...     {"composition": {"Li": 1, "O": 2}, "energy_per_atom": -3.2},
        ... ]
        >>> widget = pmv.convex_hull(entries=entries)

        With custom visualization options:
        >>> widget = pmv.convex_hull(
        ...     entries=pd,
        ...     show_unstable_labels=True,
        ...     temperature=300,
        ...     style="height: 600px;",
        ... )
    """

    entries = normalized_trait(normalize_convex_hull_entries, tl.List())
    # Display options
    show_stable = tl.Bool(default_value=True).tag(sync=True)
    show_unstable = tl.Bool(default_value=True).tag(sync=True)
    show_hull_faces = tl.Bool(default_value=True).tag(sync=True)

    def __init__(self, entries: Any | None = None, **kwargs: Any) -> None:
        """Initialize the ConvexHullWidget.

        Args:
            entries: Convex hull data -- a matscikit or pymatgen PhaseDiagram,
                pymatgen entries, or a list of dicts with composition and energy keys.
            **kwargs: Additional widget properties.
        """
        super().__init__(entries=entries, **kwargs)
