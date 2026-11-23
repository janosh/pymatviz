"""Binary phase diagram visualization widget for Jupyter notebooks."""

from __future__ import annotations

from pymatviz.widgets._schema_traits import PhaseDiagramTraits
from pymatviz.widgets.matterviz import MatterVizWidget


class PhaseDiagramWidget(PhaseDiagramTraits, MatterVizWidget):
    """MatterViz widget for visualizing isobaric binary phase diagrams.

    Renders temperature-composition phase diagrams with regions, boundaries,
    and special points (eutectic, peritectic, etc.).

    Note: This is NOT for pymatgen's PhaseDiagram class (use ConvexHullWidget for
    that). This widget renders binary phase diagrams with temperature-dependent
    phase regions.

    ``data`` is a MatterViz ``binary-phase-diagram/2`` dict, e.g. from kemio's
    ``CalphadSystem.binary_phase_diagram``.

    Examples:
        A hand-drawn sketch: liquid above a two-solid field:
        >>> import pymatviz as pmv
        >>> phases = [
        ...     {"id": id_, "model_phase": id_, "label": id_, "kind": kind}
        ...     for id_, kind in (("L", "liquid"), ("a", "solid"), ("b", "solid"))
        ... ]
        >>> liquid_box = [[0, 900], [1, 900], [1, 1500], [0, 1500]]
        >>> solid_box = [[0, 300], [1, 300], [1, 900], [0, 900]]
        >>> regions = [
        ...     {"phases": ["L"], "vertices": liquid_box},
        ...     {"phases": ["a", "b"], "vertices": solid_box},
        ... ]
        >>> empty_lists = ("boundaries", "invariants", "congruent_points")
        >>> empty_lists += ("critical_points", "unary_transitions")
        >>> empty_lists += ("polymorphic_transitions",)
        >>> data = {
        ...     "format": "binary-phase-diagram/2",
        ...     "components": ["A", "B"],
        ...     "pressure_gpa": 1e-4,
        ...     "temperature_range": [300, 1500],
        ...     "phases": phases,
        ...     "regions": regions,
        ...     "provenance": {"kind": "sketch"},
        ... } | dict.fromkeys(empty_lists, [])
        >>> widget = pmv.phase_diagram(data=data)
        >>> widget
    """

    _positional_traits = ("data",)
