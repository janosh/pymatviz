"""MatterViz Jupyter/Marimo notebook extension for visualizing crystal structures and MD
trajectories.
"""

from __future__ import annotations

from typing import Literal

from pymatviz.widgets.composition import CompositionWidget
from pymatviz.widgets.mime import register_matterviz_widgets
from pymatviz.widgets.structure import StructureWidget
from pymatviz.widgets.trajectory import TrajectoryWidget


MattervizElementColorSchemes = Literal[
    "Jmol", "Vesta", "Pastel", "Alloy", "Muted", "Dark Mode"
]

register_matterviz_widgets()  # Auto-register in all supported environments
