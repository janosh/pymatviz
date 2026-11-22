"""Helper functions for calculating coordination numbers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

from pymatviz.enums import LabelEnum


if TYPE_CHECKING:
    from typing_extensions import TypeIs


# Neighbor strategies computed in Python by matscikit.coordination_numbers: a cutoff
# in Å, or pymatgen's CrystalNN() and VoronoiNN()
type CnStrategy = float | Literal["crystal_nn", "voronoi"]
# MatterViz bonding strategies, computed in the browser
BrowserCnStrategy = Literal["electroneg_ratio", "explicit_only", "pair_cutoffs"]


def is_browser_strategy(strategy: Any) -> TypeIs[BrowserCnStrategy]:
    """Whether strategy names a MatterViz bonding strategy (computed in the browser)."""
    return strategy in get_args(BrowserCnStrategy)


class CnSplitMode(LabelEnum):
    """How to split coordination number histograms into bar series. Values match
    MatterViz's CoordinationBarPlot ``split_mode``.
    """

    none = "none", "All sites"
    by_element = "by_element", "By element"
    by_structure = "by_structure", "By structure"
