"""Helper functions for calculating coordination numbers."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import matscikit as mk

from pymatviz.enums import LabelEnum
from pymatviz.structures import site_symbols, to_structure


if TYPE_CHECKING:
    from pymatviz.typing import AnyStructure

# Neighbor strategies computed in Python by matscikit.coordination_numbers, as
# pymatgen's CrystalNN() and VoronoiNN()
type CnStrategy = float | Literal["crystal_nn", "voronoi"]
PYTHON_CN_STRATEGIES = ("crystal_nn", "voronoi")


class CnSplitMode(LabelEnum):
    """How to split coordination number histograms into bar series. Values match
    MatterViz's CoordinationBarPlot ``split_mode``.
    """

    none = "none", "All sites"
    by_element = "by_element", "By element"
    by_structure = "by_structure", "By structure"


def coordination_nums_in_structure(
    structure: AnyStructure,
    strategy: CnStrategy = 3.0,
    group_by: Literal["element", "site"] = "element",
) -> dict[str, list[int]]:
    """Get coordination numbers (CN) for each element in a structure.

    Args:
        structure (AnyStructure): A structure or molecule (matscikit, pymatgen, ASE,
            phonopy).
        strategy (float | "crystal_nn" | "voronoi"): A cutoff radius in Å (neighbors
            as pymatgen's ``get_neighbors`` finds them), "crystal_nn" (pymatgen's
            CrystalNN) or "voronoi" (VoronoiNN). Defaults to 3.0 Å.
        group_by ("element" | "site"): Group CNs by element symbol (the
            highest-occupancy one for disordered sites) or by 1-based site index.

    Returns:
        dict[str, list[int]]: Map of group keys to lists of coordination numbers.
            E.g. {"Si": [4, 4, 4], "O": [2, 2, 2, 2, 2, 2]} for SiO2.

    Raises:
        ValueError: For an invalid group_by or strategy, or a structure the strategy
            can't handle (e.g. "crystal_nn" on disordered or non-periodic ones).
    """
    if group_by not in ("element", "site"):
        raise ValueError(f"Invalid {group_by=}, must be 'element' or 'site'")
    structure = to_structure(structure)
    keys = (
        site_symbols(structure)
        if group_by == "element"
        else [str(idx) for idx in range(1, len(structure) + 1)]
    )
    (site_cns,) = mk.coordination_numbers([structure], strategy)
    cns: dict[str, list[int]] = defaultdict(list)
    for key, cn in zip(keys, site_cns.tolist(), strict=True):
        cns[key] += [cn]
    return cns
