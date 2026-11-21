"""Helper functions for calculating coordination numbers."""

from __future__ import annotations

from collections import defaultdict
from inspect import isclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pymatviz.enums import LabelEnum
from pymatviz.structures import is_instance_of, site_symbols, to_pymatgen, to_structure


if TYPE_CHECKING:
    import matscikit as mk
    from pymatgen.analysis.local_env import NearNeighbors

    from pymatviz.typing import AnyStructure

# pymatgen's get_neighbor_list numerical_tol: neighbors up to r + tol count as within
# r, and pairs closer than tol (coincident atoms) are not neighbors
NEIGHBOR_TOL = 1e-8


class CnSplitMode(LabelEnum):
    """How to split coordination number histograms into bar series. Values match
    MatterViz's CoordinationBarPlot ``split_mode``.
    """

    none = "none", "All sites"
    by_element = "by_element", "By element"
    by_structure = "by_structure", "By structure"


def is_near_neighbors(strategy: Any) -> bool:
    """Whether strategy is a pymatgen NearNeighbors instance or subclass, without
    importing pymatgen.
    """
    # pymatgen moved local_env from pymatgen.analysis to pymatgen.core
    nn_classes = (
        "pymatgen.core.local_env.NearNeighbors",
        "pymatgen.analysis.local_env.NearNeighbors",
    )
    if isclass(strategy):
        return any(
            f"{cls.__module__}.{cls.__qualname__}" in nn_classes
            for cls in strategy.__mro__
        )
    return is_instance_of(strategy, *nn_classes)


def site_neighbor_distances(struct: mk.Structure, cutoff: float) -> list[np.ndarray]:
    """Sorted distances from each site to its neighbors within cutoff (Å), with
    pymatgen's tolerance: up to cutoff + NEIGHBOR_TOL, coincident atoms excluded.
    """
    centers, _, _, distances = struct.neighbor_list(cutoff + 2 * NEIGHBOR_TOL)
    keep = (distances > NEIGHBOR_TOL) & (distances <= cutoff + NEIGHBOR_TOL)
    centers, distances = centers[keep], distances[keep]
    order = np.lexsort((distances, centers))
    splits = np.cumsum(np.bincount(centers, minlength=len(struct)))[:-1]
    return np.split(distances[order], splits)


def site_coordination_numbers(
    struct: mk.Structure, strategy: float | NearNeighbors | type[NearNeighbors]
) -> list[int]:
    """Coordination number of each site in a structure.

    Args:
        struct (mk.Structure): Structure or molecule (periodic boundaries off).
        strategy (float | NearNeighbors | type[NearNeighbors]): A cutoff radius in Å,
            or a pymatgen NearNeighbors instance or subclass (instantiated), which
            needs the optional pymatgen dependency.

    Raises:
        TypeError: For any other strategy.
    """
    if isinstance(strategy, int | float) and not isinstance(strategy, bool):
        return [len(dists) for dists in site_neighbor_distances(struct, strategy)]
    if is_near_neighbors(strategy):
        nn_instance: Any = strategy() if isclass(strategy) else strategy
        pmg_struct = to_pymatgen(struct)
        return [
            len(nn_instance.get_nn_info(pmg_struct, idx)) for idx in range(len(struct))
        ]
    raise TypeError(
        f"Invalid {strategy=}. Expected float, NearNeighbors instance, or "
        "NearNeighbors subclass."
    )


def coordination_nums_in_structure(
    structure: AnyStructure,
    strategy: float | NearNeighbors | type[NearNeighbors] = 3.0,
    group_by: Literal["element", "site"] = "element",
) -> dict[str, list[int]]:
    """Get coordination numbers (CN) for each element in a structure.

    Args:
        structure (AnyStructure): A structure or molecule (matscikit, pymatgen, ASE,
            phonopy).
        strategy (float | NearNeighbors | type[NearNeighbors]): A cutoff radius in Å,
            or a pymatgen NearNeighbors instance or subclass. Defaults to 3.0 Å.
        group_by ("element" | "site"): Group CNs by element symbol (the
            highest-occupancy one for disordered sites) or by 1-based site index.

    Returns:
        dict[str, list[int]]: Map of group keys to lists of coordination numbers.
            E.g. {"Si": [4, 4, 4], "O": [2, 2, 2, 2, 2, 2]} for SiO2.
    """
    if group_by not in ("element", "site"):
        raise ValueError(f"Invalid {group_by=}, must be 'element' or 'site'")
    structure = to_structure(structure)
    keys = (
        site_symbols(structure)
        if group_by == "element"
        else [str(idx) for idx in range(1, len(structure) + 1)]
    )
    cns: dict[str, list[int]] = defaultdict(list)
    for key, cn in zip(
        keys, site_coordination_numbers(structure, strategy), strict=True
    ):
        cns[key] += [cn]
    return cns
