"""Helper functions for calculating coordination numbers."""

from __future__ import annotations

from collections import defaultdict
from inspect import isclass
from typing import TYPE_CHECKING, Literal

from pymatgen.analysis.local_env import NearNeighbors

from pymatviz.enums import LabelEnum
from pymatviz.structure import helpers


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from pymatgen.core import IMolecule, IStructure, Site


class CnSplitMode(LabelEnum):
    """How to split coordination number histograms into bar series. Values match
    MatterViz's CoordinationBarPlot ``split_mode``.
    """

    none = "none", "All sites"
    by_element = "by_element", "By element"
    by_structure = "by_structure", "By structure"


def normalize_get_neighbors(
    strategy: float | NearNeighbors | type[NearNeighbors],
) -> Callable[[Site, Any], list[Any]]:
    """Normalize get_neighbors function."""
    if isinstance(strategy, int | float):
        return lambda site, structure: structure.get_neighbors(site, r=strategy)
    is_nn_class = isclass(strategy) and issubclass(strategy, NearNeighbors)
    nn_instance = strategy() if is_nn_class else strategy
    if isinstance(nn_instance, NearNeighbors):
        return lambda site, structure: nn_instance.get_nn_info(
            structure, structure.index(site)
        )
    raise TypeError(
        f"Invalid {strategy=}. Expected float, NearNeighbors instance, or "
        "NearNeighbors subclass."
    )


def coordination_nums_in_structure(
    structure: IStructure | IMolecule,
    strategy: float | NearNeighbors | type[NearNeighbors] = 3.0,
    group_by: Literal["element", "specie", "site"] = "element",
) -> dict[str, list[int]]:
    """Get coordination numbers (CN) for each element in a structure.

    Args:
        structure (IStructure | IMolecule): A pymatgen structure or molecule
        strategy (float | NearNeighbors | type[NearNeighbors]): Neighbor-finding
            strategy. Can be one of:
            - float: Cutoff distance for neighbor search in Angstroms
            - NearNeighbors: An instance of a NearNeighbors subclass
            - Type[NearNeighbors]: A NearNeighbors subclass (will be instantiated)
            Defaults to 3.0 (Angstroms cutoff)
        group_by ("element" | "specie" | "site"): How to group the coordination numbers.
            Can be one of:
            - "element": Group by element symbol
            - "site": Group by site
            - "specie": Group by specie

    Returns:
        dict[str, list[int]]: Map of element symbols to lists of coordination numbers.
            E.g. {"Si": [4, 4, 4], "O": [2, 2, 2, 2, 2, 2]} for SiO2. Each number
            represents the CN of one atom of that element.

    Example:
        >>> from pymatgen.core import Structure
        >>> structure = Structure.from_file("SiO2.cif")
        >>> cns = coordination_nums_in_structure(structure)
        >>> print(cns)
        {"Si": [4, 4, 4], "O": [2, 2, 2, 2, 2, 2]}
    """
    get_neighbors = normalize_get_neighbors(strategy=strategy)

    cns: dict[str, list[int]] = defaultdict(list)

    # Calculate CNs for all sites in the structure
    for idx, site in enumerate(structure, start=1):
        site_species = helpers.get_site_species(site)
        key = {
            "element": helpers.get_site_symbol(site),
            "site": str(idx),
            "specie": str(site_species),  # Preserve oxidation state for specie mode
        }[group_by]
        cn = len(get_neighbors(site, structure))
        cns[key] += [cn]

    return cns
