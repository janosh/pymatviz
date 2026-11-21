"""Helper functions for calculating coordination numbers."""

from __future__ import annotations

from collections import defaultdict
from inspect import isclass
from typing import TYPE_CHECKING, Any, Literal

from pymatviz.enums import LabelEnum
from pymatviz.structure import helpers


if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatgen.analysis.local_env import NearNeighbors
    from pymatgen.core import IMolecule, IStructure


class CnSplitMode(LabelEnum):
    """How to split coordination number histograms into bar series. Values match
    MatterViz's CoordinationBarPlot ``split_mode``.
    """

    none = "none", "All sites"
    by_element = "by_element", "By element"
    by_structure = "by_structure", "By structure"


def normalize_get_neighbors(
    strategy: float | NearNeighbors | type[NearNeighbors],
) -> Callable[[IStructure | IMolecule], list[list[Any]]]:
    """Normalize a neighbor-finding strategy to a function returning the neighbor
    list of every site in a structure or molecule (in site order).

    A float cutoff runs a single neighbor search for all sites of a periodic
    structure (molecules lack get_all_neighbors, so get per-site neighbors).
    """
    if isinstance(strategy, int | float):
        from pymatgen.core import IMolecule

        cutoff = float(strategy)

        def get_all_neighbors(structure: IStructure | IMolecule) -> list[list[Any]]:
            """Neighbors of each site within the cutoff."""
            if isinstance(structure, IMolecule):
                return [structure.get_neighbors(site, r=cutoff) for site in structure]
            return structure.get_all_neighbors(r=cutoff)

        return get_all_neighbors
    from pymatgen.analysis.local_env import NearNeighbors

    is_nn_class = isclass(strategy) and issubclass(strategy, NearNeighbors)
    nn_instance = strategy() if is_nn_class else strategy
    if isinstance(nn_instance, NearNeighbors):
        # index sites by position: structure.index(site) is O(N) per site and
        # returns the first of duplicate sites
        # NearNeighbors annotates Structure but also handles IStructure/molecules
        return lambda structure: [
            nn_instance.get_nn_info(structure, idx)  # ty: ignore[invalid-argument-type]
            for idx in range(len(structure))
        ]
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
    site_neighbors = normalize_get_neighbors(strategy=strategy)(structure)

    cns: dict[str, list[int]] = defaultdict(list)

    # Calculate CNs for all sites in the structure
    for idx, (site, neighbors) in enumerate(
        zip(structure, site_neighbors, strict=True), start=1
    ):
        site_species = helpers.get_site_species(site)
        key = {
            "element": helpers.get_site_symbol(site),
            "site": str(idx),
            "specie": str(site_species),  # Preserve oxidation state for specie mode
        }[group_by]
        cns[key] += [len(neighbors)]

    return cns
