"""Helper functions for calculating coordination numbers."""

from __future__ import annotations

from collections import defaultdict
from inspect import isclass
from typing import TYPE_CHECKING, Literal

from pymatgen.analysis.local_env import NearNeighbors

from pymatviz.enums import LabelEnum


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from pymatgen.core import PeriodicSite, Structure


class CnSplitMode(LabelEnum):
    """How to split the coordination number histogram into subplots."""

    none = "none", "None"
    by_element = "by element", "By element"
    by_structure = "by structure", "By structure"
    by_structure_and_element = "by structure and element", "By structure and element"


def create_hover_text(
    *,
    struct_key: str,
    elem_symbol: str,
    cn: int,
    count: int,
    hover_data: dict[str, str],
    data: dict[str, Any],
    is_single_structure: bool,
) -> str:
    """Create hover text for a single bar in the histogram."""
    hover_text = f"Formula: {struct_key}<br>" if not is_single_structure else ""
    hover_text += f"Element: {elem_symbol}<br>" if elem_symbol else ""
    hover_text += f"Coordination number: {cn}<br>Count: {count}"

    if hover_data:
        hover_text += "<br>" + "<br>".join(
            f"{label}: {data['hover_data'][key][idx] if idx < len(data['hover_data'][key]) else 'N/A'}"  # noqa: E501
            for idx, (key, label) in enumerate(hover_data.items())
        )

    return hover_text


def normalize_get_neighbors(
    strategy: float | NearNeighbors | type[NearNeighbors],
) -> Callable[[PeriodicSite, Structure], list[dict[str, Any]]]:
    """Normalize get_neighbors function."""
    # Prepare the neighbor-finding strategy
    if isinstance(strategy, int | float):
        cutoff = strategy
        return lambda site, structure: structure.get_neighbors(site, r=cutoff)
    if isinstance(strategy, NearNeighbors):
        return lambda site, structure: strategy.get_nn_info(
            structure, structure.index(site)
        )

    if isclass(strategy) and issubclass(strategy, NearNeighbors):
        nn_instance = strategy()
        return lambda site, structure: nn_instance.get_nn_info(
            structure, structure.index(site)
        )
    raise TypeError(
        f"Invalid {strategy=}. Expected float, NearNeighbors instance, or "
        "NearNeighbors subclass."
    )


def calculate_average_cn(
    structure: Structure,
    element: str,
    get_neighbors: Callable[[PeriodicSite, Structure], list[dict[str, Any]]],
) -> float:
    """Calculate the average coordination number for a given element in a structure."""
    element_sites = [site for site in structure if site.specie.symbol == element]
    cn_sum = sum(len(get_neighbors(site, structure)) for site in element_sites)
    return cn_sum / len(element_sites) if element_sites else 0


def coordination_nums_in_structure(
    structure: Structure,
    strategy: float | NearNeighbors | type[NearNeighbors] = 3.0,
    group_by: Literal["element", "specie", "site"] = "element",
) -> dict[str, list[int]]:
    """Get coordination numbers (CN) for each element in a structure.

    Args:
        structure (Structure): A pymatgen Structure object
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

    # Store coordination numbers for each group
    cns: dict[str, list[int]] = defaultdict(list)

    # Calculate CNs for all sites in the structure
    for idx, site in enumerate(structure, start=1):
        key = {
            "element": site.specie.symbol,
            "site": str(idx),
            "specie": str(site.specie),
        }[group_by]
        cn = len(get_neighbors(site, structure))
        cns[key] += [cn]

    return cns
