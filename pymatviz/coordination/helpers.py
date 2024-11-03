"""Visualizations of coordination numbers distributions."""

from __future__ import annotations

from inspect import isclass
from typing import TYPE_CHECKING

from pymatgen.analysis.local_env import NearNeighbors

from pymatviz.enums import LabelEnum


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from pymatgen.core import PeriodicSite, Structure


class CnSplitMode(LabelEnum):
    """How to split the coordination number histogram into subplots."""

    none = "none"
    by_element = "by element"
    by_structure = "by structure"
    by_structure_and_element = "by structure and element"


def create_hover_text(
    struct_key: str,
    elem_symbol: str,
    cn: int,
    count: int,
    hover_data: dict[str, str],
    data: dict[str, Any],
    is_single_structure: bool,  # noqa: FBT001
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
