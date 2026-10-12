from __future__ import annotations

from typing import Literal

import pytest
from pymatgen.analysis.local_env import CrystalNN, NearNeighbors, VoronoiNN
from pymatgen.core import Lattice, Structure

from pymatviz.coordination.helpers import (
    calculate_average_cn,
    coordination_nums_in_structure,
    normalize_get_neighbors,
)


def test_normalize_get_neighbors_with_float() -> None:
    """Test normalize_get_neighbors with float cutoff."""
    # Create a simple cubic structure
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    get_neighbors = normalize_get_neighbors(strategy=3.5)
    neighbors = get_neighbors(struct[0], struct)

    assert len(neighbors) == 8  # Simple cubic should have 8 nearest neighbors


@pytest.mark.parametrize(
    "strategy",
    [pytest.param(VoronoiNN(), id="instance"), pytest.param(CrystalNN, id="class")],
)
def test_normalize_get_neighbors_with_nn(
    strategy: NearNeighbors | type[NearNeighbors],
) -> None:
    """Normalize neighbor-finder instances and classes to neighbor dictionaries."""
    struct = Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    neighbors = normalize_get_neighbors(strategy)(struct[0], struct)
    assert isinstance(neighbors, list)
    assert all(isinstance(neighbor, dict) for neighbor in neighbors)


def test_normalize_get_neighbors_invalid_input() -> None:
    """Test normalize_get_neighbors with invalid input."""
    with pytest.raises(TypeError, match="Invalid strategy="):
        normalize_get_neighbors(strategy="invalid")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(("element", "expected_cn"), [("Na", 6.0), ("K", 0)])
def test_calculate_average_cn(element: str, expected_cn: float) -> None:
    """Average CN per element, 0 for elements absent from the structure."""
    struct = Structure(
        Lattice.cubic(4.0),
        ["Na", "Na", "Cl", "Cl"],
        [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5]],
    )
    get_neighbors = normalize_get_neighbors(strategy=3.0)
    assert calculate_average_cn(struct, element, get_neighbors) == expected_cn


@pytest.mark.parametrize(
    ("group_by", "expected"),
    [
        ("element", {"Na": [6, 6], "Cl": [6, 6]}),
        ("site", {"1": [6], "2": [6], "3": [6], "4": [6]}),
        ("specie", {"Na+": [6, 6], "Cl-": [6, 6]}),
    ],
)
def test_coordination_nums_in_structure_group_by(
    group_by: Literal["element", "site", "specie"], expected: dict[str, list[int]]
) -> None:
    """Test different group_by options."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(
        lattice,
        ["Na", "Na", "Cl", "Cl"],
        [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5]],
    ).add_oxidation_state_by_guess()

    cns = coordination_nums_in_structure(struct, strategy=3.0, group_by=group_by)
    assert cns == expected


@pytest.mark.parametrize(
    ("strategy", "expected_cn"),
    [
        (3.0, {"Na": [0], "Cl": [0]}),  # Simple distance cutoff
        (VoronoiNN(), {"Na": [14], "Cl": [14]}),  # VoronoiNN instance
        (CrystalNN(), {"Na": [0], "Cl": [0]}),  # CrystalNN instance
        (VoronoiNN, {"Na": [14], "Cl": [14]}),  # VoronoiNN class
    ],
)
def test_coordination_nums_in_structure_strategies(
    strategy: float | NearNeighbors, expected_cn: dict[str, list[int]]
) -> None:
    """Test different neighbor-finding strategies."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    cns = coordination_nums_in_structure(struct, strategy=strategy)
    assert cns == expected_cn
