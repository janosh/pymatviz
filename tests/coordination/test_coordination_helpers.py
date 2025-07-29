from __future__ import annotations

from typing import Literal

import pytest
from pymatgen.analysis.local_env import CrystalNN, NearNeighbors, VoronoiNN
from pymatgen.core import Lattice, Structure

from pymatviz.coordination.helpers import (
    CnSplitMode,
    calculate_average_cn,
    coordination_nums_in_structure,
    normalize_get_neighbors,
)
from pymatviz.enums import LabelEnum


def test_cn_split_mode_values() -> None:
    """Test that CnSplitMode enum has the expected values."""
    assert issubclass(CnSplitMode, LabelEnum)
    assert CnSplitMode.none.value == "none"
    assert CnSplitMode.by_element.value == "by element"
    assert CnSplitMode.by_structure.value == "by structure"
    assert CnSplitMode.by_structure_and_element.value == "by structure and element"


def test_normalize_get_neighbors_with_float() -> None:
    """Test normalize_get_neighbors with float cutoff."""
    # Create a simple cubic structure
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    get_neighbors = normalize_get_neighbors(strategy=3.5)
    neighbors = get_neighbors(struct[0], struct)

    assert len(neighbors) == 8  # Simple cubic should have 8 nearest neighbors


def test_normalize_get_neighbors_with_nn_instance() -> None:
    """Test normalize_get_neighbors with NearNeighbors instance."""
    struct = Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    voronoi = VoronoiNN()
    get_neighbors = normalize_get_neighbors(strategy=voronoi)
    neighbors = get_neighbors(struct[0], struct)

    assert isinstance(neighbors, list)
    assert all(isinstance(n, dict) for n in neighbors)


def test_normalize_get_neighbors_with_nn_class() -> None:
    """Test normalize_get_neighbors with NearNeighbors class."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    get_neighbors = normalize_get_neighbors(strategy=CrystalNN)
    neighbors = get_neighbors(struct[0], struct)

    assert isinstance(neighbors, list)
    assert all(isinstance(n, dict) for n in neighbors)


def test_normalize_get_neighbors_invalid_input() -> None:
    """Test normalize_get_neighbors with invalid input."""
    with pytest.raises(TypeError, match="Invalid strategy="):
        normalize_get_neighbors(strategy="invalid")


def test_calculate_average_cn() -> None:
    """Test calculate_average_cn function."""
    # Create a simple cubic structure where Na should have 6 neighbors
    lattice = Lattice.cubic(4.0)
    struct = Structure(
        lattice,
        ["Na", "Na", "Cl", "Cl"],
        [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5]],
    )

    get_neighbors = normalize_get_neighbors(strategy=3.0)
    avg_cn = calculate_average_cn(struct, "Na", get_neighbors)

    assert avg_cn == 6.0  # Each Na should have 6 neighbors in simple cubic


def test_calculate_average_cn_empty_element() -> None:
    """Test calculate_average_cn with non-existent element."""
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    get_neighbors = normalize_get_neighbors(strategy=3.0)
    avg_cn = calculate_average_cn(structure, "K", get_neighbors)

    assert avg_cn == 0  # No K atoms in structure


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
