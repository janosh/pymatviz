from __future__ import annotations

import pytest
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.core import Lattice, Structure

from pymatviz.coordination.helpers import (
    CnSplitMode,
    calculate_average_cn,
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
