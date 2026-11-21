from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest
from pymatgen.analysis.local_env import CrystalNN, NearNeighbors, VoronoiNN
from pymatgen.core import Lattice, Molecule, Structure

from pymatviz.coordination.helpers import (
    coordination_nums_in_structure,
    site_coordination_numbers,
    site_neighbor_distances,
)
from pymatviz.structures import to_structure


@pytest.mark.parametrize("cutoff", [2.5, 3.5, 6.0])
@pytest.mark.parametrize("as_molecule", [False, True])
def test_site_neighbor_distances_match_pymatgen(
    cutoff: float, as_molecule: bool
) -> None:
    """Per-site neighbor distances equal pymatgen's get_neighbors, incl. molecules."""
    rng = np.random.default_rng(seed=0)
    struct = Structure(Lattice.cubic(6.0), ["Na", "Cl"] * 10, rng.uniform(size=(20, 3)))
    if as_molecule:
        struct = Molecule.from_sites(struct)  # ty: ignore[invalid-assignment]
    distances = site_neighbor_distances(to_structure(struct), cutoff)
    expected = [struct.get_neighbors(site, r=cutoff) for site in struct]  # ty: ignore[invalid-argument-type]
    assert len(distances) == len(expected)
    for dists, nns in zip(distances, expected, strict=True):
        np.testing.assert_allclose(
            dists, sorted(nn.nn_distance for nn in nns), rtol=0, atol=1e-12
        )
    assert sum(map(len, distances)) > 0


@pytest.mark.parametrize(
    "strategy",
    [pytest.param(VoronoiNN(), id="instance"), pytest.param(CrystalNN, id="class")],
)
def test_site_coordination_numbers_with_nn(
    strategy: NearNeighbors | type[NearNeighbors],
) -> None:
    """NearNeighbors instances and classes give one CN per site."""
    struct = Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    cns = site_coordination_numbers(to_structure(struct), strategy)
    assert len(cns) == len(struct)
    assert all(isinstance(cn, int) for cn in cns)


def test_coordination_nums_duplicate_sites() -> None:
    """NearNeighbors strategies get each site's own index, also for duplicate sites
    (structure.index(site) returned the first duplicate's index).
    """

    class IndexNN(NearNeighbors):
        """Toy strategy returning as many neighbors as the queried site index."""

        def get_nn_info(self, structure: Structure, n: int) -> list[dict]:  # noqa: ARG002
            """Return n placeholder neighbors for site n."""
            return [{}] * n

    struct = Structure(Lattice.cubic(4.0), ["Na"] * 3, [[0, 0, 0]] * 3)
    cns = coordination_nums_in_structure(struct, strategy=IndexNN, group_by="site")
    assert cns == {"1": [0], "2": [1], "3": [2]}


@pytest.mark.parametrize(
    ("strategy", "group_by", "match"),
    [("invalid", "element", "Invalid strategy="), (3.0, "specie", "Invalid group_by=")],
)
def test_coordination_nums_invalid_input(
    strategy: Any, group_by: Any, match: str
) -> None:
    """Reject unknown strategies and group_by modes."""
    struct = Structure(Lattice.cubic(4.0), ["Na"], [[0, 0, 0]])
    with pytest.raises((TypeError, ValueError), match=match):
        coordination_nums_in_structure(struct, strategy=strategy, group_by=group_by)


@pytest.mark.parametrize(
    ("group_by", "expected"),
    [
        ("element", {"Na": [6, 6], "Cl": [6, 6]}),
        ("site", {"1": [6], "2": [6], "3": [6], "4": [6]}),
    ],
)
def test_coordination_nums_in_structure_group_by(
    group_by: Literal["element", "site"], expected: dict[str, list[int]]
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
