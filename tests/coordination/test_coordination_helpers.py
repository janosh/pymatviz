from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pytest
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.core import Lattice, Molecule, Structure

from pymatviz.coordination.helpers import coordination_nums_in_structure


@pytest.mark.parametrize(
    ("strategy", "as_molecule"),
    [
        *((cutoff, False) for cutoff in (2.5, 3.5, 6.0)),
        (3.5, True),
        ("crystal_nn", False),
        ("voronoi", False),
    ],
)
def test_coordination_nums_match_pymatgen(
    strategy: float | Literal["crystal_nn", "voronoi"], as_molecule: bool
) -> None:
    """Per-site CNs equal pymatgen's get_neighbors / CrystalNN / VoronoiNN."""
    rng = np.random.default_rng(seed=0)
    struct = Structure(Lattice.cubic(6.0), ["Na", "Cl"] * 10, rng.uniform(size=(20, 3)))
    system = Molecule.from_sites(struct) if as_molecule else struct
    cns = coordination_nums_in_structure(system, strategy=strategy, group_by="site")
    if isinstance(strategy, float):
        expected = [len(system.get_neighbors(site, r=strategy)) for site in system]  # ty: ignore[invalid-argument-type]
    else:
        nn = CrystalNN() if strategy == "crystal_nn" else VoronoiNN()
        expected = [len(nn.get_nn_info(struct, idx)) for idx in range(len(struct))]
    assert [cn for (cn,) in cns.values()] == expected
    assert sum(expected) > 0


@pytest.mark.parametrize(
    ("strategy", "group_by", "match"),
    [("invalid", "element", "invalid"), (3.0, "specie", "Invalid group_by=")],
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
        (3.0, {"Na": [0], "Cl": [0]}),
        ("voronoi", {"Na": [14], "Cl": [14]}),
        ("crystal_nn", {"Na": [0], "Cl": [0]}),
    ],
)
def test_coordination_nums_in_structure_strategies(
    strategy: float | Literal["crystal_nn", "voronoi"],
    expected_cn: dict[str, list[int]],
) -> None:
    """Cutoff, Voronoi and CrystalNN strategies on CsCl-type NaCl."""
    struct = Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert coordination_nums_in_structure(struct, strategy=strategy) == expected_cn
