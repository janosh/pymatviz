from __future__ import annotations

from typing import Any, Literal

import pytest

from pymatviz.coordination.helpers import coordination_nums_in_structure
from tests.conftest import cubic_structure


@pytest.mark.parametrize(
    ("strategy", "group_by", "match"),
    [("invalid", "element", "invalid"), (3.0, "specie", "Invalid group_by=")],
)
def test_coordination_nums_invalid_input(
    strategy: Any, group_by: Any, match: str
) -> None:
    """Reject unknown strategies and group_by modes."""
    struct = cubic_structure(4.0, ["Na"], [[0, 0, 0]])
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
    struct = cubic_structure(
        4.0,
        ["Na", "Na", "Cl", "Cl"],
        [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5]],
    )

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
    struct = cubic_structure(4.0, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert coordination_nums_in_structure(struct, strategy=strategy) == expected_cn
