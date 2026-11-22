"""Test sunburst plots for space groups and chemical systems."""

from __future__ import annotations

from typing import Any, Literal, get_args

import matscikit as mk
import numpy as np
import pandas as pd
import pytest

import pymatviz as pmv
from pymatviz.sunburst import spacegroup_sunburst
from tests.conftest import rocksalt


@pytest.fixture
def rock_salt() -> mk.Structure:
    """Conventional rocksalt NaCl cell with eight octahedrally coordinated sites."""
    return rocksalt()


def spg_leaves(widget: pmv.widgets.SunburstWidget) -> dict[str, tuple[str, int]]:
    """Map leaf IDs to (label, value) of a crystal system/space group sunburst."""
    return {
        leaf["id"]: (leaf["label"], leaf["value"])
        for root in widget.data
        for leaf in root["children"]
    }


def test_spacegroup_sunburst_all_groups() -> None:
    """All 230 space groups nest under the 7 crystal systems with label+value text."""
    widget = spacegroup_sunburst(range(1, 231))
    assert isinstance(widget, pmv.widgets.SunburstWidget)
    assert widget.label_text == "label+value"
    assert {root["id"] for root in widget.data} == set(
        get_args(pmv.typing.CrystalSystem)
    )
    leaves = spg_leaves(widget)
    assert len(leaves) == 230
    assert leaves["cubic/225"] == ("225", 1)
    assert spacegroup_sunburst([1], label_text="percent").label_text == "percent"


@pytest.mark.parametrize("invalid", [None, float("nan"), pd.NA, 0, 231, 2.5, "invalid"])
def test_spacegroup_sunburst_invalid_data(invalid: Any) -> None:
    """Reject missing and invalid groups instead of losing or mislabeling slices."""
    with pytest.raises(ValueError, match=r"missing values|Invalid space group"):
        spacegroup_sunburst([1, invalid])


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([1], {"triclinic/1": ("1", 1)}),
        (np.array([225, 225, 1]), {"cubic/225": ("225", 2), "triclinic/1": ("1", 1)}),
        (["P1"], {"triclinic/P1": ("P1", 1)}),
        (
            pd.Series([*[1] * 3, *[2] * 10, *[3] * 5]),
            {
                "triclinic/2": ("2", 10),
                "monoclinic/3": ("3", 5),
                "triclinic/1": ("1", 3),
            },
        ),
    ],
)
def test_spacegroup_sunburst_counts(data: Any, expected: dict[str, Any]) -> None:
    """Numbers, symbols and Series are counted per space group."""
    assert spg_leaves(spacegroup_sunburst(data)) == expected


def test_spacegroup_sunburst_symbol_counts(spg_symbols: list[str]) -> None:
    """Space-group symbols are grouped into crystal systems with counts."""
    widget = spacegroup_sunburst(spg_symbols)
    totals = {
        root["id"]: sum(leaf["value"] for leaf in root["children"])
        for root in widget.data
    }
    assert (totals["monoclinic"], totals["cubic"], totals["hexagonal"]) == (4, 5, 3)
    leaves = spg_leaves(widget)
    assert leaves["monoclinic/C2/m"][0] == "C2/m"
    assert leaves["cubic/P-43m"][0] == "P-43m"


def test_spacegroup_sunburst_structure_counts(
    structures: tuple[mk.Structure, mk.Structure],
) -> None:
    """Structure inputs are converted to space group counts via moyopy."""
    assert spg_leaves(spacegroup_sunburst(structures)) == {
        "monoclinic/12": ("12", 1),
        "tetragonal/129": ("129", 1),
    }


@pytest.mark.parametrize("pbc", [(False,) * 3, (True, True, False)])
def test_spacegroup_sunburst_rejects_non_periodic(
    structures: list[mk.Structure], pbc: tuple[bool, bool, bool]
) -> None:
    """Structures not periodic along all three axes are rejected with how to fix it."""
    struct = mk.Structure(np.eye(3) * 3, ["Si"], [[0, 0, 0]], pbc=pbc)
    with pytest.raises(
        TypeError, match=r"periodic along a, b and c.*atoms\.pbc = True"
    ):
        spacegroup_sunburst([*structures, struct])
    assert spacegroup_sunburst([*structures, struct.replace(pbc=(True,) * 3)])


@pytest.mark.parametrize(
    ("max_spacegroups", "overflow", "expected_systems", "expected_others"),
    [
        # Edge cases for max_spacegroups
        (None, "other", 7, None),  # no limit
        (10, "other", 7, None),  # greater than number of systems
        # Normal cases with "other" mode. max_spacegroups applies per crystal system:
        # spg 1-2 are triclinic (2 entries), spg 3-7 monoclinic (5 entries)
        (
            1,
            "other",
            4,
            [("Other (1 more not shown)", 1), ("Other (4 more not shown)", 4)],
        ),
        (2, "other", 5, [("Other (3 more not shown)", 3)]),  # triclinic fits fully
        (3, "other", 6, [("Other (2 more not shown)", 2)]),
        # Cases with "drop" mode (per crystal system)
        (1, "drop", 2, None),  # top 1 per crystal system
        (2, "drop", 4, None),  # top 2 per crystal system
        (3, "drop", 5, None),  # top 3 monoclinic + both triclinic
    ],
)
def test_spacegroup_sunburst_max_spacegroups(
    max_spacegroups: int | None,
    overflow: Literal["other", "drop"],
    expected_systems: int,
    expected_others: list[tuple[str, int]] | None,
) -> None:
    """Test spacegroup_sunburst with max_spacegroups functionality and edge cases.

    max_spacegroups limits space groups shown *per crystal system* (regression: the
    limit used to be applied globally, silently dropping entire crystal systems).
    """
    # Create a dataset with multiple space groups per crystal system
    spg_numbers = (1, 2, 3, 4, 5, 6, 7)

    widget = pmv.spacegroup_sunburst(
        spg_numbers, max_spacegroups=max_spacegroups, overflow=overflow
    )
    space_groups = dict(spg_leaves(widget).values())
    assert len(space_groups) == expected_systems
    others = sorted(
        (label, val) for label, val in space_groups.items() if "more not shown" in label
    )
    assert others == sorted(expected_others or [])


@pytest.mark.parametrize("max_spacegroups", [0, -1])
def test_spacegroup_sunburst_rejects_limit_below_one(max_spacegroups: int) -> None:
    """max_spacegroups below 1 raises instead of silently meaning no limit."""
    with pytest.raises(
        ValueError, match=f"at least 1 or None, got limit={max_spacegroups}"
    ):
        pmv.spacegroup_sunburst([1, 2], max_spacegroups=max_spacegroups)


def test_spacegroup_sunburst_invalid_limit_mode() -> None:
    """Invalid overflow raises."""
    with pytest.raises(ValueError, match="Invalid overflow="):
        pmv.spacegroup_sunburst([1, 2, 3], max_spacegroups=1, overflow="invalid")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("normalize", [False, True])
def test_chem_env_sunburst_rock_salt(rock_salt: mk.Structure, normalize: bool) -> None:
    """ChemEnv classifies all eight rocksalt sites as CN 6 octahedra (O:6)."""
    empty = mk.Structure(rock_salt.lattice, [], np.zeros((0, 3)))
    widget = pmv.chem_env_hierarchy([empty, rock_salt], normalize=normalize)
    expected = 1 if normalize else len(rock_salt)
    assert widget.data == [
        dict(
            id="CN 6",
            label="CN 6",
            children=[dict(id="CN 6/O:6", label="O:6", value=expected)],
        )
    ]
