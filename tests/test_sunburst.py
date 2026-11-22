"""Test sunburst plots for space groups and chemical systems."""

from __future__ import annotations

from typing import Any, Literal, get_args

import kemio
import numpy as np
import pandas as pd
import pytest
from kemio.options import CrystalSystem

import pymatviz as pmv
from pymatviz.sunburst import spacegroup_sunburst
from tests.conftest import SI_STRUCTS, rocksalt


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
    assert {root["id"] for root in widget.data} == set(get_args(CrystalSystem))
    leaves = spg_leaves(widget)
    assert len(leaves) == 230
    assert leaves["cubic/225"] == ("225", 1)
    assert spacegroup_sunburst([1], label_text="percent").label_text == "percent"


@pytest.mark.parametrize(
    ("invalid", "err", "match"),
    [
        *[(val, ValueError, "missing values") for val in (None, float("nan"), pd.NA)],
        *[(val, ValueError, r"must be in \[1, 230\]") for val in (0, 231, 2.5)],
        ("invalid", TypeError, "one kind"),
        (True, TypeError, "bool"),
    ],
)
def test_spacegroup_sunburst_invalid_data(
    invalid: Any, err: type[Exception], match: str
) -> None:
    """Reject missing, out-of-range, mixed-kind and bool groups instead of losing
    or mislabeling slices.
    """
    with pytest.raises(err, match=match):
        spacegroup_sunburst([1, invalid])


def test_spacegroup_sunburst_merges_spellings() -> None:
    """Symbol spellings and int/float numbers of one group share a slice labeled
    with the canonical symbol.
    """
    assert spg_leaves(spacegroup_sunburst(["Fm-3m", "F m -3 m", "P1"])) == {
        "cubic/Fm-3m": ("Fm-3m", 2),
        "triclinic/P1": ("P1", 1),
    }
    assert spg_leaves(spacegroup_sunburst([225, 225.0])) == {"cubic/225": ("225", 2)}


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
        (SI_STRUCTS, {"monoclinic/12": ("12", 1), "tetragonal/129": ("129", 1)}),
    ],
)
def test_spacegroup_sunburst_counts(data: Any, expected: dict[str, Any]) -> None:
    """Numbers, symbols, Series and structures are counted per space group."""
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


@pytest.mark.parametrize("pbc", [(False,) * 3, (True, True, False)])
def test_spacegroup_sunburst_rejects_non_periodic(
    structures: list[kemio.Structure], pbc: tuple[bool, bool, bool]
) -> None:
    """Structures not periodic along all three axes are rejected with how to fix it."""
    struct = kemio.Structure(np.eye(3) * 3, ["Si"], [[0, 0, 0]], pbc=pbc)
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
    """max_spacegroups limits space groups shown *per crystal system* (regression:
    the limit used to be applied globally, silently dropping entire crystal systems).
    """
    widget = pmv.spacegroup_sunburst(
        range(1, 8), max_spacegroups=max_spacegroups, overflow=overflow
    )
    space_groups = dict(spg_leaves(widget).values())
    assert len(space_groups) == expected_systems
    others = sorted(
        (label, val) for label, val in space_groups.items() if "more not shown" in label
    )
    assert others == sorted(expected_others or [])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_spacegroups": 0}, "at least 1 or None, got limit=0"),
        ({"max_spacegroups": -1}, "at least 1 or None, got limit=-1"),
        ({"max_spacegroups": 1, "overflow": "invalid"}, "Invalid overflow="),
    ],
)
def test_spacegroup_sunburst_invalid_limits(kwargs: dict[str, Any], match: str) -> None:
    """max_spacegroups below 1 (instead of silently meaning no limit) and unknown
    overflow modes raise.
    """
    with pytest.raises(ValueError, match=match):
        pmv.spacegroup_sunburst([1, 2, 3], **kwargs)


@pytest.mark.parametrize("normalize", [False, True])
def test_chem_env_sunburst_rock_salt(normalize: bool) -> None:
    """ChemEnv classifies all eight rocksalt sites as CN 6 octahedra (O:6)."""
    rock_salt = rocksalt()
    empty = kemio.Structure(rock_salt.lattice, [], np.zeros((0, 3)))
    widget = pmv.chem_env_hierarchy([empty, rock_salt], normalize=normalize)
    expected = 1 if normalize else len(rock_salt)
    assert widget.data == [
        dict(
            id="CN 6",
            label="CN 6",
            children=[dict(id="CN 6/O:6", label="O:6", value=expected)],
        )
    ]
