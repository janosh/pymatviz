"""Test sunburst plots for space groups and chemical systems."""

from __future__ import annotations

import sys
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

import pymatviz as pmv
from pymatviz.sunburst import spacegroup_sunburst


@pytest.fixture
def rock_salt() -> Structure:
    """Conventional rocksalt NaCl cell with eight octahedrally coordinated sites."""
    return Structure.from_spacegroup(
        225, Lattice.cubic(5.64), ["Na+", "Cl-"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def spg_leaves(widget: pmv.SunburstWidget) -> dict[str, tuple[str, int]]:
    """Map leaf IDs to (label, value) of a crystal system/space group sunburst."""
    return {
        leaf["id"]: (leaf["label"], leaf["value"])
        for root in widget.data
        for leaf in root["children"]
    }


def test_spacegroup_sunburst_all_groups() -> None:
    """All 230 space groups nest under the 7 crystal systems with label+value text."""
    widget = spacegroup_sunburst(range(1, 231))
    assert isinstance(widget, pmv.SunburstWidget)
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
    structures: tuple[Structure, Structure],
) -> None:
    """Structure inputs are converted to space group counts via moyopy."""
    assert spg_leaves(spacegroup_sunburst(structures)) == {
        "monoclinic/12": ("12", 1),
        "tetragonal/129": ("129", 1),
    }


def test_spacegroup_sunburst_requires_moyopy(
    structures: list[Structure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report the missing required backend for Structure input."""
    monkeypatch.setitem(sys.modules, "moyopy", None)
    with pytest.raises(ModuleNotFoundError):
        spacegroup_sunburst(structures)


def test_spacegroup_sunburst_reports_moyopy_conversion_failure(
    structures: list[Structure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report index and type when moyopy cannot convert a structure."""

    def reject_structure(_struct: Any) -> int:
        """Simulate unsupported structure input."""
        raise TypeError("unsupported structure")

    monkeypatch.setattr(
        "pymatviz.sunburst.spacegroup.get_spacegroup_number", reject_structure
    )

    with pytest.raises(TypeError, match="structure at index 0 \\(Structure\\)"):
        spacegroup_sunburst(structures)


@pytest.mark.parametrize(
    ("max_slices", "max_slices_mode", "expected_systems", "expected_others"),
    [
        # Edge cases for max_slices
        (None, "other", 7, None),  # no limit
        (0, "other", 7, None),  # treated as None
        (-1, "other", 7, None),  # negative treated as None
        (10, "other", 7, None),  # greater than number of systems
        # Normal cases with "other" mode. max_slices applies per crystal system:
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
def test_spacegroup_sunburst_max_slices(
    max_slices: int | None,
    max_slices_mode: Literal["other", "drop"],
    expected_systems: int,
    expected_others: list[tuple[str, int]] | None,
) -> None:
    """Test spacegroup_sunburst with max_slices functionality and edge cases.

    max_slices limits space groups shown *per crystal system* (regression: the
    limit used to be applied globally, silently dropping entire crystal systems).
    """
    # Create a dataset with multiple space groups per crystal system
    spg_numbers = (1, 2, 3, 4, 5, 6, 7)

    widget = pmv.spacegroup_sunburst(
        spg_numbers, max_slices=max_slices, max_slices_mode=max_slices_mode
    )
    space_groups = dict(spg_leaves(widget).values())
    assert len(space_groups) == expected_systems
    others = sorted(
        (label, val) for label, val in space_groups.items() if "more not shown" in label
    )
    assert others == sorted(expected_others or [])


def test_spacegroup_sunburst_invalid_limit_mode() -> None:
    """Invalid max_slices_mode raises."""
    with pytest.raises(ValueError, match="Invalid max_slices_mode="):
        pmv.spacegroup_sunburst([1, 2, 3], max_slices=1, max_slices_mode="invalid")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("normalize", [False, True])
def test_chem_env_sunburst_rock_salt(rock_salt: Structure, normalize: bool) -> None:
    """CrystalNN classifies all eight rocksalt sites as CN 6 octahedra."""
    empty = Structure(rock_salt.lattice, [], [])
    widget = pmv.chem_env_sunburst([empty, rock_salt], normalize=normalize)
    expected = 1 if normalize else len(rock_salt)
    assert widget.data == [
        dict(
            id="CN 6",
            label="CN 6",
            children=[
                dict(id="CN 6/octahedral:6", label="octahedral:6", value=expected)
            ],
        )
    ]
