"""Test sunburst plots for space groups and chemical systems."""

from __future__ import annotations

import sys
from typing import Any, Literal, get_args

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

import pymatviz as pmv
from pymatviz.sunburst import ShowCounts, spacegroup_sunburst


@pytest.fixture
def rock_salt() -> Structure:
    """Conventional rocksalt NaCl cell with eight octahedrally coordinated sites."""
    return Structure.from_spacegroup(
        225, Lattice.cubic(5.64), ["Na+", "Cl-"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


@pytest.mark.parametrize("show_counts", get_args(ShowCounts))
def test_spacegroup_sunburst(show_counts: ShowCounts) -> None:
    """Test spacegroup_sunburst with different show_counts options."""
    fig = spacegroup_sunburst(range(1, 231), show_counts=show_counts)

    assert set(fig.data[0].parents) == {"", *get_args(pmv.typing.CrystalSystem)}
    assert fig.data[0].branchvalues == "total"

    if show_counts == "value":
        assert fig.data[0].texttemplate is not None
        assert "N=" in fig.data[0].texttemplate
    elif show_counts == "value+percent":
        assert fig.data[0].texttemplate is not None
        assert "N=" in fig.data[0].texttemplate
        assert "percentEntry" in fig.data[0].texttemplate
    elif show_counts == "percent":
        assert fig.data[0].textinfo is not None
        assert "percent" in fig.data[0].textinfo
    elif show_counts is False:
        assert fig.data[0].textinfo is None


def test_spacegroup_sunburst_invalid_show_counts() -> None:
    """Test that invalid show_counts values raise ValueError."""
    with pytest.raises(ValueError, match=r"Invalid.*show_counts"):
        spacegroup_sunburst([1], show_counts="invalid")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("invalid", [None, float("nan"), pd.NA, 0, 231, 2.5, "invalid"])
def test_spacegroup_sunburst_invalid_data(invalid: Any) -> None:
    """Reject missing and invalid groups instead of losing or mislabeling slices."""
    with pytest.raises(ValueError, match=r"missing values|Invalid space group"):
        spacegroup_sunburst([1, invalid])


def test_spacegroup_sunburst_single_item() -> None:
    """Test with single-item input."""
    fig = spacegroup_sunburst([1], show_counts="value")
    assert len(fig.data[0].ids) == 2  # one for crystal system, one for spg number

    fig = spacegroup_sunburst(["P1"], show_counts="value+percent")
    assert len(fig.data[0].ids) == 2


def test_spacegroup_sunburst_series_counts() -> None:
    """Test with other types of input."""
    series = pd.Series([*[1] * 3, *[2] * 10, *[3] * 5])
    fig = spacegroup_sunburst(series, show_counts="value")
    values = [*map(int, fig.data[0].values)]
    assert values == [10, 5, 3, 13, 5], f"actual {values=}"


def test_spacegroup_sunburst_symbol_counts(spg_symbols: list[str]) -> None:
    """Space-group symbols are grouped into crystal systems with counts."""
    fig = spacegroup_sunburst(spg_symbols, show_counts="value")
    trace = fig.data[0]
    values_by_label = dict(zip(trace.labels, map(int, trace.values), strict=True))
    parents_by_label = dict(zip(trace.labels, trace.parents, strict=True))

    assert values_by_label["monoclinic"] == 4
    assert values_by_label["cubic"] == 5
    assert values_by_label["hexagonal"] == 3
    assert parents_by_label["C2/m"] == "monoclinic"
    assert parents_by_label["P-43m"] == "cubic"


def test_spacegroup_sunburst_structure_counts(
    structures: tuple[Structure, Structure],
) -> None:
    """Structure inputs are converted to space group counts via moyopy."""
    fig = spacegroup_sunburst(structures, show_counts="value")
    trace = fig.data[0]

    assert list(trace.labels) == ["12", "129", "monoclinic", "tetragonal"]
    assert list(trace.parents) == ["monoclinic", "tetragonal", "", ""]
    assert [*map(int, trace.values)] == [1, 1, 1, 1]


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

    fig = pmv.spacegroup_sunburst(
        spg_numbers, max_slices=max_slices, max_slices_mode=max_slices_mode
    )

    # Get all space group entries (excluding crystal system level)
    space_groups = {
        label: val
        for idx, (label, val) in enumerate(
            zip(fig.data[0].labels, fig.data[0].values, strict=True)
        )
        if fig.data[0].parents[idx] != ""  # exclude root level
    }
    assert len(space_groups) == expected_systems

    # Check for expected "Other" entries
    if expected_others:
        other_entries = [
            (label, val)
            for label, val in space_groups.items()
            if "more not shown" in str(label)
        ]
        assert len(other_entries) == len(expected_others)

        for expected_label, expected_count in expected_others:
            found = any(
                expected_label in label and val == expected_count
                for label, val in other_entries
            )
            assert found, (
                f"{expected_label=} with {expected_count=} not in {other_entries=}"
            )
    else:
        assert not any("Other" in str(label) for label in space_groups)


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
