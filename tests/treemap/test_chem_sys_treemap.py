"""Test treemap plots for chemical systems."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, get_args

import plotly.graph_objects as go
import pytest
from pymatgen.core import Composition

import pymatviz as pmv
from pymatviz.typing import FormulaGroupBy, ShowCounts


if TYPE_CHECKING:
    from pymatgen.core import Structure

    from pymatviz.typing import FormulaGroupBy


# Type alias for arity formatter function
ArityFormatter = Callable[[str, int, int], str]

# Test data fixtures
TEST_SYSTEMS: list[str] = [
    "Fe2O3",
    "FeO",
    "Li2O",
    "Na2O",
    "K2O",
    "CaO",
    "MgO",
    "Al2O3",
    "SiO2",
    "TiO2",
    "LiFeO2",
    "NaFeO2",
    "KFeO2",
    "CaFeO3",
    "MgFeO3",
]

HIGH_ARITY_SYSTEMS: list[str] = [
    "Zr42.5Ga2.5Cu55",
    "Zr37.5Ga5Cu57.5",
    "Zr40Ga7.5Cu52.5",
    "Zr40Ga5Cu55",
    "Zr42.5Ga5Cu52.5",
    "Zr48Al8Cu32Ag8Pd4",
    "Zr48Al8Cu30Ag8Pd6",
    "Zr48Al8Cu34Ni2Ag8",
    "Zr48Al8Cu32Ni4Ag8",
    "Zr48Al8Cu30Ni6Ag8",
]

GROUPING_TEST_SYSTEMS: list[str] = [
    "Fe2O3",
    "Fe4O6",
    "FeO",
    "Li2O",
    "Li2O",
    "LiFeO2",
    "Li3FeO3",
]


def custom_formatter(arity: str, count: int, _total: int) -> str:
    """Format arity name with count in square brackets."""
    return f"{arity} [{count}]"


def test_chem_sys_treemap_basic() -> None:
    """Test basic functionality with mixed arity systems."""
    systems = ["Fe-O", "Li-P-O", "Fe", "O", "Li-O"]
    fig = pmv.chem_sys_treemap(systems)

    # Verify figure structure
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "treemap"

    # Extract data for testing
    labels = fig.data[0].labels
    parents = fig.data[0].parents
    values = fig.data[0].values

    # Verify treemap structure
    assert "" in parents, "Root node missing"
    assert len(labels) == len(parents) == len(values), "Mismatched data lengths"

    # Check arity levels
    arity_levels = {
        label.split(" (")[0]
        for label in labels
        if any(x in label for x in ("unary", "binary", "ternary"))
    }
    assert arity_levels == {"unary", "binary", "ternary"}

    # Check elements and systems at each level
    unary_elements = {
        label
        for label, parent in zip(labels, parents, strict=False)
        if "unary" in parent
    }
    binary_systems = {
        label
        for label, parent in zip(labels, parents, strict=False)
        if "binary" in parent
    }
    ternary_systems = {
        label
        for label, parent in zip(labels, parents, strict=False)
        if "ternary" in parent
    }

    assert unary_elements == {"Fe", "O"}
    assert binary_systems == {"Fe-O", "Li-O"}
    assert ternary_systems == {"Li-O-P"}

    # Verify hover template
    assert "hovertemplate" in fig.data[0]
    assert "Count:" in fig.data[0].hovertemplate
    assert "percentEntry" in fig.data[0].hovertemplate


@pytest.mark.parametrize(
    ("systems", "expected_arity_levels", "test_id"),
    [
        (["Fe", "O", "Li-Fe-O"], {"unary", "ternary"}, "skip_binary"),
        (["Fe2O3", "FeO", "Li2O", "LiFeO2"], {"binary", "ternary"}, "skip_unary"),
        (TEST_SYSTEMS, {"binary", "ternary"}, "regular_systems"),
        (HIGH_ARITY_SYSTEMS, {"ternary", "quinary"}, "high_arity"),
    ],
)
def test_chem_sys_treemap_arity_scenarios(
    systems: list[str], expected_arity_levels: set[str], test_id: str
) -> None:
    """Test different arity scenarios."""
    fig = pmv.chem_sys_treemap(systems)
    labels = fig.data[0].labels

    # Check arity levels
    arity_levels = {
        label.split(" (")[0]
        for label in labels
        if any(x in label for x in ("unary", "binary", "ternary", "quinary"))
    }
    assert arity_levels == expected_arity_levels, f"Test {test_id} failed"

    # Apply customizations for one test case
    if test_id == "skip_binary":
        fig.update_traces(marker=dict(cornerradius=5))
        assert fig.data[0].marker.cornerradius == 5


@pytest.mark.parametrize("show_counts", get_args(ShowCounts))
def test_chem_sys_treemap_show_counts(show_counts: ShowCounts) -> None:
    """Test different show_counts options."""
    systems = ["Fe2O3", "FeO", "Li2O", "LiFeO2"]
    fig = pmv.chem_sys_treemap(systems, show_counts=show_counts)

    if show_counts == "value":
        assert "texttemplate" in fig.data[0]
        assert "N=" in fig.data[0].texttemplate
        assert "percentEntry" not in fig.data[0].texttemplate
    elif show_counts == "value+percent":
        assert "texttemplate" in fig.data[0]
        assert "N=" in fig.data[0].texttemplate
        assert "percentEntry" in fig.data[0].texttemplate
    elif show_counts == "percent":
        assert "textinfo" in fig.data[0]
        assert "percent" in fig.data[0].textinfo
    elif show_counts is False:
        assert fig.data[0].textinfo is None


@pytest.mark.parametrize(
    ("show_arity_counts", "expected_in_labels", "expected_not_in_labels"),
    [
        (custom_formatter, ["[3]", "[1]"], []),
        (False, [], ["N=", "%"]),
        (True, ["N="], []),  # Default behavior
    ],
)
def test_chem_sys_treemap_arity_formatting(
    show_arity_counts: ArityFormatter | bool,
    expected_in_labels: list[str],
    expected_not_in_labels: list[str],
) -> None:
    """Test arity formatting options."""
    systems = ["Fe2O3", "FeO", "Li2O", "LiFeO2"]
    fig = pmv.chem_sys_treemap(systems, show_arity_counts=show_arity_counts)
    labels = fig.data[0].labels
    arity_labels = {
        label for label in labels if any(x in label for x in ("binary", "ternary"))
    }

    for expected in expected_in_labels:
        assert any(expected in label for label in arity_labels)

    for not_expected in expected_not_in_labels:
        assert not any(not_expected in label for label in arity_labels)


def test_chem_sys_treemap_input_types_and_errors(structures: list[Structure]) -> None:
    """Test input types handling and error conditions."""
    # Test with mixed input types
    systems = [
        "Fe2O3",  # formula string
        Composition("LiPO4"),  # pymatgen composition
        "Na2CO3",  # formula string
        structures[0],  # pymatgen structure
        "Li-Fe-P-O",  # chemical system string
        "Fe-O",  # chemical system string
    ]
    fig = pmv.chem_sys_treemap(systems)
    assert isinstance(fig, go.Figure)

    # Check that systems with same elements are grouped
    labels = fig.data[0].labels
    parents = fig.data[0].parents
    values = fig.data[0].values

    binary_count = sum(
        val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "binary" in parent and label == "Fe-O"
    )
    assert binary_count == 2, "Fe2O3 and Fe-O should be merged"

    # Test with only Structure objects
    fig_structs = pmv.chem_sys_treemap(structures)
    assert isinstance(fig_structs, go.Figure)


@pytest.mark.parametrize(
    ("invalid_input", "expected_error"),
    [
        ("invalid", "Invalid show_counts="),
        ([], "Empty input: data sequence is empty"),
        ([1], "Expected str, Composition or Structure"),
    ],
)
def test_chem_sys_treemap_errors(invalid_input: Any, expected_error: str) -> None:
    """Test error conditions."""
    if invalid_input == "invalid":
        with pytest.raises((ValueError, TypeError), match=expected_error):
            pmv.chem_sys_treemap(["Fe-O"], show_counts=invalid_input)
    else:
        with pytest.raises((ValueError, TypeError), match=expected_error):
            pmv.chem_sys_treemap(invalid_input)


@pytest.mark.parametrize(
    ("group_by", "expected_fe_counts"),
    [
        ("formula", {"Fe2O3": 1, "Fe4O6": 1, "FeO": 1, "Li2O": 2}),
        ("reduced_formula", {"Fe2O3": 2, "FeO": 1, "Li2O": 2}),  # Fe2O3 includes Fe4O6
        ("chem_sys", {"Fe-O": 3, "Li-O": 2, "Fe-Li-O": 2}),
    ],
)
def test_chem_sys_treemap_grouping_modes(
    group_by: FormulaGroupBy, expected_fe_counts: dict[str, int]
) -> None:
    """Test different grouping modes with strict verification."""
    fig = pmv.chem_sys_treemap(GROUPING_TEST_SYSTEMS, group_by=group_by)

    labels = fig.data[0].labels
    parents = fig.data[0].parents
    values = fig.data[0].values

    # Get counts for relevant systems
    if group_by == "chem_sys":
        actual_counts = {
            label: val
            for label, parent, val in zip(labels, parents, values, strict=False)
            if any(x in parent for x in ("binary", "ternary"))
        }
    else:
        # For formula modes, the formulas are leaf nodes with chem_sys as parents
        formula_nodes = {
            label: val
            for label, parent, val in zip(labels, parents, values, strict=False)
            if any(x in parent for x in ("binary", "ternary")) and "/" in parent
        }
        actual_counts = formula_nodes

    # Verify each expected count
    for system, count in expected_fe_counts.items():
        if system in actual_counts:
            assert actual_counts[system] == count, (
                f"Incorrect count for {system} in {group_by} mode"
            )


def test_chem_sys_treemap_other_entries_styling() -> None:
    """Test that 'Other' entries have custom styling applied."""
    max_cells = 3  # Small enough to create "Other" entries
    fig = pmv.chem_sys_treemap(TEST_SYSTEMS, max_cells=max_cells)
    labels = fig.data[0].labels

    # Find all "Other" entries
    other_indices = [
        idx
        for idx, label in enumerate(labels)
        if "Other" in label and "more not shown" in label
    ]

    # Verify that we have at least one "Other" entry
    assert len(other_indices) > 0

    # Verify styling
    assert hasattr(fig.data[0], "marker")
    assert hasattr(fig.data[0].marker, "colors")
    assert fig.data[0].marker.colors is not None

    # Verify that "Other" entries have the custom color
    for idx in other_indices:
        assert fig.data[0].marker.colors[idx] == "rgba(255,255,255,0.1)"


@pytest.mark.parametrize(
    ("max_cells", "arity", "expected_systems", "expected_other_count"),
    [
        (5, "binary", 6, 4),  # 5 regular + 1 "Other" with 4 systems
        (3, "ternary", 4, 2),  # 3 regular + 1 "Other" with 2 systems
        (0, "binary", 9, 0),  # All systems, no "Other"
        (None, "binary", 9, 0),  # All systems, no "Other"
        (15, "binary", 9, 0),  # All systems, no "Other"
    ],
)
def test_chem_sys_treemap_max_cells(
    max_cells: int | None,
    arity: str,
    expected_systems: int,
    expected_other_count: int,
) -> None:
    """Test max_cells parameter."""
    fig = pmv.chem_sys_treemap(TEST_SYSTEMS, max_cells=max_cells)

    labels = fig.data[0].labels
    parents = fig.data[0].parents
    values = fig.data[0].values

    # Get systems for the specified arity
    arity_systems = {
        label: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if arity in parent
    }

    # Verify number of systems
    assert len(arity_systems) == expected_systems

    # Check for "Other" entry
    other_entries = [
        (label, val)
        for label, parent, val in zip(labels, parents, values, strict=False)
        if arity in parent and "Other" in label
    ]

    if expected_other_count > 0:
        assert len(other_entries) == 1
        other_label, other_value = other_entries[0]
        assert "more not shown" in other_label
        assert f"{expected_other_count}" in other_label
        assert other_value == expected_other_count

        # Verify "Other" entry has different styling
        colors = (
            fig.data[0].marker.colors if hasattr(fig.data[0].marker, "colors") else None
        )
        if colors:
            other_indices = [
                idx
                for idx, label in enumerate(labels)
                if "Other" in label and arity in parents[idx]
            ]
            assert len(other_indices) > 0
            assert colors[other_indices[0]] == "rgba(255,255,255,0.1)"
    else:
        assert len(other_entries) == 0


def test_chem_sys_treemap_high_arity_systems() -> None:
    """Test high-arity systems verification."""
    fig = pmv.chem_sys_treemap(HIGH_ARITY_SYSTEMS)
    labels = fig.data[0].labels
    parents = fig.data[0].parents
    values = fig.data[0].values

    # Verify arity levels
    arity_levels = {
        label.split(" (")[0]
        for label in labels
        if any(x in label for x in ("ternary", "quinary"))
    }
    assert arity_levels == {"ternary", "quinary"}

    # Verify arity counts
    arity_counts = {
        label.split(" (")[0]: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if parent == ""
    }
    assert arity_counts == {"ternary": 5, "quinary": 5}

    # Verify chemical systems
    ternary_systems = {
        label: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "ternary" in parent
    }
    quinary_systems = {
        label: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "quinary" in parent
    }

    assert len(ternary_systems) == 1
    assert "Cu-Ga-Zr" in ternary_systems
    assert ternary_systems["Cu-Ga-Zr"] == 5

    assert len(quinary_systems) == 2
    assert "Ag-Al-Cu-Pd-Zr" in quinary_systems
    assert "Ag-Al-Cu-Ni-Zr" in quinary_systems
    assert quinary_systems["Ag-Al-Cu-Pd-Zr"] == 2
    assert quinary_systems["Ag-Al-Cu-Ni-Zr"] == 3
