"""Test treemap plots for chemical systems."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, Literal, get_args

import plotly.graph_objects as go
import pytest
from pymatgen.core import Composition

import pymatviz as pmv
from pymatviz.treemap import ShowCounts


if TYPE_CHECKING:
    from pymatgen.core import Structure

# Type alias for arity formatter function
ArityFormatter = Callable[[str, int, int], str]

# Test data for chemical systems
TEST_SYSTEMS: list[str] = [
    # Binary systems
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
    # Ternary systems
    "LiFeO2",
    "NaFeO2",
    "KFeO2",
    "CaFeO3",
    "MgFeO3",
]

# High-arity test systems
HIGH_ARITY_SYSTEMS: list[str] = [
    "Zr42.5Ga2.5Cu55",
    "Zr37.5Ga5Cu57.5",
    "Zr40Ga7.5Cu52.5",
    "Zr40Ga5Cu55",
    "Zr42.5Ga5Cu52.5",  # ternary
    "Zr48Al8Cu32Ag8Pd4",
    "Zr48Al8Cu30Ag8Pd6",
    "Zr48Al8Cu34Ni2Ag8",
    "Zr48Al8Cu32Ni4Ag8",
    "Zr48Al8Cu30Ni6Ag8",  # quinary
]

# Test data for grouping
GROUPING_TEST_SYSTEMS: list[str] = [
    "Fe2O3",
    "Fe4O6",
    "FeO",  # binary Fe-O systems
    "Li2O",
    "Li2O",  # binary Li-O with duplicate
    "LiFeO2",
    "Li3FeO3",  # ternary Fe-Li-O systems
]


# Custom arity formatter for testing
def custom_formatter(arity: str, count: int, _total: int) -> str:
    """Format arity name with count in square brackets."""
    return f"{arity} [{count}]"


def test_chem_sys_treemap_basic() -> None:
    """Test chem_sys_treemap plot with various scenarios."""
    # Test basic functionality with mixed arity systems
    systems = ["Fe-O", "Li-P-O", "Fe", "O", "Li-O"]
    fig = pmv.chem_sys_treemap(systems)

    # Verify figure structure
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "treemap"
    assert hasattr(fig.data[0], "labels")
    assert hasattr(fig.data[0], "parents")
    assert hasattr(fig.data[0], "values")

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
    assert arity_levels == {"unary", "binary", "ternary"}, (
        "Missing or extra arity levels"
    )

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

    assert unary_elements == {"Fe", "O"}, "Incorrect unary elements"
    assert binary_systems == {"Fe-O", "Li-O"}, "Incorrect binary systems"
    assert ternary_systems == {"Li-O-P"}, "Incorrect ternary systems"

    # Verify counts
    unary_counts = {
        label: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "unary" in parent
    }
    binary_counts = {
        label: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "binary" in parent
    }
    ternary_counts = {
        label: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "ternary" in parent
    }

    assert all(count == 1 for count in unary_counts.values()), "Incorrect unary counts"
    assert all(count == 1 for count in binary_counts.values()), (
        "Incorrect binary counts"
    )
    assert all(count == 1 for count in ternary_counts.values()), (
        "Incorrect ternary counts"
    )

    # Verify hover template
    assert "hovertemplate" in fig.data[0], "Missing hover template"
    assert "Count:" in fig.data[0].hovertemplate, "Incorrect hover template"
    assert "percentEntry" in fig.data[0].hovertemplate, (
        "Missing percentage in hover template"
    )


def test_chem_sys_treemap_empty_level_and_customizations() -> None:
    """Test treemap with gaps in arity and customization options."""
    # Only unary and ternary systems, no binary
    systems = ["Fe", "O", "Li-Fe-O"]

    # Test with customizations
    fig = pmv.chem_sys_treemap(
        systems,
        color_discrete_sequence=["red", "blue", "green"],
        show_counts="value+percent",
    )

    # Apply additional customizations
    fig.update_traces(
        marker=dict(cornerradius=5),
        textinfo="label+value+percent entry",
    )

    # Verify customizations were applied
    assert fig.data[0].marker.cornerradius == 5, "Rounded corners not applied"
    assert fig.data[0].textinfo == "label+value+percent entry", "Text info not applied"

    # Extract data for testing
    labels = fig.data[0].labels
    parents = fig.data[0].parents
    values = fig.data[0].values

    # Check arity levels (should only have unary and ternary since no binary systems)
    arity_levels = {
        label.split(" (")[0]
        for label in labels
        if any(x in label for x in ("unary", "binary", "ternary"))
    }
    assert arity_levels == {"unary", "ternary"}, (
        "Should only have unary and ternary levels"
    )
    assert "binary" not in arity_levels, "Binary level should not be present"

    # Verify no binary systems
    binary_systems = {
        label
        for label, parent in zip(labels, parents, strict=False)
        if "binary" in parent
    }
    assert not binary_systems, "There should be no binary systems"

    # Verify counts
    unary_values = [
        val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "unary" in parent
    ]
    ternary_values = [
        val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "ternary" in parent and label == "Fe-Li-O"
    ]

    assert all(val == 1 for val in unary_values), "Each unary element should count once"
    assert len(ternary_values) == 1
    assert ternary_values[0] == 1


@pytest.mark.parametrize("show_counts", get_args(ShowCounts))
def test_chem_sys_treemap_show_counts_and_arity_formatting(
    show_counts: ShowCounts,
) -> None:
    """Test different show_counts options and arity formatting."""
    systems = ["Fe2O3", "FeO", "Li2O", "LiFeO2"]

    # Test with different show_counts options
    fig = pmv.chem_sys_treemap(systems, show_counts=show_counts)

    # Verify show_counts behavior
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

    # Test custom arity formatter
    fig_custom = pmv.chem_sys_treemap(systems, show_arity_counts=custom_formatter)
    labels = fig_custom.data[0].labels
    arity_labels = {
        label for label in labels if any(x in label for x in ("binary", "ternary"))
    }

    assert any("[3]" in label for label in arity_labels), (
        "Custom formatter not applied to binary systems"
    )
    assert any("[1]" in label for label in arity_labels), (
        "Custom formatter not applied to ternary systems"
    )

    # Test with show_arity_counts=False
    fig_no_counts = pmv.chem_sys_treemap(systems, show_arity_counts=False)
    labels = fig_no_counts.data[0].labels
    arity_labels = {
        label for label in labels if any(x in label for x in ("binary", "ternary"))
    }

    assert all("N=" not in label for label in arity_labels), (
        "Counts should not be shown"
    )
    assert all("%" not in label for label in arity_labels), (
        "Percentages should not be shown"
    )


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

    # Verify figure was created
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

    # Test error conditions
    with pytest.raises(ValueError, match="Invalid show_counts="):
        pmv.chem_sys_treemap(systems, show_counts="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Empty input: data sequence is empty"):
        pmv.chem_sys_treemap([])

    with pytest.raises(TypeError, match="Expected str, Composition or Structure"):
        pmv.chem_sys_treemap([1])


def test_chem_sys_treemap_grouping_modes() -> None:
    """Test different grouping modes with strict verification."""
    # Test all three grouping modes
    grouping_modes: Final[
        dict[Literal["formula", "reduced_formula", "chem_sys"], dict[str, int]]
    ] = {
        "formula": {
            "Fe2O3": 1,
            "Fe4O6": 1,
            "FeO": 1,  # each formula separate
            "Li2O": 2,  # duplicate
            "LiFeO2": 1,
            "Li3FeO3": 1,  # each formula separate
        },
        "reduced_formula": {
            "Fe2O3": 2,  # Fe2O3 and Fe4O6
            "FeO": 1,
            "Li2O": 2,  # duplicates
            "LiFeO2": 1,
            "Li3FeO3": 1,
        },
        "chem_sys": {
            "Fe-O": 3,  # Fe2O3, Fe4O6, FeO
            "Li-O": 2,  # two Li2O
            "Fe-Li-O": 2,  # LiFeO2, Li3FeO3
        },
    }

    for group_by, expected_counts in grouping_modes.items():
        fig = pmv.chem_sys_treemap(GROUPING_TEST_SYSTEMS, group_by=group_by)

        # Extract data
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
            # For formula and reduced_formula modes, the formulas are leaf nodes with
            # chem_sys as parents
            formula_nodes = {
                label: val
                for label, parent, val in zip(labels, parents, values, strict=False)
                if any(x in parent for x in ("binary", "ternary")) and "/" in parent
            }
            actual_counts = formula_nodes

        # Verify each expected count
        for system, count in expected_counts.items():
            assert system in actual_counts, (
                f"System {system} missing in {group_by} mode"
            )
            assert actual_counts[system] == count, (
                f"Incorrect count for {system} in {group_by} mode"
            )

        # Verify the structure based on grouping mode
        if group_by != "chem_sys":
            # In formula modes, we should have formula nodes with chem_sys parents
            formula_parent_nodes = [parent for parent in parents if "/" in parent]
            assert len(formula_parent_nodes) > 0, (
                f"Formula parent nodes missing in {group_by} mode"
            )
        else:
            # In chem_sys mode, we should have direct chem_sys nodes
            chem_sys_nodes = [
                label
                for label, parent in zip(labels, parents, strict=False)
                if any(x in parent for x in ("binary", "ternary"))
            ]
            assert len(chem_sys_nodes) == len(expected_counts), (
                f"Incorrect number of chem_sys nodes in {group_by} mode"
            )


def test_chem_sys_treemap_other_entries_styling() -> None:
    """Test that 'Other' entries have custom styling applied to make them visually
    distinct."""
    # Create a treemap with a small max_cells value to ensure we get "Other" entries
    max_cells = (
        3  # Small enough to create "Other" entries for both binary and ternary systems
    )
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

    # Verify that marker.colors is set
    assert hasattr(fig.data[0], "marker"), "Figure data has no marker attribute"
    assert hasattr(fig.data[0].marker, "colors"), "marker has no colors attribute"
    assert fig.data[0].marker.colors is not None, "marker.colors is None"

    # Verify that "Other" entries have the custom color (lightgray)
    for idx in other_indices:
        assert fig.data[0].marker.colors[idx] == "rgba(255,255,255,0.1)"

    # Verify that non-"Other" entries do not have the lightgray color
    non_other_indices = [
        idx
        for idx, label in enumerate(labels)
        if not ("Other" in label and "more not shown" in label)
    ]
    for idx in non_other_indices:
        assert fig.data[0].marker.colors[idx] != "lightgray"

    # Test with different max_cells values to ensure consistent styling
    for test_max_cells in [1, 2, 5]:
        test_fig = pmv.chem_sys_treemap(TEST_SYSTEMS, max_cells=test_max_cells)

        # Find "Other" entries
        test_other_indices = [
            idx
            for idx, label in enumerate(test_fig.data[0].labels)
            if "Other" in label and "more not shown" in label
        ]

        if test_other_indices:
            # Verify that marker.colors is set
            assert hasattr(test_fig.data[0], "marker"), (
                "Figure data has no marker attribute"
            )
            assert hasattr(test_fig.data[0].marker, "colors"), (
                "marker has no colors attribute"
            )
            assert test_fig.data[0].marker.colors is not None, "marker.colors is None"

            # Verify that "Other" entries have the custom color
            for idx in test_other_indices:
                assert test_fig.data[0].marker.colors[idx] == "rgba(255,255,255,0.1)"


def test_chem_sys_treemap_high_arity_and_max_cells() -> None:
    """Test high-arity systems and max_cells parameter."""
    # Test high-arity systems
    fig_high = pmv.chem_sys_treemap(HIGH_ARITY_SYSTEMS)
    # Extract data
    labels = fig_high.data[0].labels
    parents = fig_high.data[0].parents
    values = fig_high.data[0].values

    # Verify arity levels
    arity_levels = {
        label.split(" (")[0]
        for label in labels
        if any(x in label for x in ("ternary", "quinary"))
    }
    assert arity_levels == {"ternary", "quinary"}, "Missing arity levels"

    # Verify arity counts
    arity_counts = {
        label.split(" (")[0]: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if parent == ""
    }
    assert arity_counts == {"ternary": 5, "quinary": 5}, "Incorrect arity counts"

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
    assert ternary_systems["Cu-Ga-Zr"] == 5, "Incorrect ternary count"

    assert len(quinary_systems) == 2, "Incorrect number of quinary systems"
    assert "Ag-Al-Cu-Pd-Zr" in quinary_systems, "Missing quinary system"
    assert "Ag-Al-Cu-Ni-Zr" in quinary_systems, "Missing quinary system"
    assert quinary_systems["Ag-Al-Cu-Pd-Zr"] == 2, "Incorrect quinary count"
    assert quinary_systems["Ag-Al-Cu-Ni-Zr"] == 3, "Incorrect quinary count"

    # Test max_cells parameter
    test_cases = [
        (5, "binary", 6, True, 4),  # 5 regular + 1 "Other" with 4 systems
        (3, "ternary", 4, True, 2),  # 3 regular + 1 "Other" with 2 systems
        (0, "binary", 9, False, 0),  # All systems, no "Other"
        (None, "binary", 9, False, 0),  # All systems, no "Other"
        (15, "binary", 9, False, 0),  # All systems, no "Other"
    ]

    for max_cells, arity, expected_systems, has_other, other_count in test_cases:
        fig = pmv.chem_sys_treemap(TEST_SYSTEMS, max_cells=max_cells)

        # Extract data
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
        assert len(arity_systems) == expected_systems, (
            f"Incorrect system count with max_cells={max_cells}"
        )

        # Check for "Other" entry
        other_entries = [
            (label, val)
            for label, parent, val in zip(labels, parents, values, strict=False)
            if arity in parent and "Other" in label
        ]

        if has_other:
            assert len(other_entries) == 1, (
                f"Missing 'Other' entry with max_cells={max_cells}"
            )
            other_label, other_value = other_entries[0]
            assert "more not shown" in other_label, "Incorrect 'Other' label"
            assert f"{other_count}" in other_label, "Incorrect count in 'Other' label"
            assert other_value == other_count, "Incorrect 'Other' value"

            # Verify "Other" entry has different styling
            colors = (
                fig.data[0].marker.colors
                if hasattr(fig.data[0].marker, "colors")
                else None
            )
            if colors:
                other_indices = [
                    idx
                    for idx, label in enumerate(labels)
                    if "Other" in label and arity in parents[idx]
                ]
                assert len(other_indices) > 0, "Could not find 'Other' entry index"
                assert colors[other_indices[0]] == "rgba(255,255,255,0.1)"
        else:
            assert len(other_entries) == 0, (
                f"Unexpected 'Other' entry with max_cells={max_cells}"
            )
