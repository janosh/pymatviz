"""Test treemap plots for chemical systems."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import plotly.graph_objects as go
import pytest
from pymatgen.core import Composition

import pymatviz as pmv
from pymatviz.treemap import ShowCounts


if TYPE_CHECKING:
    from pymatgen.core import Structure


def test_chem_sys_treemap_basic() -> None:
    """Test chem_sys_treemap plot with various scenarios."""
    # Test basic functionality with mixed arity systems
    systems = [
        "Fe-O",
        "Li-P-O",
        "Fe",
        "O",
        "Li-O",
    ]  # removed Li-Fe-P-O to avoid quaternary
    fig = pmv.chem_sys_treemap(systems)
    assert isinstance(fig, go.Figure)

    # Check the number of traces (should be 1 for treemap)
    assert len(fig.data) == 1

    # Get all unique labels at each level
    labels = fig.data[0].labels
    parents = fig.data[0].parents

    # Check arity levels are present and correctly named
    arity_levels = {
        label.split(" (")[0]  # Remove the count part
        for label in labels
        if any(x in label for x in ("unary", "binary", "ternary"))
    }
    assert arity_levels == {"unary", "binary", "ternary"}

    # Check unary elements
    unary_elements = {
        label
        for label, parent in zip(labels, parents, strict=False)
        if "unary" in parent
    }
    assert unary_elements == {"Fe", "O"}  # only Fe and O appear as pure elements

    # Check binary systems
    binary_systems = {
        label
        for label, parent in zip(labels, parents, strict=False)
        if "binary" in parent
    }
    assert binary_systems == {"Fe-O", "Li-O"}

    # Check ternary systems
    ternary_systems = {
        label
        for label, parent in zip(labels, parents, strict=False)
        if "ternary" in parent
    }
    assert ternary_systems == {"Li-O-P"}  # only Li-P-O


def test_chem_sys_treemap_empty_level() -> None:
    """Test that the treemap plot handles systems with gaps in arity correctly."""
    # Only unary and ternary systems, no binary
    systems = ["Fe", "O", "Li-Fe-O"]
    fig = pmv.chem_sys_treemap(systems)

    # Get all unique labels at each level
    labels = fig.data[0].labels
    parents = fig.data[0].parents

    # Check arity levels (should only have unary and ternary since no binary systems)
    arity_levels = {
        label.split(" (")[0]  # Remove the count part
        for label in labels
        if any(x in label for x in ("unary", "binary", "ternary"))
    }
    assert arity_levels == {"unary", "ternary"}  # only levels with data are shown

    # Check that no systems have binary as parent
    binary_systems = {
        label
        for label, parent in zip(labels, parents, strict=False)
        if "binary" in parent
    }
    assert not binary_systems, "There should be no binary systems"

    # Check the values are correct
    values = fig.data[0].values

    # Find indices for unary elements and check their values
    unary_values = [
        val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "unary" in parent and label in {"Fe", "O"}
    ]
    assert all(val == 1 for val in unary_values), "Each unary element should count once"

    # Find index for ternary system and check its value
    ternary_values = [
        val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "ternary" in parent and label == "Fe-Li-O"
    ]
    assert len(ternary_values) == 1
    assert ternary_values[0] == 1, "Ternary system should count once"


@pytest.mark.parametrize("show_counts", get_args(ShowCounts))
def test_chem_sys_treemap_show_counts(show_counts: ShowCounts) -> None:
    """Test different show_counts options."""
    systems = ["Fe2O3", "FeO", "Li2O"]
    fig = pmv.chem_sys_treemap(systems, show_counts=show_counts)

    if show_counts == "value":
        assert "texttemplate" in fig.data[0]
        assert "N=" in fig.data[0].texttemplate

    elif show_counts == "value+percent":
        assert "texttemplate" in fig.data[0]
        assert "N=" in fig.data[0].texttemplate
        assert "percentEntry" in fig.data[0].texttemplate

    elif show_counts == "percent":
        assert "textinfo" in fig.data[0]
        assert "percent" in fig.data[0].textinfo

    elif show_counts is False:
        assert fig.data[0].textinfo is None


def test_chem_sys_treemap_show_arity_counts() -> None:
    """Test custom arity count formatting."""
    systems = ["Fe2O3", "FeO", "Li2O", "LiFeO2"]

    # Test default formatter
    fig = pmv.chem_sys_treemap(systems)
    labels = fig.data[0].labels
    arity_labels = {
        label for label in labels if any(x in label for x in ("binary", "ternary"))
    }
    assert any("N=" in label for label in arity_labels)
    assert any("%" in label for label in arity_labels)

    # Test custom formatter
    def custom_formatter(arity: str, count: int, _total: int) -> str:
        return f"{arity} [{count}]"

    fig = pmv.chem_sys_treemap(systems, show_arity_counts=custom_formatter)
    labels = fig.data[0].labels
    arity_labels = {
        label for label in labels if any(x in label for x in ("binary", "ternary"))
    }
    assert any("[3]" in label for label in arity_labels)  # 3 binary systems
    assert any("[1]" in label for label in arity_labels)  # 1 ternary system

    # Test with show_arity_counts=False
    fig = pmv.chem_sys_treemap(systems, show_arity_counts=False)
    labels = fig.data[0].labels
    arity_labels = {
        label for label in labels if any(x in label for x in ("binary", "ternary"))
    }
    assert all("N=" not in label for label in arity_labels)
    assert all("%" not in label for label in arity_labels)


def test_chem_sys_treemap_raises() -> None:
    """Test that chem_sys_treemap raises appropriate errors."""
    systems = ["Fe-O", "Li-P-O"]

    with pytest.raises(ValueError, match="Invalid show_counts="):
        pmv.chem_sys_treemap(systems, show_counts="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Empty input: data sequence is empty"):
        pmv.chem_sys_treemap([])


def test_chem_sys_treemap_input_types(structures: list[Structure]) -> None:
    """Test chem_sys_treemap plot with various input types."""
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

    # Check that all systems were converted to the same format
    # Fe2O3 and Fe-O should be counted together
    binary_count = sum(
        val
        for label, parent, val in zip(
            fig.data[0].labels, fig.data[0].parents, fig.data[0].values, strict=False
        )
        if "binary" in parent and label == "Fe-O"
    )
    assert binary_count == 2, "Fe2O3 and Fe-O should be merged"

    # Test with only Structure objects
    fig_structs = pmv.chem_sys_treemap(structures)
    assert isinstance(fig_structs, go.Figure)

    # Test with invalid input type
    with pytest.raises(TypeError, match="Expected str, Composition or Structure"):
        pmv.chem_sys_treemap([1])


def test_chem_sys_treemap_high_arity() -> None:
    """Test chem_sys_treemap with high-arity systems."""
    formulas = [
        "Zr42.5Ga2.5Cu55",  # ternary
        "Zr37.5Ga5Cu57.5",  # ternary
        "Zr40Ga7.5Cu52.5",  # ternary
        "Zr40Ga5Cu55",  # ternary
        "Zr42.5Ga5Cu52.5",  # ternary
        "Zr48Al8Cu32Ag8Pd4",  # quinary
        "Zr48Al8Cu30Ag8Pd6",  # quinary
        "Zr48Al8Cu34Ni2Ag8",  # quinary
        "Zr48Al8Cu32Ni4Ag8",  # quinary
        "Zr48Al8Cu30Ni6Ag8",  # quinary
    ]
    fig = pmv.chem_sys_treemap(formulas)

    # Get all unique labels at each level
    labels = fig.data[0].labels
    parents = fig.data[0].parents
    values = fig.data[0].values

    # Check arity levels are present and correctly named
    arity_levels = {
        label.split(" (")[0]  # Remove the count part
        for label in labels
        if any(x in label for x in ("ternary", "quinary"))
    }
    assert arity_levels == {"ternary", "quinary"}

    # Get counts for each arity level
    arity_counts = {
        label.split(" (")[0]: val  # Remove the count part
        for label, parent, val in zip(labels, parents, values, strict=False)
        if parent == ""  # root level = arity level
    }
    assert arity_counts == {"ternary": 5, "quinary": 5}

    # Check chemical systems under ternary
    ternary_systems = {
        label: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "ternary" in parent
    }
    assert len(ternary_systems) == 1  # all ternaries have same elements
    assert "Cu-Ga-Zr" in ternary_systems  # elements are sorted alphabetically
    assert ternary_systems["Cu-Ga-Zr"] == 5

    # Check chemical systems under quinary
    quinary_systems = {
        label: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "quinary" in parent
    }
    assert len(quinary_systems) == 2  # two different quinary systems
    assert "Ag-Al-Cu-Pd-Zr" in quinary_systems  # elements are sorted alphabetically
    assert "Ag-Al-Cu-Ni-Zr" in quinary_systems
    assert quinary_systems["Ag-Al-Cu-Pd-Zr"] == 2
    assert quinary_systems["Ag-Al-Cu-Ni-Zr"] == 3


def test_chem_sys_treemap_grouping() -> None:
    """Test chem_sys_treemap with different grouping modes."""
    # Test with multiple formulas that have the same elements
    systems = [
        "Fe2O3",  # binary
        "Fe4O6",  # same as Fe2O3 when reduced
        "FeO",  # different formula but same system
        "Li2O",  # binary
        "Li2O",  # binary (duplicate)
        "LiFeO2",  # ternary
        "Li3FeO3",  # ternary (same system as LiFeO2)
    ]

    # Test with group_by="formula" (count each formula separately)
    fig_formula = pmv.chem_sys_treemap(systems, group_by="formula")
    labels = fig_formula.data[0].labels
    parents = fig_formula.data[0].parents
    values = fig_formula.data[0].values

    formula_counts = {
        label: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "binary" in parent or "ternary" in parent
    }

    # Each unique formula should be counted separately
    assert formula_counts["Fe2O3"] == 1
    assert formula_counts["Fe4O6"] == 1
    assert formula_counts["FeO"] == 1
    assert formula_counts["Li2O"] == 2  # duplicate
    assert formula_counts["LiFeO2"] == 1
    assert formula_counts["Li3FeO3"] == 1

    # Test with group_by="reduced_formula" (group formulas with same reduced form)
    fig_reduced = pmv.chem_sys_treemap(systems, group_by="reduced_formula")
    labels = fig_reduced.data[0].labels
    parents = fig_reduced.data[0].parents
    values = fig_reduced.data[0].values

    reduced_counts = {
        label: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if "binary" in parent or "ternary" in parent
    }

    # Formulas with same reduced form should be grouped
    assert reduced_counts["Fe2O3"] == 2  # Fe2O3 and Fe4O6
    assert reduced_counts["FeO"] == 1
    assert reduced_counts["Li2O"] == 2  # duplicates
    assert reduced_counts["LiFeO2"] == 1
    assert reduced_counts["Li3FeO3"] == 1

    # Test with group_by="chem_sys" (default)
    fig_system = pmv.chem_sys_treemap(systems)  # default group_by="chem_sys"
    labels = fig_system.data[0].labels
    parents = fig_system.data[0].parents
    values = fig_system.data[0].values

    system_counts = {
        label: val
        for label, parent, val in zip(labels, parents, values, strict=False)
        if any(x in parent for x in ("binary", "ternary"))
    }

    # Formulas with same elements should be grouped
    assert system_counts["Fe-O"] == 3  # Fe2O3, Fe4O6, FeO
    assert system_counts["Li-O"] == 2  # two Li2O
    assert system_counts["Fe-Li-O"] == 2  # LiFeO2, Li3FeO3
