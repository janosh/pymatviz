"""Test sunburst plots for space groups and chemical systems."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import pandas as pd
import plotly.graph_objects as go
import pytest

import pymatviz as pmv
from pymatviz.sunburst import ShowCounts, spacegroup_sunburst


if TYPE_CHECKING:
    from pymatgen.core import Structure


@pytest.mark.parametrize("show_counts", get_args(ShowCounts))
def test_spacegroup_sunburst(show_counts: ShowCounts) -> None:
    # spg numbers
    fig = spacegroup_sunburst(range(1, 231), show_counts=show_counts)

    assert isinstance(fig, go.Figure)
    assert set(fig.data[0].parents) == {"", *get_args(pmv.typing.CrystalSystem)}
    assert fig.data[0].branchvalues == "total"

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


def test_spacegroup_sunburst_invalid_show_counts() -> None:
    """Test that invalid show_counts values raise ValueError."""
    show_counts = "invalid"
    with pytest.raises(ValueError, match=f"Invalid {show_counts=}"):
        spacegroup_sunburst([1], show_counts=show_counts)  # type: ignore[arg-type]


def test_spacegroup_sunburst_single_item() -> None:
    """Test with single-item input."""
    fig = spacegroup_sunburst([1], show_counts="value")
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].ids) == 2  # one for crystal system, one for spg number

    fig = spacegroup_sunburst(["P1"], show_counts="value+percent")
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].ids) == 2


def test_spacegroup_sunburst_other_types(
    spg_symbols: list[str], structures: list[Structure]
) -> None:
    """Test with other types of input."""
    # test with pandas series
    series = pd.Series([*[1] * 3, *[2] * 10, *[3] * 5])
    fig = spacegroup_sunburst(series, show_counts="value")
    assert isinstance(fig, go.Figure)
    values = [*map(int, fig.data[0].values)]
    assert values == [10, 5, 3, 13, 5], f"actual {values=}"

    # test with strings of space group symbols
    fig = spacegroup_sunburst(spg_symbols)
    assert isinstance(fig, go.Figure)

    # test with pymatgen structures
    fig = spacegroup_sunburst(structures)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    ("max_slices", "max_slices_mode", "expected_systems", "expected_others"),
    [
        # Edge cases for max_slices
        (None, "other", 7, None),  # no limit
        (0, "other", 7, None),  # treated as None
        (-1, "other", 7, None),  # negative treated as None
        (10, "other", 7, None),  # greater than number of systems
        # Normal cases with "other" mode
        (
            1,
            "other",
            4,
            [("Other (4 more not shown)", 4), ("Other (1 more not shown)", 1)],
        ),  # top 1 + Others
        (2, "other", 5, [("Other (3 more not shown)", 3)]),  # top 2 + Other
        (3, "other", 6, [("Other (2 more not shown)", 2)]),  # top 3 + Other
        # Cases with "drop" mode
        (1, "drop", 2, None),  # only top 1
        (2, "drop", 4, None),  # only top 2
        (3, "drop", 5, None),  # only top 3
    ],
)
def test_spacegroup_sunburst_max_slices(
    max_slices: int | None,
    max_slices_mode: Literal["other", "drop"],
    expected_systems: int,
    expected_others: list[tuple[str, int]] | None,
) -> None:
    """Test spacegroup_sunburst with max_slices functionality and edge cases.

    Args:
        max_slices: Maximum number of space groups to show per crystal system
        max_slices_mode: How to handle space groups beyond max_slices
        expected_systems: Expected number of space group entries (including "Other")
        expected_others: Expected "Other" entry labels and counts if present
    """
    # Create a dataset with multiple space groups per crystal system
    # Triclinic (1-2), Monoclinic (3-15), Orthorhombic (16-74)
    spg_numbers = (1, 2, 3, 4, 5, 6, 7)

    fig = pmv.spacegroup_sunburst(
        spg_numbers, max_slices=max_slices, max_slices_mode=max_slices_mode
    )
    assert isinstance(fig, go.Figure)

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
            (label, val) for label, val in space_groups.items() if "Other" in label
        ]
        assert len(other_entries) == len(expected_others)

        for expected_other in expected_others:
            expected_label, expected_count = expected_other
            found = False
            for label, val in other_entries:
                if label == expected_label and val == expected_count:
                    found = True
                    break
            assert found, (
                f"{expected_label=} with {expected_count=} not in {other_entries=}"
            )
    else:
        assert not any("Other" in label for label in space_groups)


def test_spacegroup_sunburst_max_slices_mode_invalid() -> None:
    """Test spacegroup_sunburst with invalid max_slices_mode."""
    spg_numbers = [1, 2, 3]
    with pytest.raises(ValueError, match="Invalid max_slices_mode="):
        pmv.spacegroup_sunburst(spg_numbers, max_slices=1, max_slices_mode="invalid")  # type: ignore[arg-type]


def test_chem_sys_sunburst_basic() -> None:
    """Test chem_sys_sunburst plot with various scenarios."""
    # Test basic functionality with mixed arity systems
    systems = ["Fe-O", "Li-P-O", "Li-Fe-P-O", "Fe", "O", "Li-O"]
    fig = pmv.chem_sys_sunburst(systems)
    assert isinstance(fig, go.Figure)

    # Check the number of traces (should be 1 for sunburst)
    assert len(fig.data) == 1

    # Get all unique labels at each level
    labels = fig.data[0].labels
    parents = fig.data[0].parents

    # Check arity levels are present and correctly named
    arity_levels = {
        label for label in labels if label in {"unary", "binary", "ternary"}
    }
    assert arity_levels == {"unary", "binary", "ternary"}

    # Check unary elements
    unary_elements = {
        label for idx, label in enumerate(labels) if parents[idx] == "unary"
    }
    assert unary_elements == {"Fe", "O"}  # only Fe and O appear as pure elements

    # Check binary systems
    binary_systems = {
        label for idx, label in enumerate(labels) if parents[idx] == "binary"
    }
    assert binary_systems == {"Fe-O", "Li-O"}

    # Check ternary systems
    ternary_systems = {
        label for idx, label in enumerate(labels) if parents[idx] == "ternary"
    }
    assert ternary_systems == {"Li-O-P"}


def test_chem_sys_sunburst_empty_level() -> None:
    """Test that the sunburst plot handles systems with gaps in arity correctly."""
    # Only unary and ternary systems, no binary
    systems = ["Fe", "O", "Li-Fe-O"]
    fig = pmv.chem_sys_sunburst(systems)

    # Get all unique labels at each level
    labels = fig.data[0].labels
    parents = fig.data[0].parents

    # Check arity levels (should only have unary and ternary since no binary systems)
    arity_levels = {
        label for label in labels if label in {"unary", "binary", "ternary"}
    }
    assert arity_levels == {"unary", "ternary"}  # only levels with data are shown

    # Check that no systems have binary as parent
    binary_systems = {
        label for idx, label in enumerate(labels) if parents[idx] == "binary"
    }
    assert not binary_systems, "There should be no binary systems"

    # Check the values are correct
    values = fig.data[0].values  # noqa: PD011

    # Find indices for unary elements and check their values
    unary_values = [val for idx, val in enumerate(values) if labels[idx] in {"Fe", "O"}]
    assert all(val == 1 for val in unary_values), "Each unary element should count once"

    # Find index for ternary system and check its value
    ternary_values = [val for idx, val in enumerate(values) if labels[idx] == "Fe-Li-O"]
    assert len(ternary_values) == 1
    assert ternary_values[0] == 1, "Ternary system should count once"


def test_chem_sys_sunburst_raises() -> None:
    """Test that arity_sunburst raises appropriate errors."""
    systems = ["Fe-O", "Li-P-O"]
    with pytest.raises(ValueError, match="Invalid show_counts="):
        pmv.chem_sys_sunburst(systems, show_counts="invalid")  # type: ignore[arg-type]


def test_chem_sys_sunburst_input_types(structures: list[Structure]) -> None:
    """Test chem_sys_sunburst plot with various input types."""
    from pymatgen.core import Composition

    # Test with mixed input types
    systems = [
        "Fe2O3",  # formula string
        Composition("LiPO4"),  # pymatgen composition
        "Na2CO3",  # formula string
        structures[0],  # pymatgen structure
        "Li-Fe-P-O",  # chemical system string
        "Fe-O",  # chemical system string
    ]
    fig = pmv.chem_sys_sunburst(systems)
    assert isinstance(fig, go.Figure)

    # Check that all systems were converted to the same format
    # Fe2O3 and Fe-O should be counted together
    binary_count = sum(
        val
        for idx, val in enumerate(fig.data[0].values)
        if fig.data[0].parents[idx] == "binary" and fig.data[0].labels[idx] == "Fe-O"
    )
    assert binary_count == 2, "Fe2O3 and Fe-O should be merged"

    # Test with only Structure objects
    fig_structs = pmv.chem_sys_sunburst(structures)
    assert isinstance(fig_structs, go.Figure)

    # Test with invalid input type
    with pytest.raises(TypeError, match="Expected str, Composition or Structure"):
        pmv.chem_sys_sunburst([1])


def test_chem_sys_sunburst_high_arity() -> None:
    """Test chem_sys_sunburst plot with various scenarios."""
    # Test with mixed arity systems
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
    fig = pmv.chem_sys_sunburst(formulas)
    assert isinstance(fig, go.Figure)

    # Check the number of traces (should be 1 for sunburst)
    assert len(fig.data) == 1

    # Get all unique labels at each level
    labels = fig.data[0].labels
    parents = fig.data[0].parents
    values = fig.data[0].values  # noqa: PD011

    # Check arity levels are present and correctly named
    arity_levels = {label for label in labels if label in {"ternary", "quinary"}}
    assert arity_levels == {"ternary", "quinary"}

    # Get counts for each arity level
    arity_counts = {
        labels[idx]: values[idx]
        for idx in range(len(labels))
        if parents[idx] == ""  # root level = arity level
    }
    assert arity_counts == {"ternary": 5, "quinary": 5}

    # Check chemical systems under ternary
    ternary_systems = {
        labels[idx]: values[idx]
        for idx in range(len(labels))
        if parents[idx] == "ternary"
    }
    assert len(ternary_systems) == 1  # all ternaries have same elements
    assert "Cu-Ga-Zr" in ternary_systems  # elements are sorted alphabetically
    assert ternary_systems["Cu-Ga-Zr"] == 5

    # Check chemical systems under quinary
    quinary_systems = {
        labels[idx]: values[idx]
        for idx in range(len(labels))
        if parents[idx] == "quinary"
    }
    assert len(quinary_systems) == 2  # two different quinary systems
    assert "Ag-Al-Cu-Pd-Zr" in quinary_systems  # elements are sorted alphabetically
    assert "Ag-Al-Cu-Ni-Zr" in quinary_systems
    assert quinary_systems["Ag-Al-Cu-Pd-Zr"] == 2
    assert quinary_systems["Ag-Al-Cu-Ni-Zr"] == 3


def test_chem_sys_sunburst_large_dataset() -> None:
    """Test chem_sys_sunburst with a larger dataset including various edge cases."""
    from pymatgen.core import Composition

    systems = [
        # Binary systems with different stoichiometry
        "Fe2O3",
        "FeO",
        "Fe3O4",  # all should be counted as Fe-O
        "Li2O",
        "Li2O",
        "Li2O",  # multiple instances of same formula
        # Ternary systems
        "LiFePO4",
        "Li3PO4",
        "LiFeO2",  # mixed stoichiometry
        "Na3AlF6",
        "Na3AlF6",  # multiple instances
        "KMgCl3",
        "K2MgCl4",  # same elements, different stoichiometry
        # Quaternary systems
        "KAlSiO4",
        "NaAlSiO4",
        "LiAlSiO4",  # MSiAlO4 family
        "Li2FeSiO4",
        "Na2FeSiO4",  # M2FeSiO4 family
        # High-arity systems
        "Y3Al5O12",  # garnet
        "Ba2NaNb5O15",  # tungsten bronze
        "Pb(Zr0.52Ti0.48)O3",  # PZT
        # Edge cases
        Composition("H2O"),  # simple molecule as Composition
        "O2",  # diatomic
        "Fe",  # pure element
        "U",  # heavy element
    ]  # removed empty string as it should be handled in edge cases test

    fig = pmv.chem_sys_sunburst(systems)
    assert isinstance(fig, go.Figure)

    labels = fig.data[0].labels
    parents = fig.data[0].parents
    values = fig.data[0].values  # noqa: PD011

    # Check all arity levels are present
    arity_levels = {
        label
        for label in labels
        if label in {"unary", "binary", "ternary", "quaternary", "quinary"}
    }
    assert "unary" in arity_levels
    assert "binary" in arity_levels
    assert "ternary" in arity_levels
    assert "quaternary" in arity_levels

    # Get counts for each system
    system_counts = {
        labels[idx]: values[idx]
        for idx in range(len(labels))
        if parents[idx] != ""  # exclude arity level labels
    }

    # Check binary systems are properly merged
    assert system_counts.get("Fe-O") == 3  # Fe2O3, FeO, Fe3O4
    assert system_counts.get("H-O") == 1  # H2O
    assert system_counts.get("Li-O") == 3  # three Li2O

    # Check ternary systems
    fe_li_systems = {
        label: val
        for label, val in system_counts.items()
        if all(elem in label for elem in ["Fe", "Li"])
    }
    assert len(fe_li_systems) > 0  # should have some Fe-Li containing systems

    # Check quaternary systems
    al_si_systems = {
        label: val
        for label, val in system_counts.items()
        if all(elem in label for elem in ["Al", "Si"])
    }
    assert len(al_si_systems) > 0  # should have some Al-Si containing systems


def test_chem_sys_sunburst_empty_and_invalid() -> None:
    """Test chem_sys_sunburst with empty and invalid inputs."""
    with pytest.raises(ValueError, match="Empty input: data sequence is empty"):
        pmv.chem_sys_sunburst([])

    for case in [
        [""],  # list with empty string
        [""],  # list with empty string
        [" "],  # list with whitespace
        ["   "],  # multiple spaces
        ["\t"],  # tab
        ["\n"],  # newline
    ]:
        with pytest.raises(ValueError, match="Invalid formula"):
            pmv.chem_sys_sunburst(case)


def test_chem_sys_sunburst_invalid_formulas() -> None:
    """Test chem_sys_sunburst with invalid chemical formulas."""
    invalid_cases = [
        ["H2O", "NotAFormula", "Fe2O3"],  # invalid element name
        ["H2O", "Li2O", "123"],  # pure number
        ["H2O", "Li2O", "!@#"],  # special characters
        ["H2O", "Li2O", "Na++"],  # invalid charge syntax
    ]

    for case in invalid_cases:
        with pytest.raises(ValueError, match="Invalid formula"):
            pmv.chem_sys_sunburst(case)


def test_chem_sys_sunburst_case_and_whitespace() -> None:
    """Test chem_sys_sunburst handles case and whitespace variations."""
    # leading/trailing whitespace variations should all count as single formula
    systems = ["Fe2O3", " Fe2O3", "Fe2O3 ", " Fe2O3 "]

    fig = pmv.chem_sys_sunburst(systems)
    assert isinstance(fig, go.Figure)

    # All variations should be counted as the same system
    binary_count = sum(
        val
        for idx, val in enumerate(fig.data[0].values)
        if fig.data[0].parents[idx] == "binary" and fig.data[0].labels[idx] == "Fe-O"
    )
    assert binary_count == 4, "All Fe2O3 variations should be counted together"


def test_chem_sys_sunburst_complex_formulas() -> None:
    """Test chem_sys_sunburst with complex chemical formulas."""
    systems = [
        "(UO2)3(PO4)2",  # complex grouping
        "Ca5(PO4)3F",  # fluorapatite
        "KFe3(SO4)2(OH)6",  # jarosite
        "Pb(Zr0.52Ti0.48)O3",  # PZT with fractional occupancy
        "La0.7Sr0.3MnO3",  # LSMO with fractional composition
    ]

    fig = pmv.chem_sys_sunburst(systems)
    assert isinstance(fig, go.Figure)

    # Check that complex formulas are properly parsed
    labels = fig.data[0].labels
    parents = fig.data[0].parents

    # Get all chemical systems (excluding arity level labels)
    chemical_systems = {label for idx, label in enumerate(labels) if parents[idx] != ""}

    # Check that elements are properly extracted and sorted
    assert "O-P-U" in chemical_systems  # from (UO2)3(PO4)2
    assert "Ca-F-O-P" in chemical_systems  # from Ca5(PO4)3F
    assert "Fe-H-K-O-S" in chemical_systems  # from KFe3(SO4)2(OH)6
    assert "O-Pb-Ti-Zr" in chemical_systems  # from PZT
    assert "La-Mn-O-Sr" in chemical_systems  # from LSMO


def test_chem_sys_sunburst_grouping() -> None:
    """Test chem_sys_sunburst with different grouping modes."""
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
    fig_formula = pmv.chem_sys_sunburst(systems, group_by="formula")
    assert isinstance(fig_formula, go.Figure)

    # Get counts for each formula
    labels = fig_formula.data[0].labels
    parents = fig_formula.data[0].parents
    values = fig_formula.data[0].values  # noqa: PD011

    formula_counts = {
        labels[idx]: values[idx]
        for idx in range(len(labels))
        if parents[idx] not in {"", "unary", "binary", "ternary"}  # only formula level
    }

    # Each unique formula should be counted separately
    assert formula_counts["Fe2O3"] == 1
    assert formula_counts["Fe4O6"] == 1
    assert formula_counts["FeO"] == 1
    assert formula_counts["Li2O"] == 2  # duplicate
    assert formula_counts["LiFeO2"] == 1
    assert formula_counts["Li3FeO3"] == 1

    # Test with group_by="reduced_formula" (group formulas with same reduced form)
    fig_reduced = pmv.chem_sys_sunburst(systems, group_by="reduced_formula")
    assert isinstance(fig_reduced, go.Figure)

    # Get counts for each reduced formula
    labels = fig_reduced.data[0].labels
    parents = fig_reduced.data[0].parents
    values = fig_reduced.data[0].values  # noqa: PD011

    reduced_counts = {
        labels[idx]: values[idx]
        for idx in range(len(labels))
        if parents[idx] not in {"", "unary", "binary", "ternary"}  # only formula level
    }

    # Formulas with same reduced form should be grouped
    assert reduced_counts["Fe2O3"] == 2  # Fe2O3 and Fe4O6
    assert reduced_counts["FeO"] == 1
    assert reduced_counts["Li2O"] == 2  # duplicates
    assert reduced_counts["LiFeO2"] == 1
    assert reduced_counts["Li3FeO3"] == 1

    # Test with group_by="chem_sys" (default, group by chemical system)
    fig_system = pmv.chem_sys_sunburst(systems)  # default group_by="chem_sys"
    assert isinstance(fig_system, go.Figure)

    # Get counts for each chemical system
    labels = fig_system.data[0].labels
    parents = fig_system.data[0].parents
    values = fig_system.data[0].values  # noqa: PD011

    system_counts = {
        labels[idx]: values[idx]
        for idx in range(len(labels))
        if parents[idx] in {"binary", "ternary"}  # chemical system level
    }

    # Formulas with same elements should be grouped
    assert system_counts["Fe-O"] == 3  # Fe2O3, Fe4O6, FeO
    assert system_counts["Li-O"] == 2  # two Li2O
    assert system_counts["Fe-Li-O"] == 2  # LiFeO2, Li3FeO3


def test_chem_sys_sunburst_grouping_edge_cases() -> None:
    """Test chem_sys_sunburst grouping with edge cases."""
    systems = [
        # Different notations for the same system
        "Fe2O3",
        "Fe(III)2O3",  # with oxidation state
        "Fe2O3.0",  # with decimal
        "Fe2(O)3",  # with parentheses
        # Systems with fractional occupancies
        "Li0.5Na0.5O",  # mixed alkali
        "LiNaO2",  # same system, different notation
        # Complex formulas
        "(UO2)3(PO4)2",
        "U3P2O14",  # same system, different notation
    ]

    # Test with group_by="formula" (each notation counts separately)
    fig_formula = pmv.chem_sys_sunburst(systems, group_by="formula")
    assert isinstance(fig_formula, go.Figure)

    formula_counts = {
        label: val
        for idx, (label, val) in enumerate(
            zip(fig_formula.data[0].labels, fig_formula.data[0].values, strict=True)
        )
        if fig_formula.data[0].parents[idx]
        not in {"", "binary", "ternary", "quaternary"}
    }
    assert len(formula_counts) == len(systems), (
        "Each formula should be counted separately"
    )

    # Test with group_by="reduced_formula" (normalize notations)
    fig_reduced = pmv.chem_sys_sunburst(systems, group_by="reduced_formula")
    assert isinstance(fig_reduced, go.Figure)

    reduced_counts = {
        label: val
        for idx, (label, val) in enumerate(
            zip(fig_reduced.data[0].labels, fig_reduced.data[0].values, strict=True)
        )
        if fig_reduced.data[0].parents[idx]
        not in {"", "binary", "ternary", "quaternary"}
    }
    assert reduced_counts["Fe2O3"] == 3  # all Fe2O3 variations
    assert reduced_counts["NaLiO2"] == 1
    assert reduced_counts["U3(PO7)2"] == 2  # both uranium phosphate variations

    # Test with group_by="chem_sys" (default)
    fig_system = pmv.chem_sys_sunburst(systems)
    assert isinstance(fig_system, go.Figure)

    system_counts = {
        label: val
        for idx, (label, val) in enumerate(
            zip(fig_system.data[0].labels, fig_system.data[0].values, strict=True)
        )
        if fig_system.data[0].parents[idx] in {"binary", "ternary", "quaternary"}
    }

    # Check that different notations are properly grouped
    assert system_counts["Fe-O"] == 3  # all Fe2O3 variations
    assert system_counts["Li-Na-O"] == 2  # both mixed alkali variations
    assert system_counts["O-P-U"] == 2  # both uranium phosphate variations


@pytest.mark.parametrize(
    ("max_slices", "expected_systems", "expected_other"),
    [
        (None, 4, None),  # no limit
        (0, 4, None),  # treated as None
        (1, 2, ("Other (3 more not shown)", 3)),  # top 1 + Other
        (2, 3, ("Other (2 more not shown)", 2)),  # top 2 + Other
        (3, 4, ("Other (1 more not shown)", 1)),  # top 3 + Other
        (10, 4, None),  # greater than number of systems
        (-1, 4, None),  # negative treated as None
    ],
)
def test_chem_sys_sunburst_max_slices_edge_cases(
    max_slices: int | None,
    expected_systems: int,
    expected_other: tuple[str, int] | None,
) -> None:
    """Test chem_sys_sunburst with edge cases for max_slices."""
    systems = ["Fe2O3", "Li2O", "Na2O", "K2O"]
    fig = pmv.chem_sys_sunburst(systems, max_slices=max_slices, max_slices_mode="other")
    assert isinstance(fig, go.Figure)

    # Get all binary systems
    binary = {
        label: val
        for idx, (label, val) in enumerate(
            zip(fig.data[0].labels, fig.data[0].values, strict=True)
        )
        if fig.data[0].parents[idx] == "binary"
    }
    assert len(binary) == expected_systems

    if expected_other:
        other_label, other_count = expected_other
        other = next(label for label in binary if "Other" in label)
        assert other == other_label
        assert binary[other] == other_count
    else:
        assert not any("Other" in label for label in binary)


@pytest.mark.parametrize(
    ("max_slices_mode", "expected_systems", "expected_other"),
    [
        ("other", 3, ("Other (1 more not shown)", 1)),  # top 2 + Other
        ("drop", 2, None),  # only top 2
    ],
)
def test_chem_sys_sunburst_max_slices_mode(
    max_slices_mode: Literal["other", "drop"],
    expected_systems: int,
    expected_other: tuple[str, int] | None,
) -> None:
    """Test chem_sys_sunburst with different max_slices_mode values."""
    systems = ["Fe2O3", "Fe2O3", "Fe2O3", "Li2O", "Li2O", "Na2O"]
    fig = pmv.chem_sys_sunburst(systems, max_slices=2, max_slices_mode=max_slices_mode)
    assert isinstance(fig, go.Figure)

    binary = {
        fig.data[0].labels[idx]: fig.data[0].values[idx]  # noqa: PD011
        for idx in range(len(fig.data[0].labels))
        if fig.data[0].parents[idx] == "binary"
    }
    assert len(binary) == expected_systems
    assert binary["Fe-O"] == 3
    assert binary["Li-O"] == 2

    if expected_other:
        other_label, other_count = expected_other
        other = next(label for label in binary if "Other" in label)
        assert other == other_label
        assert binary[other] == other_count
    else:
        assert not any("Other" in label for label in binary)


def test_chem_sys_sunburst_max_slices_mode_invalid() -> None:
    """Test chem_sys_sunburst with invalid max_slices_mode."""
    systems = ["Fe2O3", "Li2O"]
    with pytest.raises(ValueError, match="Invalid max_slices_mode="):
        pmv.chem_sys_sunburst(systems, max_slices=1, max_slices_mode="invalid")  # type: ignore[arg-type]
