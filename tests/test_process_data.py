from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pymatgen.core import Composition

import pymatviz as pmv
from pymatviz.enums import ElemCountMode


if TYPE_CHECKING:
    from pymatviz.typing import FormulaGroupBy


@pytest.mark.parametrize(
    ("count_mode", "counts"),
    [
        (ElemCountMode.composition, {"Fe": 22, "O": 63, "P": 12}),
        (ElemCountMode.fractional_composition, {"Fe": 2.5, "O": 5, "P": 0.5}),
        (ElemCountMode.reduced_composition, {"Fe": 13, "O": 27, "P": 3}),
        (ElemCountMode.occurrence, {"Fe": 8, "O": 8, "P": 3}),
    ],
)
def test_count_elements(count_mode: ElemCountMode, counts: dict[str, float]) -> None:
    series = pmv.count_elements(
        ["Fe2 O3"] * 5 + ["Fe4 P4 O16"] * 3, count_mode=count_mode
    )
    expected = pd.Series(counts, index=pmv.df_ptable.index, name="count")
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_by_atomic_nums() -> None:
    series_in = pd.Series(1, index=range(1, 119))
    el_cts = pmv.count_elements(series_in)
    expected = pd.Series(1, index=pmv.df_ptable.index, name="count")

    pd.testing.assert_series_equal(expected, el_cts)


@pytest.mark.parametrize("range_limits", [(-1, 10), (100, 200)])
def test_count_elements_bad_atomic_nums(range_limits: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="assumed to represent atomic numbers"):
        pmv.count_elements(dict.fromkeys(range(*range_limits), 0))

    with pytest.raises(ValueError, match="assumed to represent atomic numbers"):
        # string and integer keys for atomic numbers should be handled equally
        pmv.count_elements({str(idx): 0 for idx in range(*range_limits)})


def test_count_elements_composition_objects() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = pmv.count_elements(compositions, count_mode=ElemCountMode.composition)
    expected = pd.Series(
        {"Fe": 22, "O": 63, "P": 12}, index=pmv.df_ptable.index, name="count"
    )
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_composition_objects_fractional() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = pmv.count_elements(
        compositions, count_mode=ElemCountMode.fractional_composition
    )
    expected = pd.Series(
        {"Fe": 2.5, "O": 5, "P": 0.5}, index=pmv.df_ptable.index, name="count"
    )
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_composition_objects_reduced() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = pmv.count_elements(
        compositions, count_mode=ElemCountMode.reduced_composition
    )
    expected = pd.Series(
        {"Fe": 13, "O": 27, "P": 3}, index=pmv.df_ptable.index, name="count"
    )
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_composition_objects_occurrence() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = pmv.count_elements(compositions, count_mode=ElemCountMode.occurrence)
    expected = pd.Series(
        {"Fe": 8, "O": 8, "P": 3}, index=pmv.df_ptable.index, name="count"
    )
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_mixed_input() -> None:
    mixed_input = ["Fe2O3", Composition("Fe4P4O16"), "LiCoO2", Composition("NaCl")]
    series = pmv.count_elements(mixed_input, count_mode=ElemCountMode.composition)
    expected = pd.Series(
        {"Fe": 6, "O": 21, "P": 4, "Li": 1, "Co": 1, "Na": 1, "Cl": 1},
        index=pmv.df_ptable.index,
        name="count",
    )
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_exclude_elements() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = pmv.count_elements(
        compositions, count_mode=ElemCountMode.composition, exclude_elements=["Fe", "P"]
    )
    expected = pd.Series(
        {"O": 63}, index=pmv.df_ptable.index.drop(["Fe", "P"]), name="count"
    )
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_exclude_invalid_elements() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    exclude_elements = ["Fe", "P", "Zz"]
    with pytest.raises(
        ValueError,
        match=re.escape(f"Unexpected symbol(s) Zz in {exclude_elements=}"),
    ):
        pmv.count_elements(
            compositions,
            count_mode=ElemCountMode.composition,
            exclude_elements=exclude_elements,
        )


def test_count_elements_fill_value() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = pmv.count_elements(compositions, count_mode=ElemCountMode.composition)
    expected = pd.Series(
        {"Fe": 22, "O": 63, "P": 12}, index=pmv.df_ptable.index, name="count"
    )
    pd.testing.assert_series_equal(series, expected, check_dtype=False)

    # Test previous default value (fill_value = 0)
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = pmv.count_elements(
        compositions, count_mode=ElemCountMode.composition, fill_value=0
    )
    expected = pd.Series(
        {"Fe": 22, "O": 63, "P": 12}, index=pmv.df_ptable.index, name="count"
    ).fillna(0)
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_formulas_basic() -> None:
    """Test basic functionality with formula strings."""
    data = ["Fe2O3", "Fe4O6", "FeO", "Li2O", "LiFeO2"]
    df_out = pmv.count_formulas(data)

    # Test DataFrame structure
    assert set(df_out.columns) == {"arity_name", "chem_sys", "count"}
    assert len(df_out) == 3  # 2 binary systems (Fe-O, Li-O) and 1 ternary (Li-Fe-O)

    # Test arity counts
    arity_counts = df_out.groupby("arity_name")["count"].sum()
    assert arity_counts["binary"] == 4  # Fe2O3, Fe4O6, FeO, Li2O
    assert arity_counts["ternary"] == 1  # LiFeO2


def test_count_formulas_empty() -> None:
    """Test handling of empty input."""
    with pytest.raises(ValueError, match="Empty input: data sequence is empty"):
        pmv.count_formulas([])


def test_count_formulas_invalid_formula() -> None:
    """Test handling of invalid formulas."""
    with pytest.raises(ValueError, match="Invalid formula"):
        pmv.count_formulas(["Fe2O3", "NotAFormula"])


def test_count_formulas_composition_objects() -> None:
    """Test handling of Composition objects."""
    data = [
        Composition("Fe2O3"),
        Composition("Fe4O6"),  # same as Fe2O3 when reduced
        Composition("FeO"),
        Composition("Li2O"),
    ]
    df_out = pmv.count_formulas(data, group_by="reduced_formula")

    # Should have 3 unique reduced formulas: Fe2O3 (2 entries), FeO, Li2O
    assert len(df_out) == 3
    assert df_out["count"].sum() == 4

    # Test that Fe2O3 and Fe4O6 are counted together
    fe_o_counts = df_out[df_out["formula"].str.contains("Fe")]
    assert len(fe_o_counts) == 2  # Fe2O3 and FeO
    assert fe_o_counts[fe_o_counts["formula"] == "Fe2O3"]["count"].iloc[0] == 2


@pytest.mark.parametrize(
    ("group_by", "expected_formulas", "expected_counts"),
    [
        (
            "formula",
            ["Fe2O3", "Fe4O6", "FeO"],  # all formulas kept separate
            [1, 1, 1],
        ),
        (
            "reduced_formula",
            ["Fe2O3", "FeO"],  # Fe4O6 -> Fe2O3
            [2, 1],
        ),
        (
            "chem_sys",
            ["Fe-O"],  # all Fe-O formulas grouped together
            [3],
        ),
    ],
)
def test_count_formulas_grouping_modes(
    group_by: FormulaGroupBy, expected_formulas: list[str], expected_counts: list[int]
) -> None:
    """Test different grouping modes."""
    data = ["Fe2O3", "Fe4O6", "FeO"]
    df_out = pmv.count_formulas(data, group_by=group_by)

    if group_by == "chem_sys":
        assert list(df_out["chem_sys"]) == expected_formulas
    else:
        assert list(df_out["formula"]) == expected_formulas
    assert list(df_out["count"]) == expected_counts


def test_count_formulas_mixed_input() -> None:
    """Test handling of mixed input types."""
    data = [
        "Fe2O3",
        Composition("Fe4O6"),
        "Fe-O",  # chemical system string
        Composition("FeO"),
    ]
    df_out = pmv.count_formulas(data, group_by="chem_sys")

    # All should be grouped into Fe-O system
    assert len(df_out) == 1
    assert df_out["chem_sys"].iloc[0] == "Fe-O"
    assert df_out["count"].iloc[0] == 4
