"""Tests for element and formula counting."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pymatgen.core import Composition

import pymatviz as pmv
from pymatviz.composition import count_elements, count_formulas
from pymatviz.enums import ElemCountMode


if TYPE_CHECKING:
    from typing import Any

    from pymatviz.typing import FormulaGroupBy


FE_P_O_COMPS = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3


@pytest.mark.parametrize(
    "inputs",
    [["Fe2 O3"] * 5 + ["Fe4 P4 O16"] * 3, FE_P_O_COMPS],
    ids=["strings", "compositions"],
)
@pytest.mark.parametrize(
    ("count_mode", "counts"),
    [
        (ElemCountMode.composition, {"Fe": 22, "O": 63, "P": 12}),
        (ElemCountMode.fractional_composition, {"Fe": 2.5, "O": 5, "P": 0.5}),
        (ElemCountMode.reduced_composition, {"Fe": 13, "O": 27, "P": 3}),
        (ElemCountMode.occurrence, {"Fe": 8, "O": 8, "P": 3}),
    ],
)
def test_count_elements(
    inputs: list[str] | list[Composition],
    count_mode: ElemCountMode,
    counts: dict[str, float],
) -> None:
    series = count_elements(inputs, count_mode=count_mode)
    expected = pd.Series(counts, index=pmv.utils.df_ptable.index, name="count")
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


@pytest.mark.parametrize(
    "count_mode",
    [ElemCountMode.composition, ElemCountMode.fractional_composition],
)
def test_count_elements_exact_sums(count_mode: ElemCountMode) -> None:
    """Element sums over many formulas are exactly rounded (math.fsum)."""
    formulas = ["LiFePO4", "Ba0.5Sr0.5TiO3", "Fe2O3", "NaCl"] * 251
    attr = "element_composition" if count_mode == "composition" else count_mode
    amounts: dict[str, list[float]] = {}
    for formula in formulas:
        for elem, amt in getattr(Composition(formula), attr).as_dict().items():
            amounts.setdefault(elem, []).append(amt)
    expected = {elem: math.fsum(vals) for elem, vals in amounts.items()}
    counts = count_elements(formulas, count_mode=count_mode)
    assert counts.dropna().to_dict() == expected
    assert counts.dtype == float
    assert counts.index.equals(pmv.utils.df_ptable.index)


@pytest.mark.parametrize("index_type", [int, float, str])
def test_count_elements_by_atomic_nums(index_type: type) -> None:
    """Accept integer atomic numbers without changing counts."""
    series_in = pd.Series(1, index=list(map(index_type, range(1, 119))))
    el_cts = count_elements(series_in)
    expected = pd.Series(1, index=pmv.utils.df_ptable.index, name="count")

    pd.testing.assert_series_equal(expected, el_cts)


@pytest.mark.parametrize(
    ("values", "kwargs", "match"),
    [
        ({"Fe": 1, "Zz": 2}, {}, r"Unexpected element symbol\(s\): Zz"),
        ({1.5: 4}, {}, "Atomic numbers must be integers"),
        ({1.5: 4, "2": 3}, {}, "Atomic numbers must be integers"),
        *[
            (values, {}, "assumed to represent atomic numbers")
            for start, stop in [(-1, 10), (100, 200)]
            for values in (
                dict.fromkeys(range(start, stop), 0),
                {str(idx): 0 for idx in range(start, stop)},
            )
        ],
        (
            ["Fe2O3", "Fe4P4O16"],
            {"exclude_elements": ["Fe", "P", "Zz"]},
            re.escape("Unexpected symbol(s) Zz in excluded_elements=['Fe', 'P', 'Zz']"),
        ),
    ],
)
def test_count_elements_raises(values: Any, kwargs: dict[str, Any], match: str) -> None:
    """Reject unknown symbols, non-integral or out-of-range atomic numbers."""
    with pytest.raises(ValueError, match=match):
        count_elements(values, **kwargs)


@pytest.mark.parametrize(
    ("values", "kwargs", "counts", "dropped"),
    [
        (
            ["Fe2O3", Composition("Fe4P4O16"), "LiCoO2", Composition("NaCl")],
            {},
            {"Fe": 6, "O": 21, "P": 4, "Li": 1, "Co": 1, "Na": 1, "Cl": 1},
            [],
        ),
        (FE_P_O_COMPS, {"exclude_elements": ["Fe", "P"]}, {"O": 63}, ["Fe", "P"]),
        (FE_P_O_COMPS, {"exclude_elements": "Fe"}, {"O": 63, "P": 12}, ["Fe"]),
    ],
    ids=["mixed-input", "exclude-list", "exclude-str"],
)
@pytest.mark.parametrize("fill_value", [None, 0])
def test_count_elements_options(
    values: list[str | Composition],
    kwargs: dict[str, Any],
    counts: dict[str, int],
    dropped: list[str],
    fill_value: float | None,
) -> None:
    """Count mixed inputs, drop excluded elements and fill missing ones."""
    series = count_elements(values, fill_value=fill_value, **kwargs)
    expected = pd.Series(
        counts, index=pmv.utils.df_ptable.index.drop(dropped), name="count"
    )
    if fill_value is not None:
        expected = expected.fillna(fill_value)
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            ["Fe2O3", "Fe4O6", "FeO", "Li2O", "LiFeO2"],
            [("binary", "Fe-O", 3), ("binary", "Li-O", 1), ("ternary", "Fe-Li-O", 1)],
        ),
        (  # sorted by arity before name
            ["Li2O", "Fe2O3", "NaCl", "C", "Li-Fe-O"],
            [
                ("unary", "C", 1),
                ("binary", "Cl-Na", 1),
                ("binary", "Fe-O", 1),
                ("binary", "Li-O", 1),
                ("ternary", "Fe-Li-O", 1),
            ],
        ),
        (
            ["Fe2O3", Composition("Fe4O6"), "Fe-O", Composition("FeO")],
            [("binary", "Fe-O", 4)],
        ),
    ],
    ids=["formulas", "arity-order", "mixed-input"],
)
def test_count_formulas_chem_sys(
    data: list[str | Composition], expected: list[tuple[str, str, int]]
) -> None:
    """Group formulas, chemical systems and compositions by sorted chemical system."""
    df_out = count_formulas(data)
    assert list(df_out.columns) == ["arity_name", "chem_sys", "count"]
    assert list(df_out.itertuples(index=False, name=None)) == expected


@pytest.mark.parametrize(
    ("data", "group_by", "error_match"),
    [
        ([], "chem_sys", "Empty input: data sequence is empty"),
        (["Fe2O3", "NotAFormula"], "chem_sys", "Invalid formula"),
        (["Fe-Zz"], "chem_sys", "Invalid elements in system"),
        (["Fe2O3"], "invalid", "Invalid group_by="),
    ],
)
def test_count_formulas_raises(data: list, group_by: str, error_match: str) -> None:
    """Reject empty data, invalid formulas/elements and unknown group_by."""
    with pytest.raises(ValueError, match=error_match):
        count_formulas(data, group_by=group_by)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("group_by", "expected"),
    [
        ("formula", {"Fe2O3": 1, "Fe4O6": 1, "FeO": 1}),
        ("reduced_formula", {"Fe2O3": 2, "FeO": 1}),
        ("chem_sys", {"Fe-O": 3}),
    ],
)
@pytest.mark.parametrize("as_comps", [False, True])
def test_count_formulas_grouping_modes(
    group_by: FormulaGroupBy, expected: dict[str, int], as_comps: bool
) -> None:
    """Formulas count separately, by reduced formula or by chemical system."""
    data = ["Fe2O3", "Fe4O6", "FeO"]
    df_out = count_formulas(
        [Composition(formula) for formula in data] if as_comps else data,
        group_by=group_by,
    )
    label_col = "chem_sys" if group_by == "chem_sys" else "formula"
    if as_comps and group_by == "formula":  # str(Composition) spells out amounts
        expected = {"Fe2 O3": 1, "Fe4 O6": 1, "Fe1 O1": 1}
    assert dict(zip(df_out[label_col], df_out["count"], strict=True)) == expected
