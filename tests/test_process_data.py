import re

import pandas as pd
import pytest
from pymatgen.core import Composition

from pymatviz.enums import ElemCountMode
from pymatviz.process_data import count_elements
from pymatviz.utils import df_ptable


@pytest.mark.parametrize(
    "count_mode, counts",
    [
        (ElemCountMode.composition, {"Fe": 22, "O": 63, "P": 12}),
        (ElemCountMode.fractional_composition, {"Fe": 2.5, "O": 5, "P": 0.5}),
        (ElemCountMode.reduced_composition, {"Fe": 13, "O": 27, "P": 3}),
        (ElemCountMode.occurrence, {"Fe": 8, "O": 8, "P": 3}),
    ],
)
def test_count_elements(count_mode: ElemCountMode, counts: dict[str, float]) -> None:
    series = count_elements(["Fe2 O3"] * 5 + ["Fe4 P4 O16"] * 3, count_mode=count_mode)
    expected = pd.Series(counts, index=df_ptable.index, name="count").fillna(0)
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_by_atomic_nums() -> None:
    series_in = pd.Series(1, index=range(1, 119))
    el_cts = count_elements(series_in)
    expected = pd.Series(1, index=df_ptable.index, name="count")

    pd.testing.assert_series_equal(expected, el_cts)


@pytest.mark.parametrize("range_limits", [(-1, 10), (100, 200)])
def test_count_elements_bad_atomic_nums(range_limits: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="assumed to represent atomic numbers"):
        count_elements(dict.fromkeys(range(*range_limits), 0))

    with pytest.raises(ValueError, match="assumed to represent atomic numbers"):
        # string and integer keys for atomic numbers should be handled equally
        count_elements({str(idx): 0 for idx in range(*range_limits)})


def test_count_elements_composition_objects() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = count_elements(compositions, count_mode=ElemCountMode.composition)
    expected = pd.Series(
        {"Fe": 22, "O": 63, "P": 12}, index=df_ptable.index, name="count"
    ).fillna(0)
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_composition_objects_fractional() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = count_elements(
        compositions, count_mode=ElemCountMode.fractional_composition
    )
    expected = pd.Series(
        {"Fe": 2.5, "O": 5, "P": 0.5}, index=df_ptable.index, name="count"
    ).fillna(0)
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_composition_objects_reduced() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = count_elements(compositions, count_mode=ElemCountMode.reduced_composition)
    expected = pd.Series(
        {"Fe": 13, "O": 27, "P": 3}, index=df_ptable.index, name="count"
    ).fillna(0)
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_composition_objects_occurrence() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = count_elements(compositions, count_mode=ElemCountMode.occurrence)
    expected = pd.Series(
        {"Fe": 8, "O": 8, "P": 3}, index=df_ptable.index, name="count"
    ).fillna(0)
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_mixed_input() -> None:
    mixed_input = ["Fe2O3", Composition("Fe4P4O16"), "LiCoO2", Composition("NaCl")]
    series = count_elements(mixed_input, count_mode=ElemCountMode.composition)
    expected = pd.Series(
        {"Fe": 6, "O": 21, "P": 4, "Li": 1, "Co": 1, "Na": 1, "Cl": 1},
        index=df_ptable.index,
        name="count",
    ).fillna(0)
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_exclude_elements() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = count_elements(
        compositions, count_mode=ElemCountMode.composition, exclude_elements=["Fe", "P"]
    )
    expected = pd.Series(
        {"O": 63}, index=df_ptable.index.drop(["Fe", "P"]), name="count"
    ).fillna(0)
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_exclude_invalid_elements() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    exclude_elements = ["Fe", "P", "Zz"]
    with pytest.raises(
        ValueError,
        match=re.escape(f"Unexpected symbol(s) Zz in {exclude_elements=}"),
    ):
        count_elements(
            compositions,
            count_mode=ElemCountMode.composition,
            exclude_elements=exclude_elements,
        )


def test_count_elements_fill_value() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = count_elements(
        compositions, count_mode=ElemCountMode.composition, fill_value=None
    )
    expected = pd.Series(
        {"Fe": 22, "O": 63, "P": 12}, index=df_ptable.index, name="count"
    )
    pd.testing.assert_series_equal(series, expected, check_dtype=False)
