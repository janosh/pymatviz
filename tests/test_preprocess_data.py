from __future__ import annotations

import re
from typing import TYPE_CHECKING, get_args

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pymatviz._preprocess_data import (
    SupportedDataType,
    check_for_missing_inf,
    get_df_nest_level,
    preprocess_ptable_data,
    replace_missing_and_infinity,
)
from pymatviz.enums import Key


if TYPE_CHECKING:
    from typing import ClassVar


class TestPreprocessPtableData:
    test_dict: ClassVar = {
        "H": 1,  # int
        "He": [2.0, 4.0],  # float list
        "Li": np.array([6.0, 8.0]),  # float array
        "Na": 11.0,  # float
        "Mg": {"a": -1, "b": 14.0}.values(),  # dict_values
        "Al": {-1, 2.3},  # mixed int/float set
    }

    @staticmethod
    def _validate_output_df(output_df: pd.DataFrame) -> None:
        assert isinstance(output_df, pd.DataFrame)

        assert list(output_df) == [Key.heat_val]
        assert list(output_df.index) == ["H", "He", "Li", "Na", "Mg", "Al"]

        assert_allclose(output_df.loc["H", Key.heat_val], [1.0])
        assert_allclose(output_df.loc["He", Key.heat_val], [2.0, 4.0])
        assert_allclose(output_df.loc["Li", Key.heat_val], [6.0, 8.0])
        assert_allclose(output_df.loc["Na", Key.heat_val], [11.0])
        assert_allclose(output_df.loc["Mg", Key.heat_val], [-1.0, 14.0])

        assert output_df.attrs["vmin"] == -1.0
        assert output_df.attrs["mean"] == 4.63
        assert output_df.attrs["vmax"] == 14.0

    def test_from_pd_dataframe(self) -> None:
        input_df: pd.DataFrame = pd.DataFrame(
            self.test_dict.items(), columns=[Key.element, Key.heat_val]
        ).set_index(Key.element)

        output_df: pd.DataFrame = preprocess_ptable_data(input_df)

        self._validate_output_df(output_df)

    def test_from_bad_pd_dataframe(self) -> None:
        """Test auto-fix of badly formatted pd.DataFrame."""
        test_dict = {
            "He": [2.0, 4.0],  # float list
            "Li": np.array([6.0, 8.0]),  # float array
            "Mg": {"a": -1, "b": 14.0}.values(),  # dict_values
        }

        input_df_0 = pd.DataFrame(test_dict)

        # Elements as a row, and no proper row/column names
        output_df_0 = preprocess_ptable_data(input_df_0)

        assert_allclose(output_df_0.loc["He", Key.heat_val], [2.0, 4.0])
        assert_allclose(output_df_0.loc["Li", Key.heat_val], [6.0, 8.0])
        assert_allclose(output_df_0.loc["Mg", Key.heat_val], [-1.0, 14.0])

        # Elements as a column, and no proper row/column names
        input_df_1 = input_df_0.copy().transpose()
        output_df_1 = preprocess_ptable_data(input_df_1)

        assert_allclose(output_df_1.loc["He", Key.heat_val], [2.0, 4.0])
        assert_allclose(output_df_1.loc["Li", Key.heat_val], [6.0, 8.0])
        assert_allclose(output_df_1.loc["Mg", Key.heat_val], [-1.0, 14.0])

        # No elements column/row at all
        df_without_complet_elem = pd.DataFrame(
            {
                "Hellow": [2.0, 4.0],  # Not an element
                "Li": np.array([6.0, 8.0]),
                "Mg": {"a": -1, "b": 14.0}.values(),
            }
        )
        with pytest.raises(ValueError, match="Cannot normalize the given DataFrame"):
            preprocess_ptable_data(df_without_complet_elem)

    def test_from_pd_series(self) -> None:
        input_series: pd.Series = pd.Series(self.test_dict)

        output_df = preprocess_ptable_data(input_series)

        self._validate_output_df(output_df)

    def test_from_dict(self) -> None:
        input_dict = self.test_dict

        output_df = preprocess_ptable_data(input_dict)

        self._validate_output_df(output_df)

    def test_unsupported_type(self) -> None:
        for invalid_data in ([0, 1, 2], range(5), "test", None):
            err_msg = (
                f"{type(invalid_data).__name__} unsupported, "
                f"choose from {get_args(SupportedDataType)}"
            )
            with pytest.raises(TypeError, match=re.escape(err_msg)):
                preprocess_ptable_data(invalid_data)

    def test_get_vmin_vmax(self) -> None:
        # Test without nested list/array
        test_dict_0 = {"H": 1, "He": [2, 4], "Li": np.array([6, 8])}

        output_df_0 = preprocess_ptable_data(test_dict_0)

        assert output_df_0.attrs["vmin"] == 1
        assert output_df_0.attrs["vmax"] == 8

        # Test with nested list/array
        test_dict_1 = {
            "H": 1,
            "He": [[2, 3], [4, 5]],
            "Li": [np.array([6, 7]), np.array([8, 9])],
        }

        output_df_1 = preprocess_ptable_data(test_dict_1)

        assert output_df_1.attrs["vmin"] == 1
        assert output_df_1.attrs["vmax"] == 9


def test_check_for_missing_inf() -> None:
    # Test a normal DataFrame
    normal_df = pd.DataFrame(
        {
            "Fe": [1, 2, 3],
            "O": [4, 5, 6],
        }.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert check_for_missing_inf(normal_df, Key.heat_val) == (False, False)

    # Test DataFrame with missing value (NaN)
    df_with_missing = pd.DataFrame(
        {
            "Fe": [1, 2, np.nan],
            "O": [4, 5, 6],
        }.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert check_for_missing_inf(df_with_missing, Key.heat_val) == (True, False)

    # Test DataFrame with infinity
    df_with_inf = pd.DataFrame(
        {
            "Fe": [1, 2, np.inf],
            "O": [4, 5, 6],
        }.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert check_for_missing_inf(df_with_inf, Key.heat_val) == (False, True)

    # Test DataFrame with missing value (NaN) and infinity
    df_with_nan_inf = pd.DataFrame(
        {
            "Fe": [1, 2, np.inf],
            "O": [4, 5, np.nan],
        }.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert check_for_missing_inf(df_with_nan_inf, Key.heat_val) == (True, True)


def test_get_df_nest_level() -> None:
    # Test nest level 0
    df_level_0 = pd.DataFrame(
        {
            "Fe": 1,
            "O": 2,
        }.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert get_df_nest_level(df_level_0, Key.heat_val) == 0

    # Test nest level 1
    df_level_1 = pd.DataFrame(
        {
            "Fe": [1, 2, 3],
            "O": [4, 5, 6],
        }.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert get_df_nest_level(df_level_1, Key.heat_val) == 1

    df_level_1_arr = pd.DataFrame(
        {
            "Fe": 1,
            "O": np.array([4, 5, 6]),
        }.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert get_df_nest_level(df_level_1_arr, Key.heat_val) == 1

    # Test nest level 2
    df_level_2 = pd.DataFrame(
        {
            "Fe": [1, 2, 3],
            "O": [[4, 5], [6, 7]],  # get max level
        }.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert get_df_nest_level(df_level_2, Key.heat_val) == 2

    df_level_2_arr = pd.DataFrame(
        {
            "Fe": [1, 2, 3],
            "O": np.array([[4, 5], [6, 7]]),  # get max level
        }.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert get_df_nest_level(df_level_2_arr, Key.heat_val) == 2


class TestReplaceMissingAndInfinity:
    def test_replace_missing(self) -> None:
        df_with_nan = pd.DataFrame(
            {
                "Fe": [1, 2, 3],
                "O": [4, 5, np.nan],
            }.items(),
            columns=[Key.element, Key.heat_val],
        ).set_index(Key.element)

        # Test missing strategy: mean (default)
        with pytest.warns(match="NaN found in data"):
            processed_df_mean = replace_missing_and_infinity(
                df_with_nan.copy(), Key.heat_val
            )
        assert_allclose(processed_df_mean.loc["O", Key.heat_val], [4, 5, 3])

        # Test missing strategy: zero
        with pytest.warns(match="NaN found in data"):
            processed_df_zero = replace_missing_and_infinity(
                df_with_nan.copy(), Key.heat_val, "zero"
            )
        assert_allclose(processed_df_zero.loc["O", Key.heat_val], [4, 5, 0])

    def test_replace_infinity(self) -> None:
        df_with_inf = pd.DataFrame(
            {
                "Fe": [1, 2, np.inf],
                "O": [4, 5, 6],
            }.items(),
            columns=[Key.element, Key.heat_val],
        ).set_index(Key.element)

        with pytest.warns(match="Infinity found in data"):
            processed_df = replace_missing_and_infinity(df_with_inf, Key.heat_val)
        assert_allclose(processed_df.loc["Fe", Key.heat_val], [1, 2, 6])

    def test_replace_both(self) -> None:
        df_with_both = pd.DataFrame(
            {
                "Fe": [1, 2, np.inf],
                "O": [4, 5, np.nan],
            }.items(),
            columns=[Key.element, Key.heat_val],
        ).set_index(Key.element)

        processed_df = replace_missing_and_infinity(df_with_both, Key.heat_val, "zero")
        assert_allclose(processed_df.loc["Fe", Key.heat_val], [1, 2, 5])
        assert_allclose(processed_df.loc["O", Key.heat_val], [4, 5, 0])

    def test_too_deep_nest(self) -> None:
        df_level_2 = pd.DataFrame(
            {
                "Fe": [1, 2, 3],
                "O": [[4, 5], [6, np.nan]],
            }.items(),
            columns=[Key.element, Key.heat_val],
        ).set_index(Key.element)

        with pytest.raises(RuntimeError, match="Unable to replace NaN and inf"):
            replace_missing_and_infinity(df_level_2, col=Key.heat_val)
