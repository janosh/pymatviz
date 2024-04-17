from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pymatviz.ptable_new import _data_preprocessor


class TestDataPreprocessor:
    test_dict: ClassVar = {"H": 1.0, "He": [2.0, 4.0], "Li": np.array([6.0, 8.0])}

    @staticmethod
    def _validate_output_df(output_df: pd.DataFrame) -> None:
        assert isinstance(output_df, pd.DataFrame)

        assert output_df.columns.tolist() == ["Value"]
        assert output_df.index.tolist() == ["H", "He", "Li"]

        assert_allclose(output_df.loc["H", "Value"], [1.0])
        assert_allclose(output_df.loc["He", "Value"], [2.0, 4.0])
        assert_allclose(output_df.loc["Li", "Value"], [6.0, 8.0])

        assert output_df.attrs["vmin"] == 1.0
        assert output_df.attrs["vmax"] == 8.0

    def test_with_pd_dataframe(self) -> None:
        input_df: pd.DataFrame = pd.DataFrame(
            self.test_dict.items(), columns=["Element", "Value"]
        ).set_index("Element")

        output_df: pd.DataFrame = _data_preprocessor(input_df)

        self._validate_output_df(output_df)

    def test_with_pd_series(self) -> None:
        input_series: pd.Series = pd.Series(self.test_dict)

        output_df = _data_preprocessor(input_series)

        self._validate_output_df(output_df)

    def test_with_dict(self) -> None:
        input_dict = self.test_dict

        output_df = _data_preprocessor(input_dict)

        self._validate_output_df(output_df)

    def test_unsupported_type(self) -> None:
        invalid_data = [0, 1, 2]

        with pytest.raises(TypeError, match="Unsupported data type"):
            _data_preprocessor(invalid_data)

    def test_missing_imputate(self) -> None:
        pass

    def test_anomaly_handle(self) -> None:
        pass
