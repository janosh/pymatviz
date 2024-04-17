from __future__ import annotations

import pandas as pd

from pymatviz.ptable_new import _data_preprocessor


class TestDataPreprocessor:
    test_dict = {"H": 1.0, "He": [2.0, 4.0]}

    @staticmethod
    def _validate_output_df(output_df):
        assert isinstance(output_df, pd.DataFrame)

        assert output_df.columns.tolist() == ["Value"]
        assert output_df.index.tolist() == ["H", "He"]

        assert output_df.at["H", "Value"] == 1.0
        assert output_df.at["He", "Value"] == [2.0, 4.0]

        assert output_df.attrs["vmin"] == 1.0
        assert output_df.attrs["vmax"] == 4.0

    def test_with_pd_dataframe(self):
        input_df = pd.DataFrame(
            self.test_dict.items(), columns=["Element", "Value"]
        ).set_index("Element")

        output_df = _data_preprocessor(input_df)

        self._validate_output_df(output_df)

    def test_with_pd_series(self):
        pass

    def test_with_dict(self):
        output_df = _data_preprocessor(self.test_dict)

        self._validate_output_df(output_df)

    def test_unsupported_type(self):
        pass

    def test_missing_imputate(self):
        pass

    def test_anomaly_handle(self):
        pass
