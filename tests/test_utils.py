"""Tests for pymatviz.utils: package paths, element lookups and df_to_arrays."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pymatviz.utils import PKG_DIR, ROOT, df_to_arrays
from tests.conftest import y_pred, y_true


def test_dir_globals() -> None:
    assert os.path.isdir(PKG_DIR)
    assert os.path.isdir(ROOT)

    assert os.path.dirname(PKG_DIR) == ROOT
    assert set(os.listdir(ROOT)) >= {"examples", "pymatviz", "tests", "assets"}


def test_df_to_arrays() -> None:
    """Columns or arrays give the same data, and bad df or column names raise."""
    df_regr = pd.DataFrame({"x": y_true, "y": y_pred})
    for arrays in (df_to_arrays(None, y_true, y_pred), df_to_arrays(df_regr, "x", "y")):
        np.testing.assert_allclose(arrays, [y_true, y_pred], rtol=0, atol=0)
    with pytest.raises(TypeError, match="df should be pandas DataFrame or None"):
        df_to_arrays("foo", y_true, y_pred)  # ty: ignore[invalid-argument-type]
    with pytest.raises(KeyError, match="not-a-col"):
        df_to_arrays(df_regr, "not-a-col", "x")
    with pytest.raises(ValueError, match="got column names but no df"):
        df_to_arrays(None, "x", y_pred)
    with pytest.raises(ValueError, match="with a df, pass column names"):
        df_to_arrays(df_regr, "x", y_pred)
    with pytest.raises(ValueError, match="with a df, pass column names"):
        df_to_arrays(df_regr, "x", [y_pred])
    _, y_cols = df_to_arrays(df_regr, "x", ["y", "x"])
    assert list(y_cols) == ["y", "x"]
    np.testing.assert_array_equal(y_cols["y"], y_pred)
