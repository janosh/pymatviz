"""Tests for pymatviz.utils: package paths, element lookups and df_to_arrays."""

from __future__ import annotations

import os
from typing import Any

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


DF_REGR = pd.DataFrame({"x": y_true, "y": y_pred})


def test_df_to_arrays() -> None:
    """Columns or arrays give the same data, a list of columns gives a DataFrame."""
    for arrays in (df_to_arrays(None, y_true, y_pred), df_to_arrays(DF_REGR, "x", "y")):
        np.testing.assert_array_equal(arrays, [y_true, y_pred])
    _, y_cols = df_to_arrays(DF_REGR, "x", ["y", "x"])
    assert list(y_cols) == ["y", "x"]
    np.testing.assert_array_equal(y_cols["y"], y_pred)


@pytest.mark.parametrize(
    ("df", "args", "err_cls", "match"),
    [
        ("foo", (y_true, y_pred), TypeError, "df should be pandas DataFrame or None"),
        (DF_REGR, ("not-a-col", "x"), KeyError, "not-a-col"),
        (None, ("x", y_pred), ValueError, "got column names but no df"),
        (DF_REGR, ("x", y_pred), ValueError, "with a df, pass column names"),
        (DF_REGR, ("x", [y_pred]), ValueError, "with a df, pass column names"),
    ],
)
def test_df_to_arrays_raises(
    df: Any, args: tuple[Any, ...], err_cls: type[Exception], match: str
) -> None:
    """Non-DataFrame df, unknown columns and mixing names with arrays raise."""
    with pytest.raises(err_cls, match=match):
        df_to_arrays(df, *args)
