"""Tests for pymatviz.utils: package paths, element lookups, align_series and
df_to_arrays.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pymatviz.utils import PKG_DIR, ROOT, align_series, df_to_arrays
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


def test_align_series() -> None:
    """Series pair by index label (inner join, first Series' order), else pass."""
    xs = pd.Series([1.0, 2.0, 3.0], index=list("abc"), name="x")
    ys = pd.Series([30.0, 10.0, 20.0, 99.0], index=list("cabz"), name="y")
    x_al, y_al = df_to_arrays(None, xs, ys)
    np.testing.assert_array_equal(x_al, [1, 2, 3])
    np.testing.assert_array_equal(y_al, [10, 20, 30])  # pairs (1,10) (2,20) (3,30)
    # Series in a Mapping (e.g. named predictions) align too, other values pass
    x_al, preds, label = align_series(xs, {"m1": ys, "m2": ys[::-1]}, "label")
    assert label == "label"
    assert list(x_al.index) == list(preds["m1"].index) == list(preds["m2"].index)
    assert list(preds["m2"]) == [10, 20, 30]
    # equal indexes and fewer than 2 Series are no-ops, even beside plain arrays
    same = [xs, xs * 2, [7, 8, 9]]
    assert all(out is val for out, val in zip(align_series(*same), same, strict=True))
    np.testing.assert_array_equal(df_to_arrays(None, ys, [1, 2, 3, 4])[1], [1, 2, 3, 4])


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (
            (pd.Series([1], index=["a"]), pd.Series([2], index=["b"])),
            "share no index labels",
        ),
        (
            (pd.Series([1, 2], index=["a", "a"]), pd.Series([2, 3], index=["a", "b"])),
            r"must be unique, got duplicates \['a'\]",
        ),
        (
            (pd.Series([1, 2]), pd.Series([2, 3], index=[1, 0]), [4, 5]),
            r"against unlabeled \['list'\]",
        ),
        (
            (
                pd.Series([1, 2]),
                {"m1": pd.Series([2, 3], [1, 0]), "m2": {"probs_positive": [4, 5]}},
            ),
            r"against unlabeled \['dict'\]",
        ),
    ],
)
def test_align_series_raises(values: tuple[Any, ...], match: str) -> None:
    """Disjoint or duplicate indexes and plain arrays beside misaligned Series raise."""
    with pytest.raises(ValueError, match=match):
        align_series(*values)
