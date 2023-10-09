from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.offsetbox import AnchoredText

from pymatviz.utils import (
    CrystalSystem,
    add_identity_line,
    annotate_metrics,
    bin_df_cols,
    df_to_arrays,
    get_crystal_sys,
    patch_dict,
)
from tests.conftest import y_pred, y_true


if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.mark.parametrize(
    "metrics, fmt",
    [
        ("MSE", ".1"),
        (["RMSE"], ".1"),
        (("MAPE", "MSE"), ".2"),
        ({"MAE", "R2", "RMSE"}, ".3"),
        ({"MAE": 1.4, "$R^2$": 0.2, "RMSE": 1.9}, ".0"),
    ],
)
def test_annotate_metrics(metrics: dict[str, float] | Sequence[str], fmt: str) -> None:
    text_box = annotate_metrics(y_pred, y_true, metrics=metrics, fmt=fmt)

    assert isinstance(text_box, AnchoredText)

    expected = dict(MAE=0.113, R2=0.765, RMSE=0.144, MAPE=0.5900, MSE=0.0206)

    txt = ""
    if isinstance(metrics, dict):
        for key, val in metrics.items():
            txt += f"{key} = {val:{fmt}}\n"
    else:
        for key in [metrics] if isinstance(metrics, str) else metrics:
            txt += f"{key} = {expected[key]:{fmt}}\n"

    assert text_box.txt.get_text() == txt

    prefix, suffix = "Metrics:\n", "\nthe end"
    text_box = annotate_metrics(
        y_pred, y_true, metrics=metrics, fmt=fmt, prefix=prefix, suffix=suffix
    )
    assert text_box.txt.get_text() == prefix + txt + suffix


@pytest.mark.parametrize("metrics", [42, datetime.now()])
def test_annotate_metrics_raises(metrics: Any) -> None:
    with pytest.raises(
        TypeError, match=f"metrics must be dict|list|tuple|set, not {type(metrics)}"
    ):
        annotate_metrics(y_pred, y_true, metrics=metrics)


@pytest.mark.parametrize(
    "spg_num, crystal_sys",
    [
        (1, "triclinic"),
        (15, "monoclinic"),
        (16, "orthorhombic"),
        (75, "tetragonal"),
        (143, "trigonal"),
        (168, "hexagonal"),
        (230, "cubic"),
    ],
)
def test_get_crystal_sys(spg_num: int, crystal_sys: CrystalSystem) -> None:
    assert crystal_sys == get_crystal_sys(spg_num)


@pytest.mark.parametrize("spg", [-1, 0, 231])
def test_get_crystal_sys_invalid(spg: int) -> None:
    with pytest.raises(ValueError, match=f"Invalid space group {spg}"):
        get_crystal_sys(spg)


@pytest.mark.parametrize("trace_idx", [0, 1])
@pytest.mark.parametrize("line_kwds", [None, {"color": "blue"}])
def test_add_identity_line(
    plotly_scatter: go.Figure, trace_idx: int, line_kwds: dict[str, str] | None
) -> None:
    fig = add_identity_line(plotly_scatter, line_kwds=line_kwds, trace_idx=trace_idx)
    assert isinstance(fig, go.Figure)

    line = next(shape for shape in fig.layout["shapes"] if shape["type"] == "line")
    assert line["x0"] == line["y0"]  # fails if we don't handle nan since nan != nan
    assert line["x1"] == line["y1"]
    assert line["layer"] == "below"
    assert line["line"]["color"] == line_kwds["color"] if line_kwds else "gray"


def test_df_to_arrays() -> None:
    df = pd.DataFrame([y_true, y_pred]).T
    x1, y1 = df_to_arrays(None, y_true, y_pred)
    x_col, y_col = df.columns[:2]
    x2, y2 = df_to_arrays(df, x_col, y_col)
    # TODO find a mypy-compat way to check for exact equality
    assert x1 == pytest.approx(x2)
    assert y1 == pytest.approx(y2)
    assert x1 == pytest.approx(y_true)
    assert y1 == pytest.approx(y_pred)

    with pytest.raises(TypeError, match="df should be pandas DataFrame or None"):
        df_to_arrays("foo", y_true, y_pred)

    bad_col_name = "not-real-col-name"
    with pytest.raises(KeyError) as exc:
        df_to_arrays(df, bad_col_name, df.columns[0])

    assert "not-real-col-name" in str(exc.value)


@pytest.mark.parametrize("strict", [True, False])
def test_df_to_arrays_strict(strict: bool) -> None:
    try:
        args = df_to_arrays(42, "foo", "bar", strict=strict)
    except TypeError as exc:
        if strict:
            assert "df should be pandas DataFrame or None" in str(exc)  # noqa: PT017
        else:
            assert args == ("foo", "bar")


@pytest.mark.parametrize(
    "bin_by_cols, group_by_cols, n_bins, expected_n_bins, "
    "verbose, kde_col, expected_n_rows",
    [
        (["A"], [], 2, [2], True, "", 2),
        (["A", "B"], [], 2, [2, 2], True, "kde", 4),
        (["A", "B"], [], [2, 3], [2, 3], False, "kde", 5),
        (["A"], ["B"], 2, [2], False, "", 30),
    ],
)
def test_bin_df_cols(
    bin_by_cols: list[str],
    group_by_cols: list[str],
    n_bins: int | list[int],
    expected_n_bins: list[int],
    verbose: bool,
    kde_col: str,
    expected_n_rows: int,
) -> None:
    df: pd.DataFrame = pd._testing.makeDataFrame()  # random data
    idx_col = "index"
    df.index.name = idx_col
    bin_counts_col = "bin_counts"
    df_binned = bin_df_cols(
        df,
        bin_by_cols,
        group_by_cols,
        n_bins,
        verbose=verbose,
        bin_counts_col=bin_counts_col,
        kde_col=kde_col,
    )

    # ensure binned DataFrame has a minimum set of expected columns
    expected_cols = {bin_counts_col, *df, *(f"{col}_bins" for col in bin_by_cols)}
    assert {*df_binned} >= expected_cols
    assert len(df_binned) == expected_n_rows

    # validate the number of unique bins for each binned column
    df_grouped = (
        df.reset_index(names=idx_col)
        .groupby([*[f"{c}_bins" for c in bin_by_cols], *group_by_cols])
        .first()
        .dropna()
    )
    for col, expected in zip(bin_by_cols, expected_n_bins):
        binned_col = f"{col}_bins"
        assert binned_col in df_grouped.index.names

        uniq_bins = df_grouped.index.get_level_values(binned_col).nunique()
        assert uniq_bins == expected


def test_bin_df_cols_raises_value_error() -> None:
    df = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [2, 3, 4, 5]})
    bin_by_cols = ["col1", "col2"]

    with pytest.raises(ValueError) as exc:
        bin_df_cols(df, bin_by_cols, n_bins=[2])

    assert "len(bin_by_cols)=2 != len(n_bins)=1" in str(exc.value)


sample_dict = {"a": 1, "b": None, "c": [3, 4]}
ref_sample_dict = deepcopy(sample_dict)


def test_patch_dict_with_kwargs() -> None:
    with patch_dict(sample_dict, a=2, b=3, d=4) as patched_dict:
        assert patched_dict == {"a": 2, "b": 3, "c": [3, 4], "d": 4}
    assert sample_dict == ref_sample_dict


def test_patch_dict_with_args() -> None:
    with patch_dict(sample_dict, {"a": 5, "b": 6}) as patched_dict:
        assert patched_dict == {"a": 5, "b": 6, "c": [3, 4]}
    assert sample_dict == ref_sample_dict


def test_patch_dict_with_none_value() -> None:
    with patch_dict(sample_dict, b=5, c=None) as patched_dict:
        assert patched_dict == {"a": 1, "b": 5, "c": None}
    assert sample_dict == ref_sample_dict


def test_patch_dict_with_new_key() -> None:
    with patch_dict(sample_dict, d=7, e=None) as patched_dict:
        assert patched_dict == {"a": 1, "b": None, "c": [3, 4], "d": 7, "e": None}
    assert sample_dict == ref_sample_dict


def test_patch_dict_with_mutable_value() -> None:
    with patch_dict(sample_dict, c=[5, 6]) as patched_dict:
        assert patched_dict["c"] == [5, 6]
        patched_dict["c"].append(7)
        patched_dict["c"][0] = 99
        assert patched_dict == {"a": 1, "b": None, "c": [99, 6, 7]}

    assert sample_dict != patched_dict
    assert sample_dict == ref_sample_dict


def test_patch_dict_empty() -> None:
    empty_dict: dict[str, int] = {}
    with patch_dict(empty_dict, a=2) as patched_dict:
        assert patched_dict == {"a": 2}
    assert empty_dict == {}


def test_patch_dict_nested_dict() -> None:
    with patch_dict(sample_dict, c={"x": 10, "y": 20}) as patched_dict:
        assert patched_dict == {"a": 1, "b": None, "c": {"x": 10, "y": 20}}
    assert sample_dict == ref_sample_dict


def test_patch_dict_overlapping_args_kwargs() -> None:
    # kwargs should take precedence over args
    with patch_dict(sample_dict, {"a": 7}, a=8) as patched_dict:
        assert patched_dict["a"] == 8
    assert sample_dict == ref_sample_dict


def test_patch_dict_remove_key_inside_context() -> None:
    with patch_dict(sample_dict, d=7) as patched_dict:
        assert patched_dict["d"] == 7
        del patched_dict["d"]
        assert "d" not in patched_dict
    assert sample_dict == ref_sample_dict


assert ref_sample_dict == sample_dict
