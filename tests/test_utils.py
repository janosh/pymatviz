from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from random import random
from typing import TYPE_CHECKING, Literal
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest

from pymatviz.utils import (
    Backend,
    CrystalSystem,
    add_ecdf_line,
    add_identity_line,
    annotate_bars,
    annotate_metrics,
    bin_df_cols,
    crystal_sys_from_spg_num,
    df_to_arrays,
    luminance,
    patch_dict,
    pick_bw_for_contrast,
    pretty_metric_label,
    si_fmt,
    si_fmt_int,
    styled_html_tag,
)
from tests.conftest import y_pred, y_true


if TYPE_CHECKING:
    from collections.abc import Sequence


def _extract_anno_from_fig(fig: go.Figure | plt.Figure, idx: int = -1) -> str:
    # get plotly or matplotlib annotation text. idx=-1 gets the most recently added
    # annotation
    if not isinstance(fig, (go.Figure, plt.Figure)):
        raise TypeError(f"Unexpected {type(fig)=}")

    if isinstance(fig, go.Figure):
        anno_text = fig.layout.annotations[idx].text
    else:
        text_box = fig.axes[0].artists[idx]
        anno_text = text_box.txt.get_text()

    return anno_text


@pytest.mark.parametrize(
    "metrics, fmt",
    [
        ("MSE", ".1"),
        (["RMSE"], ".1"),
        (("MAPE", "MSE"), ".2"),
        ({"MAE", "R2", "RMSE"}, ".3"),
        ({"MAE": 1.4, "R2": 0.2, "RMSE": 1.9}, ".0"),
    ],
)
def test_annotate_metrics(
    metrics: dict[str, float] | Sequence[str],
    fmt: str,
    plotly_scatter: go.Figure,
    matplotlib_scatter: plt.Figure,
) -> None:
    # randomly switch between plotly and matplotlib
    fig = plotly_scatter if random() > 0.5 else matplotlib_scatter

    out_fig = annotate_metrics(y_pred, y_true, metrics=metrics, fmt=fmt, fig=fig)

    assert out_fig is fig
    backend: Backend = "plotly" if isinstance(out_fig, go.Figure) else "matplotlib"

    expected = dict(MAE=0.113, R2=0.765, RMSE=0.144, MAPE=0.5900, MSE=0.0206)

    expected_text = ""
    newline = "<br>" if isinstance(out_fig, go.Figure) else "\n"
    if isinstance(metrics, dict):
        for key, val in metrics.items():
            label = pretty_metric_label(key, backend)
            expected_text += f"{label} = {val:{fmt}}{newline}"
    else:
        for key in [metrics] if isinstance(metrics, str) else metrics:
            label = pretty_metric_label(key, backend)
            expected_text += f"{label} = {expected[key]:{fmt}}{newline}"

    anno_text = _extract_anno_from_fig(out_fig)
    assert anno_text == expected_text, f"{anno_text=}"

    prefix, suffix = f"Metrics:{newline}", f"{newline}the end"
    out_fig = annotate_metrics(
        y_pred, y_true, metrics=metrics, fmt=fmt, prefix=prefix, suffix=suffix, fig=fig
    )
    anno_text_with_fixes = _extract_anno_from_fig(out_fig)
    assert (
        anno_text_with_fixes == prefix + expected_text + suffix
    ), f"{anno_text_with_fixes=}"


@pytest.mark.parametrize("metrics", [42, datetime.now(tz=timezone.utc)])
def test_annotate_metrics_raises(metrics: dict[str, float] | Sequence[str]) -> None:
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
def test_crystal_sys_from_spg_num(spg_num: int, crystal_sys: CrystalSystem) -> None:
    assert crystal_sys == crystal_sys_from_spg_num(spg_num)


@pytest.mark.parametrize("spg", [-1, 0, 231])
def test_crystal_sys_from_spg_num_invalid(spg: int) -> None:
    with pytest.raises(ValueError, match=f"Invalid space group {spg}"):
        crystal_sys_from_spg_num(spg)


@pytest.mark.parametrize("xaxis_type", ["linear", "log"])
@pytest.mark.parametrize("yaxis_type", ["linear", "log"])
@pytest.mark.parametrize("trace_idx", [0, 1])
@pytest.mark.parametrize("line_kwds", [None, {"color": "blue"}])
def test_add_identity_line(
    plotly_scatter: go.Figure,
    xaxis_type: str,
    yaxis_type: str,
    trace_idx: int,
    line_kwds: dict[str, str] | None,
) -> None:
    # Set axis types
    plotly_scatter.layout.xaxis.type = xaxis_type
    plotly_scatter.layout.yaxis.type = yaxis_type

    fig = add_identity_line(plotly_scatter, line_kwds=line_kwds, trace_idx=trace_idx)
    assert isinstance(fig, go.Figure)

    # retrieve identity line
    line = next((shape for shape in fig.layout.shapes if shape.type == "line"), None)
    assert line is not None

    assert line.layer == "below"
    assert line.line.color == (line_kwds["color"] if line_kwds else "gray")
    # check line coordinates
    assert line.x0 == line.y0
    assert line.x1 == line.y1
    # check fig axis types
    assert fig.layout.xaxis.type == xaxis_type
    assert fig.layout.yaxis.type == yaxis_type


@pytest.mark.parametrize("line_kwds", [None, {"color": "blue"}])
def test_add_identity_matplotlib(
    matplotlib_scatter: plt.Figure, line_kwds: dict[str, str] | None
) -> None:
    expected_line_color = (line_kwds or {}).get("color", "black")
    # test Figure
    fig = add_identity_line(matplotlib_scatter, line_kwds=line_kwds)
    assert isinstance(fig, plt.Figure)

    # test Axes
    ax = add_identity_line(matplotlib_scatter.axes[0], line_kwds=line_kwds)
    assert isinstance(ax, plt.Axes)

    line = fig.axes[0].lines[-1]  # retrieve identity line
    assert line.get_color() == expected_line_color

    # test with new log scale axes
    _fig_log, ax_log = plt.subplots()
    ax_log.plot([1, 10, 100], [10, 100, 1000])
    ax_log.set(xscale="log", yscale="log")
    ax_log = add_identity_line(ax, line_kwds=line_kwds)

    line = fig.axes[0].lines[-1]
    assert line.get_color() == expected_line_color


def test_add_identity_raises() -> None:
    for fig in (None, "foo", 42.0):
        with pytest.raises(
            TypeError,
            match=f"{fig=} must be instance of plotly.graph_objs._figure.Figure | "
            f"matplotlib.figure.Figure | matplotlib.axes._axes.Axes",
        ):
            add_identity_line(fig)


def test_df_to_arrays() -> None:
    df_regr = pd.DataFrame([y_true, y_pred]).T
    x1, y1 = df_to_arrays(None, y_true, y_pred)
    x_col, y_col = df_regr.columns[:2]
    x2, y2 = df_to_arrays(df_regr, x_col, y_col)
    assert x1 == pytest.approx(x2)
    assert y1 == pytest.approx(y2)
    assert x1 == pytest.approx(y_true)
    assert y1 == pytest.approx(y_pred)

    with pytest.raises(TypeError, match="df should be pandas DataFrame or None"):
        df_to_arrays("foo", y_true, y_pred)

    bad_col_name = "not-real-col-name"
    with pytest.raises(KeyError) as exc:
        df_to_arrays(df_regr, bad_col_name, df_regr.columns[0])

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
    df: pd.DataFrame = pd._testing.makeDataFrame()  # random data  # noqa: SLF001
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


def test_bin_df_cols_raises() -> None:
    df_dummy = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [2, 3, 4, 5]})
    bin_by_cols = ["col1", "col2"]

    # test error when passing n_bins as list but list has wrong length
    with pytest.raises(ValueError) as exc:
        bin_df_cols(df_dummy, bin_by_cols, n_bins=[2])

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


@pytest.mark.parametrize(
    "v_offset,h_offset,labels,fontsize,y_max_headroom,adjust_test_pos",
    [
        (10, 0, None, 14, 1.2, False),
        (20, 0, ["label1", "label2", "label3"], 10, 1.5, True),
        (5, 5, [100, 200, 300], 16, 1.0, False),
    ],
)
def test_annotate_bars(
    v_offset: int,
    h_offset: int,
    labels: Sequence[str] | None,
    fontsize: int,
    y_max_headroom: float,
    adjust_test_pos: bool,
) -> None:
    bars = plt.bar(["A", "B", "C"], [1, 3, 2])
    ax = plt.gca()
    annotate_bars(
        ax,
        v_offset=v_offset,
        h_offset=h_offset,
        labels=labels,
        fontsize=fontsize,
        y_max_headroom=y_max_headroom,
        adjust_test_pos=adjust_test_pos,
    )

    assert len(ax.texts) == len(bars)

    if labels is None:
        labels = [str(bar.get_height()) for bar in bars]

    # test that labels have expected text and fontsize
    for text, label in zip(ax.texts, labels):
        assert text.get_text() == str(label)
        assert text.get_fontsize() == fontsize

    # test that y_max_headroom is respected
    ylim_max = ax.get_ylim()[1]
    assert ylim_max >= max(bar.get_height() for bar in bars) * y_max_headroom

    # test error when passing wrong number of labels
    bad_labels = ("label1", "label2")
    with pytest.raises(
        ValueError,
        match=f"Got {len(bad_labels)} labels but {len(bars)} bars to annotate",
    ):
        annotate_bars(ax, labels=bad_labels)

    # test error message if adjustText not installed
    err_msg = (
        "adjustText not installed, falling back to default matplotlib label "
        "placement. Use pip install adjustText."
    )
    with patch.dict("sys.modules", {"adjustText": None}), pytest.raises(
        ImportError, match=err_msg
    ):
        annotate_bars(ax, adjust_test_pos=True)


@pytest.mark.parametrize(
    "color,expected",
    [
        ((0, 0, 0), 0),  # Black
        ((1, 1, 1), 1),  # White
        ((0.5, 0.5, 0.5), 0.5),  # Gray
        ((1, 0, 0), 0.299),  # Red
        ((0, 1, 0), 0.587),  # Green
        ((0, 0, 1, 0.3), 0.114),  # Blue with alpha (should be ignored)
    ],
)
def test_luminance(color: tuple[float, float, float], expected: float) -> None:
    assert luminance(color) == pytest.approx(expected, 0.001)


@pytest.mark.parametrize(
    "color,text_color_threshold,expected",
    [
        ((1.0, 1.0, 1.0), 0.7, "black"),  # White
        ((0, 0, 0), 0.7, "white"),  # Black
        ((0.5, 0.5, 0.5), 0.7, "white"),  # Gray
        ((0.5, 0.5, 0.5), 0, "black"),  # Gray with low threshold
        ((1, 0, 0, 0.3), 0.7, "white"),  # Red with alpha (should be ignored)
        ((0, 1, 0), 0.7, "white"),  # Green
        ((0, 0, 1.0), 0.4, "white"),  # Blue with low threshold
    ],
)
def test_pick_bw_for_contrast(
    color: tuple[float, float, float],
    text_color_threshold: float,
    expected: Literal["black", "white"],
) -> None:
    assert pick_bw_for_contrast(color, text_color_threshold) == expected


def test_si_fmt() -> None:
    assert si_fmt(0) == "0.0"
    assert si_fmt(123) == "123.0"
    assert si_fmt(1234) == "1.2K"
    assert si_fmt(123456) == "123.5K"
    assert si_fmt(12345678, fmt_spec=">6.2f", sep=" ") == " 12.35 M"
    assert si_fmt(-0.00123, fmt_spec=".3g", binary=False) == "-1.23m"
    assert si_fmt(0.00000123, fmt_spec="5.1f", sep="\t", binary=True) == "  1.3\tμ"
    assert si_fmt(0.00000123, fmt_spec="5.1f", sep="\t", binary=False) == "  1.2\tμ"
    assert si_fmt(-1) == "-1.0"
    assert si_fmt(1.23456789e-10, fmt_spec="5.1f", sep="\t") == "123.5\tp"


def test_si_fmt_int() -> None:
    assert si_fmt_int(0) == "0"
    assert si_fmt_int(123) == "123"
    assert si_fmt_int(1234) == "1K"
    assert si_fmt_int(123456) == "123K"
    assert si_fmt_int(12345678, fmt_spec=">6.2f", sep=" ") == " 12.35 M"
    assert si_fmt_int(-1) == "-1"
    assert si_fmt_int(1.23456789e-10, sep="\t") == "123\tp"


@pytest.mark.parametrize(
    "text, tag, style",
    [
        ("foo", "span", ""),
        ("bar", "small", "color: red;"),
        ("baz", "div", "font-size: 0.8em;"),
        ("", "strong", "font-size: 0.8em; font-weight: lighter;"),
    ],
)
def test_styled_html_tag(text: str, tag: str, style: str) -> None:
    style = style or "font-size: 0.8em; font-weight: lighter;"
    assert (
        styled_html_tag(text, tag=tag, style=style) == f"<{tag} {style=}>{text}</{tag}>"
    )


@pytest.mark.parametrize(
    "trace_kwargs",
    [None, {}, {"name": "foo", "line_color": "red"}],
)
def test_add_ecdf_line(
    plotly_scatter: go.Figure,
    trace_kwargs: dict[str, str] | None,
) -> None:
    fig = add_ecdf_line(plotly_scatter, trace_kwargs=trace_kwargs)
    assert isinstance(fig, go.Figure)

    trace_kwargs = trace_kwargs or {}

    ecdf = fig.data[-1]  # retrieve ecdf line
    expected_name = trace_kwargs.get("name", "Cumulative")
    expected_color = trace_kwargs.get("line_color", "gray")
    assert ecdf.name == expected_name
    assert ecdf.line.color == expected_color
    assert ecdf.yaxis == "y2"
    assert fig.layout.yaxis2.range == (0, 1)
    assert fig.layout.yaxis2.title.text == expected_name
    assert fig.layout.yaxis2.color == expected_color


def test_add_ecdf_line_raises() -> None:
    for fig in (None, "foo", 42.0):
        with pytest.raises(
            TypeError,
            match=f"{fig=} must be instance of plotly.graph_objs._figure.Figure",
        ):
            add_ecdf_line(fig)
