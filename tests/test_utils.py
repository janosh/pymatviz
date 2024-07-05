from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.offsetbox import TextArea

from pymatviz.utils import (
    MATPLOTLIB,
    PLOTLY,
    CrystalSystem,
    annotate,
    bin_df_cols,
    crystal_sys_from_spg_num,
    df_to_arrays,
    get_fig_xy_range,
    luminance,
    patch_dict,
    pick_bw_for_contrast,
    pretty_label,
    si_fmt,
    si_fmt_int,
    styled_html_tag,
    validate_fig,
)
from tests.conftest import y_pred, y_true


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
    with pytest.raises(
        ValueError,
        match=f"Invalid space group number {spg}, must be 1 <= num <= 230",
    ):
        crystal_sys_from_spg_num(spg)


@pytest.mark.parametrize("spg", [1.2, "3"])
def test_crystal_sys_from_spg_num_typeerror(spg: int) -> None:
    with pytest.raises(
        TypeError, match=f"Expect integer space group number, got {spg=}"
    ):
        crystal_sys_from_spg_num(spg)


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


def test_df_to_arrays_strict() -> None:
    args = df_to_arrays(42, "foo", "bar", strict=False)
    assert args == ("foo", "bar")

    with pytest.raises(TypeError, match="df should be pandas DataFrame or None"):
        df_to_arrays(42, "foo", "bar", strict=True)


@pytest.mark.parametrize(
    "bin_by_cols, group_by_cols, n_bins, expected_n_bins, "
    "verbose, kde_col, expected_n_rows",
    [
        (["A"], [], 2, [2], True, "", 2),
        (["A", "B"], [], 2, [2, 2], True, "kde", 4),
        (["A", "B"], [], [2, 3], [2, 3], False, "kde", 6),
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
    df_float: pd.DataFrame,
) -> None:
    idx_col = "index"
    df_float.index.name = idx_col
    bin_counts_col = "bin_counts"
    df_binned = bin_df_cols(
        df_float,
        bin_by_cols,
        group_by_cols=group_by_cols,
        n_bins=n_bins,
        verbose=verbose,
        bin_counts_col=bin_counts_col,
        kde_col=kde_col,
    )

    # ensure binned DataFrame has a minimum set of expected columns
    expected_cols = {bin_counts_col, *df_float, *(f"{col}_bins" for col in bin_by_cols)}
    assert {*df_binned} >= expected_cols
    assert len(df_binned) == expected_n_rows

    # validate the number of unique bins for each binned column
    df_grouped = (
        df_float.reset_index(names=idx_col)
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
    with pytest.raises(ValueError) as exc:  # noqa: PT011
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
    a_val = 8
    with patch_dict(sample_dict, {"a": 7}, a=a_val) as patched_dict:
        assert patched_dict["a"] == a_val
    assert sample_dict == ref_sample_dict


def test_patch_dict_remove_key_inside_context() -> None:
    d_val = 7
    with patch_dict(sample_dict, d=d_val) as patched_dict:
        assert patched_dict["d"] == d_val
        del patched_dict["d"]
        assert "d" not in patched_dict
    assert sample_dict == ref_sample_dict


assert ref_sample_dict == sample_dict, "sample_dict should not be modified"


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
    assert si_fmt(12345678, fmt=">6.2f", sep=" ") == " 12.35 M"
    assert si_fmt(-0.00123, fmt=".3g", binary=False) == "-1.23m"
    assert si_fmt(0.00000123, fmt="5.1f", sep="\t", binary=True) == "  1.3\tμ"
    assert si_fmt(0.00000123, fmt="5.1f", sep="\t", binary=False) == "  1.2\tμ"
    assert si_fmt(0.321, fmt=".2f") == "0.32"
    assert si_fmt(-0.93) == "-0.9"
    assert si_fmt(-0.93, fmt=".2f") == "-0.93"
    assert si_fmt(-0.1) == "-0.1"
    assert si_fmt(-0.001) == "-1.0m"
    assert si_fmt(-0.001, decimal_threshold=0.001, fmt=".3f") == "-0.001"
    assert si_fmt(-1) == "-1.0"
    assert si_fmt(1.23456789e-10, fmt="5.1f", sep="\t") == "123.5\tp"


def test_si_fmt_int() -> None:
    assert si_fmt_int(0) == "0"
    assert si_fmt_int(123) == "123"
    assert si_fmt_int(1234) == "1K"
    assert si_fmt_int(123456) == "123K"
    assert si_fmt_int(12345678, fmt=">6.2f", sep=" ") == " 12.35 M"
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


def test_pretty_label() -> None:
    assert pretty_label("R2", MATPLOTLIB) == "$R^2$"
    assert pretty_label("R2", PLOTLY) == "R<sup>2</sup>"
    assert pretty_label("R2_adj", MATPLOTLIB) == "$R^2_{adj}$"
    assert pretty_label("R2_adj", PLOTLY) == "R<sup>2</sup><sub>adj</sub>"
    assert pretty_label("foo", MATPLOTLIB) == "foo"
    assert pretty_label("foo", PLOTLY) == "foo"

    with pytest.raises(ValueError, match="Unexpected backend='foo'"):
        pretty_label("R2", "foo")  # type: ignore[arg-type]


@pytest.mark.parametrize("color", ["red", "blue", "#FF0000"])
def test_annotate(
    color: str, plotly_scatter: go.Figure, matplotlib_scatter: plt.Figure
) -> None:
    text = "Test annotation"

    fig_plotly = annotate(text, plotly_scatter, color=color)
    assert isinstance(fig_plotly, go.Figure)
    assert fig_plotly.layout.annotations[-1].text == text
    assert fig_plotly.layout.annotations[-1].font.color == color

    fig_mpl = annotate(text, matplotlib_scatter, color=color)
    assert isinstance(fig_mpl, plt.Figure)
    assert isinstance(fig_mpl.axes[0].artists[-1].txt, TextArea)
    assert fig_mpl.axes[0].artists[-1].txt.get_text() == text
    assert fig_mpl.axes[0].artists[-1].txt._text.get_color() == color  # noqa: SLF001

    ax_mpl = annotate(text, matplotlib_scatter.axes[0], color=color)
    assert isinstance(ax_mpl, plt.Axes)
    assert ax_mpl.artists[-1].txt.get_text() == text
    assert ax_mpl.artists[-1].txt._text.get_color() == color  # noqa: SLF001


def test_annotate_invalid_fig() -> None:
    with pytest.raises(TypeError, match="Unexpected type for fig: str, must be one of"):
        annotate("test", fig="invalid")


def test_validate_fig_decorator_raises(capsys: pytest.CaptureFixture[str]) -> None:
    @validate_fig
    def generic_func(fig: Any = None, **kwargs: Any) -> Any:
        return fig, kwargs

    # check no error on valid fig types
    for fig in (None, go.Figure(), plt.gcf(), plt.gca()):
        generic_func(fig=fig)
        stdout, stderr = capsys.readouterr()
        assert stdout == ""
        assert stderr == ""

    # check TypeError on invalid fig types
    for invalid in (42, "invalid"):
        with pytest.raises(
            TypeError, match=f"Unexpected type for fig: {type(invalid).__name__}"
        ):
            generic_func(fig=invalid)


def test_get_fig_xy_range(
    plotly_scatter: go.Figure, matplotlib_scatter: plt.Figure
) -> None:
    for fig in (plotly_scatter, matplotlib_scatter, matplotlib_scatter.axes[0]):
        x_range, y_range = get_fig_xy_range(fig)
        assert isinstance(x_range, tuple)
        assert isinstance(y_range, tuple)
        assert len(x_range) == 2
        assert len(y_range) == 2
        assert x_range[0] < x_range[1]
        assert y_range[0] < y_range[1]
        for val in (*x_range, *y_range):
            assert isinstance(val, float)

    # test invalid input
    # currently suboptimal behavior: fig must be passed as kwarg to trigger helpful
    # error message
    with pytest.raises(
        TypeError, match="Unexpected type for fig: str, must be one of None"
    ):
        get_fig_xy_range(fig="invalid")
