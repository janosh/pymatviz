# ruff: noqa: SLF001
from __future__ import annotations

import copy
import re
from copy import deepcopy
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest
from matplotlib.offsetbox import TextArea

import pymatviz as pmv
from pymatviz.utils import MATPLOTLIB, PLOTLY, VALID_FIG_NAMES, CrystalSystem
from tests.conftest import y_pred, y_true


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
    assert crystal_sys == pmv.utils.crystal_sys_from_spg_num(spg_num)


@pytest.mark.parametrize("spg", [-1, 0, 231])
def test_crystal_sys_from_spg_num_invalid(spg: int) -> None:
    with pytest.raises(
        ValueError,
        match=f"Invalid space group number {spg}, must be 1 <= num <= 230",
    ):
        pmv.utils.crystal_sys_from_spg_num(spg)


@pytest.mark.parametrize("spg", [1.2, "3"])
def test_crystal_sys_from_spg_num_typeerror(spg: int) -> None:
    with pytest.raises(
        TypeError, match=f"Expect integer space group number, got {spg=}"
    ):
        pmv.utils.crystal_sys_from_spg_num(spg)


def test_df_to_arrays() -> None:
    df_regr = pd.DataFrame([y_true, y_pred]).T
    x1, y1 = pmv.utils.df_to_arrays(None, y_true, y_pred)
    x_col, y_col = df_regr.columns[:2]
    x2, y2 = pmv.utils.df_to_arrays(df_regr, x_col, y_col)
    assert x1 == pytest.approx(x2)
    assert y1 == pytest.approx(y2)
    assert x1 == pytest.approx(y_true)
    assert y1 == pytest.approx(y_pred)

    with pytest.raises(TypeError, match="df should be pandas DataFrame or None"):
        pmv.utils.df_to_arrays("foo", y_true, y_pred)

    bad_col_name = "not-real-col-name"
    with pytest.raises(KeyError) as exc:
        pmv.utils.df_to_arrays(df_regr, bad_col_name, df_regr.columns[0])

    assert "not-real-col-name" in str(exc.value)


def test_df_to_arrays_strict() -> None:
    args = pmv.utils.df_to_arrays(42, "foo", "bar", strict=False)
    assert args == ("foo", "bar")

    with pytest.raises(TypeError, match="df should be pandas DataFrame or None"):
        pmv.utils.df_to_arrays(42, "foo", "bar", strict=True)


@pytest.mark.parametrize(
    "bin_by_cols, group_by_cols, n_bins, expected_n_bins, "
    "verbose, density_col, expected_n_rows",
    [
        (["A"], [], 2, [2], True, "", 2),
        (["A", "B"], [], 2, [2, 2], True, "kde_bin_counts", 4),
        (["A", "B"], [], [2, 3], [2, 3], False, "kde_bin_counts", 6),
        (["A"], ["B"], 2, [2], False, "", 30),
    ],
)
def test_bin_df_cols(
    bin_by_cols: list[str],
    group_by_cols: list[str],
    n_bins: int | list[int],
    expected_n_bins: list[int],
    verbose: bool,
    density_col: str,
    expected_n_rows: int,
    df_float: pd.DataFrame,
) -> None:
    idx_col = "index"
    # don't move this below df_float.copy() line
    df_float.index.name = idx_col

    # keep copy of original DataFrame to assert it is not modified
    # not using df.copy(deep=True) here for extra sensitivity, doc str says
    # not as deep as deepcopy
    df_float_orig = copy.deepcopy(df_float)

    bin_counts_col = "bin_counts"
    df_binned = pmv.utils.bin_df_cols(
        df_float,
        bin_by_cols,
        group_by_cols=group_by_cols,
        n_bins=n_bins,
        verbose=verbose,
        bin_counts_col=bin_counts_col,
        density_col=density_col,
    )

    assert len(df_binned) == expected_n_rows
    assert len(df_binned) <= len(df_float)
    assert df_binned.index.name == idx_col

    # ensure binned DataFrame has a minimum set of expected columns
    expected_cols = {bin_counts_col, *df_float, *(f"{col}_bins" for col in bin_by_cols)}
    assert (
        {*df_binned} >= expected_cols
    ), f"{set(df_binned)=}\n{expected_cols=},\n{bin_by_cols=}\n{group_by_cols=}"

    # validate the number of unique bins for each binned column
    for col, n_bins_expec in zip(bin_by_cols, expected_n_bins, strict=True):
        assert df_binned[f"{col}_bins"].nunique() == n_bins_expec

    # ensure original DataFrame is not modified
    pd.testing.assert_frame_equal(df_float, df_float_orig)

    # Check that the index values of df_binned are a subset of df_float
    assert set(df_binned.index).issubset(set(df_float.index))

    # Check that bin_counts column exists and contains only integers
    assert bin_counts_col in df_binned
    assert df_binned[bin_counts_col].dtype in [int, "int64"]

    # If density column is specified, check if it exists
    if density_col:
        assert density_col in df_binned
    else:
        assert density_col not in df_binned


def test_bin_df_cols_raises() -> None:
    df_dummy = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [2, 3, 4, 5]})
    bin_by_cols = ["col1", "col2"]

    # test error when passing n_bins as list but list has wrong length
    with pytest.raises(ValueError) as exc:  # noqa: PT011
        pmv.utils.bin_df_cols(df_dummy, bin_by_cols, n_bins=[2])

    assert "len(bin_by_cols)=2 != len(n_bins)=1" in str(exc.value)


sample_dict = {"a": 1, "b": None, "c": [3, 4]}
ref_sample_dict = deepcopy(sample_dict)


def test_patch_dict_with_kwargs() -> None:
    with pmv.utils.patch_dict(sample_dict, a=2, b=3, d=4) as patched_dict:
        assert patched_dict == {"a": 2, "b": 3, "c": [3, 4], "d": 4}
    assert sample_dict == ref_sample_dict


def test_patch_dict_with_args() -> None:
    with pmv.utils.patch_dict(sample_dict, {"a": 5, "b": 6}) as patched_dict:
        assert patched_dict == {"a": 5, "b": 6, "c": [3, 4]}
    assert sample_dict == ref_sample_dict


def test_patch_dict_with_none_value() -> None:
    with pmv.utils.patch_dict(sample_dict, b=5, c=None) as patched_dict:
        assert patched_dict == {"a": 1, "b": 5, "c": None}
    assert sample_dict == ref_sample_dict


def test_patch_dict_with_new_key() -> None:
    with pmv.utils.patch_dict(sample_dict, d=7, e=None) as patched_dict:
        assert patched_dict == {"a": 1, "b": None, "c": [3, 4], "d": 7, "e": None}
    assert sample_dict == ref_sample_dict


def test_patch_dict_with_mutable_value() -> None:
    with pmv.utils.patch_dict(sample_dict, c=[5, 6]) as patched_dict:
        assert patched_dict["c"] == [5, 6]
        patched_dict["c"].append(7)
        patched_dict["c"][0] = 99
        assert patched_dict == {"a": 1, "b": None, "c": [99, 6, 7]}

    assert sample_dict != patched_dict
    assert sample_dict == ref_sample_dict


def test_patch_dict_empty() -> None:
    empty_dict: dict[str, int] = {}
    with pmv.utils.patch_dict(empty_dict, a=2) as patched_dict:
        assert patched_dict == {"a": 2}
    assert empty_dict == {}


def test_patch_dict_nested_dict() -> None:
    with pmv.utils.patch_dict(sample_dict, c={"x": 10, "y": 20}) as patched_dict:
        assert patched_dict == {"a": 1, "b": None, "c": {"x": 10, "y": 20}}
    assert sample_dict == ref_sample_dict


def test_patch_dict_overlapping_args_kwargs() -> None:
    # kwargs should take precedence over args
    a_val = 8
    with pmv.utils.patch_dict(sample_dict, {"a": 7}, a=a_val) as patched_dict:
        assert patched_dict["a"] == a_val
    assert sample_dict == ref_sample_dict


def test_patch_dict_remove_key_inside_context() -> None:
    d_val = 7
    with pmv.utils.patch_dict(sample_dict, d=d_val) as patched_dict:
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
        ("#FF0000", 0.299),  # Red
        ("#00FF00", 0.587),  # Green
        ("#0000FF", 0.114),  # Blue
        ("red", 0.299),
        ("green", 0.294650),
        ("blue", 0.114),
    ],
)
def test_luminance(color: tuple[float, float, float], expected: float) -> None:
    assert pmv.utils.luminance(color) == pytest.approx(expected, 0.001), f"{color=}"


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
    assert pmv.utils.pick_bw_for_contrast(color, text_color_threshold) == expected


def test_si_fmt() -> None:
    assert pmv.utils.si_fmt(0) == "0.0"
    assert pmv.utils.si_fmt(123) == "123.0"
    assert pmv.utils.si_fmt(1234) == "1.2K"
    assert pmv.utils.si_fmt(123456) == "123.5K"
    assert pmv.utils.si_fmt(12345678, fmt=">6.2f", sep=" ") == " 12.35 M"
    assert pmv.utils.si_fmt(-0.00123, fmt=".3g", binary=False) == "-1.23m"
    assert pmv.utils.si_fmt(0.00000123, fmt="5.1f", sep="\t", binary=True) == "  1.3\tμ"
    assert (
        pmv.utils.si_fmt(0.00000123, fmt="5.1f", sep="\t", binary=False) == "  1.2\tμ"
    )
    assert pmv.utils.si_fmt(0.321, fmt=".2f") == "0.32"
    assert pmv.utils.si_fmt(-0.93) == "-0.9"
    assert pmv.utils.si_fmt(-0.93, fmt=".2f") == "-0.93"
    assert pmv.utils.si_fmt(-0.1) == "-0.1"
    assert pmv.utils.si_fmt(-0.001) == "-1.0m"
    assert pmv.utils.si_fmt(-0.001, decimal_threshold=0.001, fmt=".3f") == "-0.001"
    assert pmv.utils.si_fmt(-1) == "-1.0"
    assert pmv.utils.si_fmt(1.23456789e-10, fmt="5.1f", sep="\t") == "123.5\tp"


def test_si_fmt_int() -> None:
    assert pmv.utils.si_fmt_int(0) == "0"
    assert pmv.utils.si_fmt_int(123) == "123"
    assert pmv.utils.si_fmt_int(1234) == "1K"
    assert pmv.utils.si_fmt_int(123456) == "123K"
    assert pmv.utils.si_fmt_int(12345678, fmt=">6.2f", sep=" ") == " 12.35 M"
    assert pmv.utils.si_fmt_int(-1) == "-1"
    assert pmv.utils.si_fmt_int(1.23456789e-10, sep="\t") == "123\tp"


class TestGetCbarLabelFormatter:
    data = np.random.default_rng().random((10, 10))
    fig, ax = plt.subplots()
    cax = ax.imshow(data, cmap="viridis")
    cbar = fig.colorbar(cax)
    cbar.set_ticks([0.0123456789])

    @pytest.mark.parametrize(
        "default_decimal_places, expected",
        [
            (3, "1.235%"),
            (4, "1.2346%"),
        ],
    )
    def test_default_decimal_places(
        self, default_decimal_places: int, expected: str
    ) -> None:
        with pytest.warns(match="Invalid cbar_label_fmt="):
            formatter = pmv.utils.get_cbar_label_formatter(
                cbar_label_fmt=".2f",  # bad f-string format for percent mode
                values_fmt=".2f",  # bad f-string format for percent mode
                values_show_mode="percent",
                sci_notation=False,
                default_decimal_places=default_decimal_places,
            )

        self.cbar.ax.yaxis.set_major_formatter(formatter)
        labels = [label.get_text() for label in self.cbar.ax.get_yticklabels()]

        assert labels[0] == expected, labels

    @pytest.mark.parametrize(
        "sci_notation, expected", [(True, "1.23"), (False, "0.01")]
    )
    def test_sci_notation(self, sci_notation: bool, expected: str) -> None:
        formatter = pmv.utils.get_cbar_label_formatter(
            cbar_label_fmt=".2f",
            values_fmt=".2f",
            values_show_mode="value",
            sci_notation=sci_notation,
        )

        self.cbar.ax.yaxis.set_major_formatter(formatter)
        labels = [label.get_text() for label in self.cbar.ax.get_yticklabels()]

        assert labels[0] == expected, labels

    @pytest.mark.parametrize(
        "values_show_mode, expected",
        [
            ("value", "0.0123"),
            ("fraction", "0.0123"),
            ("percent", "1.2%"),  # default decimal place being 1
            ("off", "0.0123"),
        ],
    )
    def test_values_show_mode(
        self,
        values_show_mode: Literal["value", "fraction", "percent", "off"],
        expected: str,
    ) -> None:
        formatter = pmv.utils.get_cbar_label_formatter(
            cbar_label_fmt=".4f",
            values_fmt=".4f",
            values_show_mode=values_show_mode,
            sci_notation=False,
        )

        self.cbar.ax.yaxis.set_major_formatter(formatter)
        labels = [label.get_text() for label in self.cbar.ax.get_yticklabels()]

        assert labels[0] == expected, labels


@pytest.mark.parametrize(
    "text, tag, title, style",
    [
        ("foo", "span", "", ""),
        ("bar", "small", "some title", "color: red;"),
        ("baz", "div", "long title " * 10, "font-size: 0.8em;"),
        ("", "strong", " ", "font-size: 0.8em; font-weight: lighter;"),
        ("", "strong", " ", "small"),
    ],
)
def test_html_tag(text: str, tag: str, title: str, style: str) -> None:
    orig_style = style
    style = {"small": "font-size: 0.8em; font-weight: lighter;"}.get(style, style)
    attrs = f" {title=} " if title else ""
    attrs += f"{style=}" if style else ""
    assert (
        pmv.utils.html_tag(text, tag=tag, title=title, style=orig_style)
        == f"<{tag}{attrs}>{text}</{tag}>"
    )


def test_pretty_label() -> None:
    assert pmv.utils.pretty_label("R2", MATPLOTLIB) == "$R^2$"
    assert pmv.utils.pretty_label("R2", PLOTLY) == "R<sup>2</sup>"
    assert pmv.utils.pretty_label("R2_adj", MATPLOTLIB) == "$R^2_{adj}$"
    assert pmv.utils.pretty_label("R2_adj", PLOTLY) == "R<sup>2</sup><sub>adj</sub>"
    assert pmv.utils.pretty_label("foo", MATPLOTLIB) == "foo"
    assert pmv.utils.pretty_label("foo", PLOTLY) == "foo"

    with pytest.raises(ValueError, match="Unexpected backend='foo'"):
        pmv.utils.pretty_label("R2", "foo")  # type: ignore[arg-type]


@pytest.mark.parametrize("color", ["red", "blue", "#FF0000"])
def test_annotate(
    color: str, plotly_scatter: go.Figure, matplotlib_scatter: plt.Figure
) -> None:
    text = "Test annotation"

    fig_plotly = pmv.utils.annotate(text, plotly_scatter, color=color)
    assert isinstance(fig_plotly, go.Figure)
    assert fig_plotly.layout.annotations[-1].text == text
    assert fig_plotly.layout.annotations[-1].font.color == color

    fig_mpl = pmv.utils.annotate(text, matplotlib_scatter, color=color)
    assert isinstance(fig_mpl, plt.Figure)
    assert isinstance(fig_mpl.axes[0].artists[-1].txt, TextArea)
    assert fig_mpl.axes[0].artists[-1].txt.get_text() == text
    assert fig_mpl.axes[0].artists[-1].txt._text.get_color() == color

    ax_mpl = pmv.utils.annotate(text, matplotlib_scatter.axes[0], color=color)
    assert isinstance(ax_mpl, plt.Axes)
    assert ax_mpl.artists[-1].txt.get_text() == text
    assert ax_mpl.artists[-1].txt._text.get_color() == color


def test_annotate_invalid_fig() -> None:
    with pytest.raises(TypeError, match="Input must be .+ got type"):
        pmv.utils.annotate("test", fig="invalid")


def test_annotate_faceted_plotly(plotly_faceted_scatter: go.Figure) -> None:
    texts = ["Annotation 1", "Annotation 2"]
    fig: go.Figure = pmv.utils.annotate(texts, plotly_faceted_scatter)

    assert len(fig.layout.annotations) == 2
    assert fig.layout.annotations[0].text == texts[0]
    assert fig.layout.annotations[1].text == texts[1]
    assert fig.layout.annotations[0].xref == "x domain"
    assert fig.layout.annotations[1].xref == "x2 domain"


def test_annotate_faceted_plotly_with_empty_string(
    plotly_faceted_scatter: go.Figure,
) -> None:
    texts = ["Annotation 1", ""]
    fig: go.Figure = pmv.utils.annotate(texts, plotly_faceted_scatter)

    assert len(fig.layout.annotations) == 1
    assert fig.layout.annotations[0].text == texts[0]


def test_annotate_faceted_plotly_with_single_string(
    plotly_faceted_scatter: go.Figure,
) -> None:
    text = "Single Annotation"
    fig: go.Figure = pmv.utils.annotate(text, plotly_faceted_scatter)

    assert len(fig.layout.annotations) == 2
    for annotation in fig.layout.annotations:
        assert annotation.text == text


def test_annotate_non_faceted_plotly_with_list_raises(
    plotly_scatter: go.Figure,
) -> None:
    text = ["Annotation 1", "Annotation 2"]
    text_type = type(text).__name__
    with pytest.raises(
        ValueError,
        match=re.escape(f"Unexpected {text_type=} for non-faceted plot, must be str"),
    ):
        pmv.utils.annotate(text, plotly_scatter)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": 0.5, "y": 0.5},
        {"font": dict(size=20, color="green")},
        {"showarrow": True, "arrowhead": 2},
    ],
)
def test_annotate_kwargs(plotly_scatter: go.Figure, kwargs: dict[str, Any]) -> None:
    fig: go.Figure = pmv.utils.annotate("Test", plotly_scatter, **kwargs)

    for key, val in kwargs.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                assert getattr(fig.layout.annotations[-1][key], sub_key) == sub_val
        else:
            assert getattr(fig.layout.annotations[-1], key) == val


def test_validate_fig_decorator_raises(capsys: pytest.CaptureFixture[str]) -> None:
    @pmv.utils.validate_fig
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
        x_range, y_range = pmv.utils.get_fig_xy_range(fig)
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
    with pytest.raises(TypeError, match="Unexpected type for fig: str, must be one of"):
        pmv.utils.get_fig_xy_range(fig="invalid")


def test_get_font_color() -> None:
    orig_template = pio.templates.default
    try:
        pio.templates.default = "plotly"
        mpl_fig = plt.figure()
        mpl_ax = mpl_fig.add_subplot(111)
        for fig, expected_color in (
            (go.Figure(), "#2a3f5f"),
            (mpl_fig, "black"),
            (mpl_ax, "black"),
        ):
            color = pmv.utils.get_font_color(fig)
            assert color == expected_color, f"{fig=}, {color=}, {expected_color=}"
    finally:
        pio.templates.default = orig_template


def test_get_font_color_invalid_input() -> None:
    fig = "invalid input"
    with pytest.raises(
        TypeError, match=re.escape(f"Input must be {VALID_FIG_NAMES}, got {type(fig)=}")
    ):
        pmv.utils.get_font_color(fig)


@pytest.mark.parametrize("color", ["red", "#00FF00", "rgb(0, 0, 255)"])
def test_get_plotly_font_color(color: str) -> None:
    fig = go.Figure().update_layout(font_color=color)
    assert pmv.utils._get_plotly_font_color(fig) == color


def test_get_plotly_font_color_from_template() -> None:
    template = pio.templates["plotly_white"]
    template.layout.font.color = "blue"
    pio.templates.default = "plotly_white"
    fig = go.Figure()
    try:
        assert pmv.utils._get_plotly_font_color(fig) == "blue"
    finally:
        pio.templates.default = "plotly"  # Reset to default template


def test_get_plotly_font_color_default() -> None:
    fig = go.Figure()
    assert pmv.utils._get_plotly_font_color(fig) == "#2a3f5f"  # Default Plotly color


@pytest.mark.parametrize("color", ["red", "#00FF00", "blue"])
def test_get_matplotlib_font_color(color: str) -> None:
    color = pmv.utils._get_matplotlib_font_color(plt.figure())
    assert color == "black"  # Default color

    fig, ax = plt.subplots()
    assert (
        pmv.utils._get_matplotlib_font_color(fig)
        == pmv.utils._get_matplotlib_font_color(ax)
        == "black"
    )

    mpl_ax = plt.figure().add_subplot(111)
    mpl_ax.xaxis.label.set_color(color)
    assert pmv.utils._get_matplotlib_font_color(mpl_ax) == color


def test_get_matplotlib_font_color_from_rcparams() -> None:
    original_color = plt.rcParams["text.color"]
    try:
        plt.rcParams["text.color"] = "green"
        fig, ax = plt.subplots()
        ax.set_xlabel("X Label", color="green")
        ax.set_ylabel("Y Label", color="green")
        ax.set_title("Title", color="green")
        ax.tick_params(colors="green")
        plt.close(fig)  # Close the figure to ensure changes are applied

        color = pmv.utils._get_matplotlib_font_color(ax)
        assert color == "green", f"Expected 'green', but got '{color}'"
    finally:
        plt.rcParams["text.color"] = original_color  # Reset to original value
