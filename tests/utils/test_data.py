from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pytest
from matplotlib.offsetbox import TextArea

import pymatviz as pmv
from pymatviz.typing import MATPLOTLIB, PLOTLY, VALID_FIG_NAMES, CrystalSystem
from pymatviz.utils.plotting import _get_matplotlib_font_color, _get_plotly_font_color


if TYPE_CHECKING:
    from typing import Any


@pytest.mark.parametrize(
    ("spg", "crystal_sys"),
    [
        (1, "triclinic"),
        (15, "monoclinic"),
        (16, "orthorhombic"),
        (75, "tetragonal"),
        (143, "trigonal"),
        (168, "hexagonal"),
        (230, "cubic"),
        # Test with short Hermann-Mauguin symbols
        ("P1", "triclinic"),
        ("P-1", "triclinic"),
        ("P2/m", "monoclinic"),
        ("C2/c", "monoclinic"),
        ("Pnma", "orthorhombic"),
        ("Cmcm", "orthorhombic"),
        ("P4/mmm", "tetragonal"),
        ("I4/mcm", "tetragonal"),
        ("R-3m", "trigonal"),
        ("R3c", "trigonal"),
        ("P6/mmm", "hexagonal"),
        ("P6_3/mmc", "hexagonal"),
        ("Fm-3m", "cubic"),
        ("Ia-3d", "cubic"),
        # Test with full Hermann-Mauguin symbols
        ("P2_1/c", "monoclinic"),
        ("P12_1/c1", "monoclinic"),
        ("C12/c1", "monoclinic"),
    ],
)
def test_spg_to_crystal_sys(spg: int, crystal_sys: CrystalSystem) -> None:
    assert crystal_sys == pmv.utils.spg_to_crystal_sys(spg)


@pytest.mark.parametrize("spg", [-1, 0, 231, 1.2, "3", "invalid", "P999", "X2/m"])
def test_spg_to_crystal_sys_invalid(spg: int) -> None:
    with pytest.raises(ValueError, match=f"Invalid space group {spg}"):
        pmv.utils.spg_to_crystal_sys(spg)


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


def test_si_fmt() -> None:
    assert pmv.utils.si_fmt(0) == "0.0"
    assert pmv.utils.si_fmt(123) == "123.0"
    assert pmv.utils.si_fmt(1234) == "1.2k"
    assert pmv.utils.si_fmt(123456) == "123.5k"
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
    assert pmv.utils.si_fmt_int(1234) == "1k"
    assert pmv.utils.si_fmt_int(123456) == "123k"
    assert pmv.utils.si_fmt_int(12345678, fmt=">6.2f", sep=" ") == " 12.35 M"
    assert pmv.utils.si_fmt_int(-1) == "-1"
    assert pmv.utils.si_fmt_int(1.23456789e-10, sep="\t") == "123\tp"


@pytest.mark.parametrize(
    ("text", "tag", "title", "style"),
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
        assert {*map(type, (*x_range, *y_range))} <= {float, np.float64}

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


def test_get_plotly_font_color_default() -> None:
    orig_template = pio.templates.default
    try:
        pio.templates.default = "plotly"
        fig = go.Figure()
        # test we get default Plotly color
        assert _get_plotly_font_color(fig) == "#2a3f5f"
    finally:
        pio.templates.default = orig_template


def test_get_plotly_font_color_from_template() -> None:
    template = pio.templates["plotly_white"]
    template.layout.font.color = "blue"
    pio.templates.default = "plotly_white"
    fig = go.Figure()
    try:
        assert _get_plotly_font_color(fig) == "blue"
    finally:
        pio.templates.default = "plotly"  # Reset to default template


@pytest.mark.parametrize("color", ["red", "#00FF00", "rgb(0, 0, 255)"])
def test_get_plotly_font_color(color: str) -> None:
    fig = go.Figure().update_layout(font_color=color)
    assert _get_plotly_font_color(fig) == color


@pytest.mark.parametrize("color", ["red", "#00FF00", "blue"])
def test_get_matplotlib_font_color(color: str) -> None:
    color_result = _get_matplotlib_font_color(plt.figure())
    assert color_result == "black"  # Default color

    fig, ax = plt.subplots()
    assert _get_matplotlib_font_color(fig) == _get_matplotlib_font_color(ax) == "black"

    mpl_ax = plt.figure().add_subplot(111)
    mpl_ax.xaxis.label.set_color(color)
    assert _get_matplotlib_font_color(mpl_ax) == color


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

        color = _get_matplotlib_font_color(ax)
        assert color == "green", f"Expected 'green', but got '{color}'"
    finally:
        plt.rcParams["text.color"] = original_color  # Reset to original value


def test_apply_matplotlib_template() -> None:
    """Test that apply_matplotlib_template() correctly sets matplotlib parameters."""
    # Store original values
    orig_params = plt.rcParams.copy()

    try:
        # Apply the template
        pmv.utils.apply_matplotlib_template()

        # Check that parameters were set correctly
        assert plt.rcParams["font.size"] == 14
        assert plt.rcParams["savefig.bbox"] == "tight"
        assert plt.rcParams["savefig.dpi"] == 200
        assert plt.rcParams["axes.titlesize"] == 16
        assert plt.rcParams["axes.titleweight"] == "bold"
        assert plt.rcParams["figure.dpi"] == 200
        assert plt.rcParams["figure.titlesize"] == 20
        assert plt.rcParams["figure.titleweight"] == "bold"
        assert plt.rcParams["figure.constrained_layout.use"] is True

    finally:  # Restore original values
        plt.rcParams.update(orig_params)


def test_hm_symbol_to_spg_num_map() -> None:
    """Test the hm_symbol_to_spg_num_map dictionary properties."""
    from pymatviz.utils.data import hm_symbol_to_spg_num_map

    # Map contains both dense and space separated format of Hermann-Mauguin symbols
    assert len(hm_symbol_to_spg_num_map) == 636

    # Test some specific mappings for common space groups
    assert hm_symbol_to_spg_num_map["P1"] == 1
    assert hm_symbol_to_spg_num_map["P-1"] == 2
    assert hm_symbol_to_spg_num_map["Fm-3m"] == 225
    assert hm_symbol_to_spg_num_map["Ia-3d"] == 230
    assert hm_symbol_to_spg_num_map["P2_1/c"] == 14
    assert hm_symbol_to_spg_num_map["P6_3/mmc"] == 194

    # Test that all values are valid space group numbers
    assert set(hm_symbol_to_spg_num_map.values()) == set(range(1, 230 + 1))


def test_spg_num_to_from_symbol_roundtrip() -> None:
    """Test that converting from number to symbol and back gives the original number."""
    for num in range(1, 230 + 1):
        symbol = pmv.utils.spg_num_to_from_symbol(num)
        num_back = pmv.utils.spg_num_to_from_symbol(symbol)
        assert num == num_back, f"Roundtrip failed for {num} -> {symbol} -> {num_back}"
