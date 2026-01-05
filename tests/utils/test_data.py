from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING

import plotly.graph_objects as go
import plotly.io as pio
import pytest

import pymatviz as pmv
from pymatviz.utils.plotting import _get_plotly_font_color


if TYPE_CHECKING:
    from typing import Any

    from pymatviz.typing import CrystalSystem


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


@pytest.mark.parametrize("color", ["red", "blue", "#FF0000"])
def test_annotate(color: str, plotly_scatter: go.Figure) -> None:
    text = "Test annotation"

    fig_plotly = pmv.utils.annotate(text, plotly_scatter, color=color)
    assert isinstance(fig_plotly, go.Figure)
    assert fig_plotly.layout.annotations[-1].text == text
    assert fig_plotly.layout.annotations[-1].font.color == color


def test_annotate_invalid_fig() -> None:
    with pytest.raises(TypeError, match="Expected plotly Figure"):
        pmv.utils.annotate("test", fig="invalid")  # type: ignore[arg-type]


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


def test_get_fig_xy_range(plotly_scatter: go.Figure) -> None:
    x_range, y_range = pmv.utils.get_fig_xy_range(plotly_scatter)
    assert isinstance(x_range, tuple)
    assert isinstance(y_range, tuple)
    assert len(x_range) == 2
    assert len(y_range) == 2
    assert x_range[0] < x_range[1]
    assert y_range[0] < y_range[1]

    # test invalid input
    # currently suboptimal behavior: fig must be passed as kwarg to trigger helpful
    # error message
    with pytest.raises(TypeError, match="Expected plotly Figure"):
        pmv.utils.get_fig_xy_range(fig="invalid")  # type: ignore[arg-type]


def test_get_font_color() -> None:
    orig_template = pio.templates.default
    try:
        pio.templates.default = "plotly"
        fig = go.Figure()
        color = pmv.utils.get_font_color(fig)
        assert color == "#2a3f5f"
    finally:
        pio.templates.default = orig_template


def test_get_font_color_invalid_input() -> None:
    fig = "invalid input"
    with pytest.raises(
        TypeError, match=re.escape(f"Input must be plotly Figure, got {type(fig)=}")
    ):
        pmv.utils.get_font_color(fig)  # type: ignore[arg-type]


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
