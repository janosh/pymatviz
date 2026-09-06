from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING

import plotly.graph_objects as go
import plotly.io as pio
import pytest

import pymatviz as pmv


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


@pytest.mark.parametrize(
    ("updates", "kwargs", "expected"),
    [
        ({}, dict(a=2, b=3, d=4), dict(a=2, b=3, c=[3, 4], d=4)),
        (dict(a=5, b=6), {}, dict(a=5, b=6, c=[3, 4])),
        ({}, dict(b=5, c=None), dict(a=1, b=5, c=None)),
        ({}, dict(d=7, e=None), dict(a=1, b=None, c=[3, 4], d=7, e=None)),
        ({}, dict(c={"x": 10, "y": 20}), dict(a=1, b=None, c={"x": 10, "y": 20})),
        (dict(a=7), dict(a=8), dict(a=8, b=None, c=[3, 4])),
    ],
    ids=["kwargs", "positional", "none", "new-keys", "nested", "kwargs-precedence"],
)
def test_patch_dict(
    updates: dict[str, Any], kwargs: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Patches yield a copy, with keyword updates taking precedence."""
    original = {"a": 1, "b": None, "c": [3, 4]}
    before = deepcopy(original)
    with pmv.utils.patch_dict(original, updates, **kwargs) as patched:
        assert patched == expected
    assert original == before


def test_patch_dict_with_mutable_value() -> None:
    """Mutating a replacement list leaves the original list unchanged."""
    original = {"a": 1, "b": None, "c": [3, 4]}
    with pmv.utils.patch_dict(original, c=[5, 6]) as patched:
        assert patched["c"] == [5, 6]
        patched["c"].append(7)
        patched["c"][0] = 99
        assert patched == {"a": 1, "b": None, "c": [99, 6, 7]}
    assert original == {"a": 1, "b": None, "c": [3, 4]}


def test_patch_dict_empty() -> None:
    """Deleting an added key from a patch leaves the original empty."""
    original: dict[str, int] = {}
    with pmv.utils.patch_dict(original, a=2) as patched:
        assert patched == {"a": 2}
        del patched["a"]
        assert patched == {}
    assert original == {}


def test_patch_dict_remove_key_inside_context() -> None:
    """Deleting an added key leaves existing original values untouched."""
    original = {"a": 1, "b": None, "c": [3, 4]}
    with pmv.utils.patch_dict(original, d=7) as patched:
        assert patched["d"] == 7
        del patched["d"]
        assert "d" not in patched
    assert original == {"a": 1, "b": None, "c": [3, 4]}


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


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_si_fmt_extreme_scales(binary: bool, sign: int) -> None:
    """Values beyond the last prefix retain their magnitude."""
    factor = 1024 if binary else 1000
    assert pmv.utils.si_fmt(sign * factor**9, binary=binary, fmt=".3g") == (
        f"{sign * factor:.3g}Y"
    )
    assert pmv.utils.si_fmt(sign * factor**-9, binary=binary, fmt=".3g") == (
        f"{sign / factor:.3g}y"
    )


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
        pmv.utils.annotate("test", fig="invalid")  # ty: ignore[invalid-argument-type]


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

    with pytest.raises(TypeError, match="Expected plotly Figure"):
        pmv.utils.get_fig_xy_range(fig="invalid")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("selector", [[0, 1], slice(None), lambda _trace: True])
@pytest.mark.parametrize("axis_type", ["linear", "log"])
def test_get_fig_xy_range_without_kaleido(
    monkeypatch: pytest.MonkeyPatch, selector: Any, axis_type: str
) -> None:
    """Data-derived bounds include all selected traces and preserve log units."""

    def no_kaleido(*_args: Any, **_kwargs: Any) -> None:
        """Simulate an unavailable static renderer."""
        raise ValueError("Kaleido unavailable")

    monkeypatch.setattr(go.Figure, "full_figure_for_development", no_kaleido)
    fig = go.Figure().add_scatter(x=[1, 10], y=[2, 20])
    fig.add_scatter(x=[100, 200], y=[300, 400])
    fig.update_xaxes(type=axis_type)
    fig.update_yaxes(type=axis_type)
    assert pmv.utils.get_fig_xy_range(fig, traces=selector) == ((1, 200), (2, 400))
    assert pmv.utils.get_fig_xy_range(fig, traces=1) == ((100, 200), (300, 400))
    fig.update_xaxes(range=[0, 3])
    expected_x = (1, 1000) if axis_type == "log" else (0, 3)
    assert pmv.utils.get_fig_xy_range(fig, traces=selector)[0] == expected_x
    fig.update_xaxes(range=[None, 3])
    assert pmv.utils.get_fig_xy_range(fig, traces=selector)[0] == (1, expected_x[1])
    with pytest.raises(ValueError, match="No valid traces"):
        pmv.utils.get_fig_xy_range(fig, traces=lambda _trace: False)


@pytest.mark.parametrize(
    ("layout_color", "template_color", "default_template", "expected"),
    [
        ("red", "blue", "plotly", "red"),
        ("#00FF00", "blue", "plotly", "#00FF00"),
        ("rgb(0, 0, 255)", "blue", "plotly", "rgb(0, 0, 255)"),
        (None, "blue", "plotly", "blue"),
        (None, None, "plotly", "#2a3f5f"),
        (None, None, "none", "black"),
    ],
)
def test_get_font_color(
    monkeypatch: pytest.MonkeyPatch,
    layout_color: str | None,
    template_color: str | None,
    default_template: str,
    expected: str,
) -> None:
    """Font colors prefer layout, figure template, then global template."""
    monkeypatch.setattr(pio.templates, "default", default_template)
    fig = go.Figure().update_layout(
        font_color=layout_color,
        template=go.layout.Template(layout=dict(font_color=template_color)),
    )
    assert pmv.utils.get_font_color(fig) == expected


def test_get_font_color_invalid_input() -> None:
    """Non-figure input reports its type."""
    fig = "invalid input"
    with pytest.raises(
        TypeError, match=re.escape(f"Input must be plotly Figure, got {type(fig)=}")
    ):
        pmv.utils.get_font_color(fig)  # ty: ignore[invalid-argument-type]


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
