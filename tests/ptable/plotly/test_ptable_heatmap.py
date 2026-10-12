from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
from plotly.exceptions import PlotlyError

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.utils import df_ptable


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any, Literal

    import plotly.graph_objects as go


def tile_labels(fig: go.Figure) -> dict[str, str]:
    """Map element symbols to the value label shown below them on each tile."""
    labels = {}
    for anno in fig.layout.annotations:
        if match := re.match(
            r"<span .*?>(?P<sym>\w+)</span><br>(?P<lbl>.*)", anno.text
        ):
            labels[match["sym"]] = match["lbl"]
    return labels


def test_ptable_heatmap(glass_formulas: list[str]) -> None:
    """Every tile is annotated and the log color scale handles non-positive values."""
    fig = pmv.ptable_heatmap(glass_formulas)
    assert len(fig.layout.annotations) == 18 * 10, (
        "not all periodic table tiles have annotations"
    )
    assert sum(anno.text != "" for anno in fig.layout.annotations) == 118, (
        "no annotations should be empty"
    )

    # test hover_props and show_values=False
    pmv.ptable_heatmap(
        glass_formulas,
        hover_props=("atomic_mass", "atomic_number", "density"),
        show_values=False,
    )
    pmv.ptable_heatmap(
        glass_formulas,
        hover_data="density = " + df_ptable[Key.density].astype(str) + " g/cm^3",
    )
    pmv.ptable_heatmap(glass_formulas, heat_mode="percent")

    # test log color scale with -1, 0, 1 and random negative value
    for val in (-9.72, -1, 0, 1):
        pmv.ptable_heatmap([f"H{val}", "O2"], log=True)
        df_ptable["tmp"] = val
        fig = pmv.ptable_heatmap(df_ptable["tmp"], log=True)
        heatmap_trace = fig.data[-1]
        assert heatmap_trace.colorbar.title.text == "tmp"
        c_scale = heatmap_trace.colorscale
        assert isinstance(c_scale, tuple)
        assert isinstance(c_scale[0], tuple)
        assert isinstance(c_scale[0][0], float)
        assert isinstance(c_scale[0][1], str)
        assert val <= max(c[0] for c in c_scale)


@pytest.mark.parametrize(
    ("kwargs", "err_cls", "match"),
    [
        (
            dict(colorscale=lambda: "bad scale"),
            TypeError,
            "should be string, list of strings or list",
        ),
        (
            dict(colorscale="foobar"),
            PlotlyError,
            "Colorscale foobar is not a built-in scale",
        ),
        (dict(log=True, heat_mode="percent"), ValueError, "Combining log color scale"),
        (
            dict(cscale_range=(0, 10, 20)),
            ValueError,
            re.escape("cscale_range=(0, 10, 20) should have length 2"),
        ),
        (
            dict(log=True, cscale_range=(-1, 50)),
            ValueError,
            "must be positive when log=True",
        ),
        (
            dict(hover_props=("atomic_mass", "invalid_prop")),
            ValueError,
            "Unsupported hover_props: invalid_prop",
        ),
        (dict(hover_props={"atomic_mass"}), ValueError, "must be dict or sequence"),
    ],
)
def test_ptable_heatmap_raises(
    kwargs: dict[str, Any], err_cls: type[Exception], match: str
) -> None:
    """Invalid colorscales, ranges and hover_props raise informative errors."""
    with pytest.raises(err_cls, match=match):
        pmv.ptable_heatmap({"Fe": 10, "O": 3}, **kwargs)


@pytest.mark.parametrize(
    (
        "exclude_elements",
        "heat_mode",
        "log",
        "show_scale",
        "font_size",
        "font_colors",
        "scale",
    ),
    [
        ((), "value", True, False, None, ["red"], 1.3),
        (["O"], "fraction", False, True, 12, ("black", "white"), 3),
        (["P", "S"], "percent", False, False, 14, ["blue"], 1.1),
        ([], "value", True, True, 10, ("green", "yellow"), 0.95),
        (["H", "He", "Li"], "value", False, True, 16, ["purple"], 1.0),
        ([], "value", False, True, None, ["red"], 1.5),
    ],
)
def test_ptable_heatmap_kwarg_combos(
    glass_formulas: list[str],
    exclude_elements: Sequence[str],
    heat_mode: Literal["value", "fraction", "percent"],
    log: bool,
    show_scale: bool,
    font_size: int | None,
    font_colors: Sequence[str],
    scale: float,
) -> None:
    """Exclusions, font sizes/colors and scale apply across kwarg combinations."""
    fig = pmv.ptable_heatmap(
        glass_formulas,
        exclude_elements=exclude_elements,
        heat_mode=heat_mode,
        log=log,
        show_scale=show_scale,
        font_size=font_size,
        font_colors=font_colors,
        scale=scale,
    )

    if exclude_elements:
        assert all(elem not in fig.data[-1].z for elem in exclude_elements)

    if len(font_colors) == 2:
        assert all(anno.font.color in font_colors for anno in fig.layout.annotations), (
            f"{font_colors=}"
        )
    elif len(font_colors) == 1:
        assert all(
            anno.font.color == font_colors[0] for anno in fig.layout.annotations
        ), f"{font_colors=}"

    assert fig.layout.width == pytest.approx(900 * scale)
    assert fig.layout.height == pytest.approx(500 * scale)
    assert fig.layout.font.size == pytest.approx((font_size or 12) * scale)


@pytest.mark.parametrize(
    "c_scale", ["Viridis", "Jet", ("blue", "red"), ((0, "blue"), (1, "red"))]
)
def test_ptable_heatmap_colorscale(c_scale: str) -> None:
    """Named and custom colorscales start at the expected color."""
    values = {"Fe": 2, "O": 3}
    fig = pmv.ptable_heatmap(values, colorscale=c_scale)
    clr_scale_start = fig.data[-1].colorscale[0]
    assert clr_scale_start == {
        "Viridis": (0, "#440154"),
        "Jet": (0, "rgb(0,0,131)"),
        ("blue", "red"): (0, "blue"),
        ((0, "blue"), (1, "red")): (0, "blue"),
    }.get(c_scale), f"{c_scale=}, {clr_scale_start=}"


@pytest.mark.parametrize(
    "colorbar", [{}, dict(orientation="v", len=0.8), dict(orientation="h", len=0.3)]
)
def test_ptable_heatmap_color_bar(
    glass_formulas: list[str], colorbar: dict[str, Any]
) -> None:
    """Colorbar length and title side follow orientation."""
    fig = pmv.ptable_heatmap(glass_formulas, colorbar=colorbar)
    # check color bar has expected length
    assert fig.data[-1].colorbar.len == colorbar.get("len", 0.4)
    # check color bar has expected title side
    assert fig.data[-1].colorbar.title.side == (
        "right" if colorbar.get("orientation") == "v" else "top"
    )


@pytest.mark.parametrize(
    "cscale_range", [(None, None), (None, 10), (2, None), (2, 87123)]
)
def test_ptable_heatmap_cscale_range(
    cscale_range: tuple[float | None, float | None],
) -> None:
    """Colorbar range defaults to the data range per unset bound."""
    fig = pmv.ptable_heatmap(df_ptable[Key.density], cscale_range=cscale_range)
    trace = fig.data[-1]
    assert "colorbar" in trace
    # check for correct color bar range
    data_min, data_max = df_ptable[Key.density].min(), df_ptable[Key.density].max()
    if cscale_range == (None, None):
        # if both None, range is dynamic based on plotted data
        assert trace.zmin == pytest.approx(data_min)
        assert trace.zmax == pytest.approx(data_max)
    else:
        assert trace.zmin == pytest.approx(cscale_range[0] or data_min), (
            f"{cscale_range=}"
        )
        assert trace.zmax == pytest.approx(cscale_range[1] or data_max), (
            f"{cscale_range=}"
        )


def test_ptable_heatmap_log_cscale_range() -> None:
    """Bounds are log10-transformed when log=True.

    Regression: linear bounds in log space inverted the colorscale.
    """
    fig = pmv.ptable_heatmap({"Fe": 10, "O": 100}, log=True, cscale_range=(5, 50))
    trace = next(trc for trc in fig.data if getattr(trc, "zmin", None) is not None)
    assert (trace.zmin, trace.zmax) == pytest.approx(np.log10([5, 50]))


@pytest.mark.parametrize(
    ("label_map", "expected"),
    [
        (None, {"Al": "1.0", "Cr": "2.0", "Fe": "3.0", "Ni": "-"}),
        (False, {"Al": "1.0", "Cr": "2.0", "Fe": "3.0", "Ni": "nan"}),
        (
            {"1.0": "one", "2.0": "two", "3.0": "three", np.nan: "N/A"},
            {"Al": "one", "Cr": "two", "Fe": "three", "Ni": "nan"},
        ),
    ],
)
def test_ptable_heatmap_label_map(
    label_map: dict[str, str] | Literal[False] | None, expected: dict[str, str]
) -> None:
    """Value labels are mapped via label_map, NaN shows as '-' by default."""
    elem_vals = dict(Al=1.0, Cr=2.0, Fe=3.0, Ni=np.nan)
    labels = tile_labels(pmv.ptable_heatmap(elem_vals, label_map=label_map))
    assert {elem: labels[elem] for elem in elem_vals} == expected


def test_ptable_heatmap_show_values() -> None:
    """show_values=False keeps element symbols but drops value labels."""
    values = {"Fe": 2, "O": 3}
    fig = pmv.ptable_heatmap(values, show_values=False)
    anno_texts = [anno.text for anno in fig.layout.annotations if anno.text]
    for text in (
        "<span style='font-weight: bold; font-size: 18.0;'>O</span>",
        "<span style='font-weight: bold; font-size: 18.0;'>Fe</span>",
    ):
        assert text in anno_texts
    assert not any("<br>" in text for text in anno_texts)


def test_ptable_heatmap_custom_label_map() -> None:
    """Callable label_map transforms formatted value labels."""
    values = {"Fe": 2, "O": 3, "H": np.nan}

    def label_map(label: str) -> str:
        """Map numeric labels to custom display strings."""
        return {2.0: "High", 3.0: "Low"}.get(float(label), label)

    fig = pmv.ptable_heatmap(values, label_map=label_map)
    annos = [
        anno.text
        for anno in fig.layout.annotations
        if anno.text.endswith(("High", "Low"))
    ]
    assert len(annos) == 2


@pytest.mark.parametrize("heat_mode", ["value", "fraction", "percent"])
def test_ptable_heatmap_color_range(
    heat_mode: Literal["value", "fraction", "percent"],
) -> None:
    """zmin/zmax match the plotted data range in every heat_mode."""
    values = {"Fe": 1, "O": 50, "H": 100}
    fig = pmv.ptable_heatmap(values, heat_mode=heat_mode)
    heatmap_trace = fig.data[-1]
    non_nan_values = [v for v in heatmap_trace.z.flat if not np.isnan(v)]
    assert min(non_nan_values) == heatmap_trace.zmin
    assert max(non_nan_values) == heatmap_trace.zmax


def test_ptable_heatmap_all_elements() -> None:
    """Data covering all elements sets a finite zmax."""
    values = {elem: idx for idx, elem in enumerate(df_ptable.index)}
    fig = pmv.ptable_heatmap(values)
    heatmap_trace = fig.data[-1]
    assert not np.isnan(heatmap_trace.zmax)
    assert heatmap_trace.zmax == len(df_ptable) - 1


def test_ptable_heatmap_hover_tooltips() -> None:
    """Hover texts show values, percentages, hover_props and hover_data."""
    # Test with non-integer values
    float_values = {"Fe": 0.2, "O": 0.3, "H": 0.5}

    # Test value mode
    fig = pmv.ptable_heatmap(float_values, heat_mode="value")
    hover_texts = fig.data[-1].text.flat

    for elem_symb, value in float_values.items():
        elem_name = df_ptable.loc[elem_symb, "name"]
        hover_text = next(text for text in hover_texts if text.startswith(elem_name))
        assert hover_text == f"{elem_name} ({elem_symb})<br>Value: {value:.3g}"

    # Test fraction and percent modes
    for heat_mode in ["fraction", "percent"]:
        fig = pmv.ptable_heatmap(float_values, heat_mode=heat_mode)  # ty: ignore[invalid-argument-type]
        hover_texts = fig.data[-1].text.flat

        for elem_symb, value in float_values.items():
            elem_name = df_ptable.loc[elem_symb, "name"]
            hover_text = next(
                text for text in hover_texts if text.startswith(elem_name)
            )
            expected_hover_text = (
                f"{elem_name} ({elem_symb})<br>Percentage: {value:.2%} ({value})"
            )
            assert hover_text == expected_hover_text

    # Test with integer values
    int_values = {"Fe": 2, "O": 3, "H": 1}

    for heat_mode in ("value", "fraction", "percent"):
        fig = pmv.ptable_heatmap(int_values, heat_mode=heat_mode)
        hover_texts = fig.data[-1].text.flat

        for elem_symb, value in int_values.items():
            elem_name = df_ptable.loc[elem_symb, "name"]
            hover_text = next(
                text for text in hover_texts if text.startswith(elem_name)
            )

            percent = value / sum(int_values.values())
            if heat_mode == "value":
                expected_hover_text = (
                    f"{elem_name} ({elem_symb})<br>Value: {value} ({percent:.2%})"
                )
            else:
                expected_hover_text = (
                    f"{elem_name} ({elem_symb})<br>Percentage: {percent:.2%} ({value})"
                )
            assert hover_text == expected_hover_text

    # Test with excluded elements
    fig = pmv.ptable_heatmap(int_values, exclude_elements=["O"])
    hover_texts = fig.data[-1].text.flat
    o_hover_text = next(text for text in hover_texts if text.startswith("Oxygen"))
    assert "Value:" not in o_hover_text

    # Test with hover_props
    fig = pmv.ptable_heatmap(int_values, hover_props=["atomic_number", "atomic_mass"])
    hover_texts = fig.data[-1].text.flat
    for hover_text in hover_texts:
        if not hover_text:
            continue
        assert "atomic_number" in hover_text
        assert "atomic_mass" in hover_text

    # Test with hover_data
    hover_data = {"Fe": "Custom Fe data", "O": "Custom O data"}
    fig = pmv.ptable_heatmap(int_values, hover_data=hover_data)
    hover_texts = fig.data[-1].text.flat
    fe_hover_text = next(text for text in hover_texts if text.startswith("Iron"))
    assert "Custom Fe data" in fe_hover_text


def test_ptable_heatmap_element_symbol_map() -> None:
    """Custom symbols replace tile text while hover keeps the real symbol."""
    values = {"Fe": 2, "O": 3, "H": 1}
    element_symbol_map = {"Fe": "Iron", "O": "Oxygen", "H": "Hydrogen"}
    fig = pmv.ptable_heatmap(values, element_symbol_map=element_symbol_map)

    # Check if custom symbols are used in tile texts
    tile_texts = [anno.text for anno in fig.layout.annotations if anno.text]
    for custom_symbol in element_symbol_map.values():
        assert any(custom_symbol in text for text in tile_texts), (
            f"Custom symbol {custom_symbol} not found in tile texts"
        )

    # Check if original element symbols are still used in hover texts
    hover_texts = fig.data[-1].text.flat
    for elem in values:
        hover_text = next(
            text for text in hover_texts if text.startswith(df_ptable.loc[elem, "name"])
        )
        assert f"({elem})" in hover_text, (
            f"Original symbol {elem} not found in hover text"
        )

    # Test with partial mapping
    partial_map = {"Fe": "This be Iron"}
    fig = pmv.ptable_heatmap(values, element_symbol_map=partial_map)
    tile_texts = [anno.text for anno in fig.layout.annotations if anno.text]
    assert any("This be Iron" in text for text in tile_texts), (
        "Custom symbol not found in tile texts"
    )
    assert any("O</span>" in text for text in tile_texts), (
        "Original symbol 'O' not found in tile texts"
    )

    # Test with None value
    fig = pmv.ptable_heatmap(values, element_symbol_map=None)
    tile_texts = [anno.text for anno in fig.layout.annotations if anno.text]
    for elem in values:
        assert any(f"{elem}</span>" in text for text in tile_texts), (
            f"Original symbol {elem} not found in tile texts for "
            "element_symbol_map=None"
        )


def test_ptable_heatmap_colorbar() -> None:
    """Test colorbar customization in ptable_heatmap."""
    data = {"Fe": 1.234, "O": 5.678}

    # Test colorbar title and formatting
    colorbar = dict(
        title="Test Title", tickformat=".2f", orientation="v", len=0.8, x=1.1
    )

    fig = pmv.ptable_heatmap(data, colorbar=colorbar)

    # Get the colorbar from the figure
    colorbar_trace = next(trace for trace in fig.data if hasattr(trace, "colorbar"))
    actual_colorbar = colorbar_trace.colorbar

    # Check colorbar properties were set correctly
    assert actual_colorbar.title.text == "<br><br>Test Title"
    assert actual_colorbar.tickformat == ".2f"
    assert actual_colorbar.orientation == "v"
    assert actual_colorbar.len == 0.8
    assert actual_colorbar.x == 1.1

    # Test horizontal colorbar title formatting
    h_colorbar = dict(title="Horizontal Title", orientation="h", y=0.8)

    fig = pmv.ptable_heatmap(data, colorbar=h_colorbar)
    h_colorbar_trace = next(trace for trace in fig.data if hasattr(trace, "colorbar"))
    actual_h_colorbar = h_colorbar_trace.colorbar

    # Check horizontal colorbar properties
    assert actual_h_colorbar.title.text == "Horizontal Title"
    assert actual_h_colorbar.orientation == "h"
    assert actual_h_colorbar.y == 0.8

    # Test disabling colorbar
    fig = pmv.ptable_heatmap(data, show_scale=False)
    assert not any(trace.showscale for trace in fig.data)


def test_ptable_heatmap_colorbar_log_mode() -> None:
    """Regression tests for https://github.com/janosh/pymatviz/pull/334 fixing
    TypeError when passing tickvals via colorbar kwargs and duplicate tick labels.
    """
    data = {"Fe": 1.234, "O": 5.678, "Si": 9.0}

    def get_colorbar(fig: go.Figure) -> go.heatmap.ColorBar:
        return next(tr for tr in fig.data if hasattr(tr, "colorbar")).colorbar

    # Default log behavior works
    fig = pmv.ptable_heatmap(data, log=True)
    cbar = get_colorbar(fig)
    assert cbar.tickvals is not None
    assert cbar.ticktext is not None
    assert len(cbar.tickvals) == len(cbar.ticktext)

    # Custom tickvals with log=True - previously raised TypeError
    custom_tickvals = [1.0, 2.0, 5.0, 10.0]
    fig = pmv.ptable_heatmap(data, colorbar={"tickvals": custom_tickvals}, log=True)
    assert np.allclose(get_colorbar(fig).tickvals, np.log10(custom_tickvals))

    # Custom tickvals + tickformat + other kwargs all preserved
    fig = pmv.ptable_heatmap(
        data,
        colorbar={"title": "Log Scale", "tickvals": [1, 5], "len": 0.6},
        log=True,
    )
    cbar = get_colorbar(fig)
    assert "Log Scale" in cbar.title.text
    assert cbar.len == 0.6
    assert np.allclose(cbar.tickvals, np.log10([1, 5]))

    # Auto-dedupe: tick labels are unique even without custom tickformat
    data_narrow = {"Fe": 0.12, "O": 0.56, "Si": 9}
    fig = pmv.ptable_heatmap(data_narrow, log=True)
    tick_labels = get_colorbar(fig).ticktext
    assert len(tick_labels) == len(set(tick_labels)), f"Duplicate labels: {tick_labels}"

    # Custom ticktext bypasses auto-generation
    custom_labels = ["low", "mid", "high"]
    fig = pmv.ptable_heatmap(
        data, colorbar={"tickvals": [1, 5, 10], "ticktext": custom_labels}, log=True
    )
    assert list(get_colorbar(fig).ticktext) == custom_labels

    # Format type preserved during auto-dedupe (e.g. ".1g" stays general format)
    fig = pmv.ptable_heatmap(data_narrow, colorbar={"tickformat": ".1g"}, log=True)
    tick_labels = get_colorbar(fig).ticktext
    assert len(tick_labels) == len(set(tick_labels)), f"Duplicate labels: {tick_labels}"

    # Non-positive tickvals raise ValueError for log scale
    for invalid_tickvals in ([0, 1, 10], [-1, 1, 10]):
        with pytest.raises(ValueError, match="tickvals must be positive"):
            pmv.ptable_heatmap(data, colorbar={"tickvals": invalid_tickvals}, log=True)


@pytest.mark.parametrize(
    ("values", "heat_mode", "fmt", "expected_labels"),
    [
        # Default fmt with heat_mode="value" (si_fmt with ".1f")
        ({"Fe": 1.234, "O": 56.78}, "value", None, {"Fe": "1.2", "O": "56.8"}),
        ({"H": 0.00123}, "value", None, {"H": "1.2m"}),  # milli
        ({"He": 12345}, "value", None, {"He": "12.3k"}),  # kilo
        # Custom fmt with heat_mode="value"
        ({"Fe": 1.234, "O": 56.78}, "value", ".2e", {"Fe": "1.23e0", "O": "5.68e1"}),
        ({"Li": 12345.67}, "value", ".1f", {"Li": "12.3k"}),
        # Default fmt with heat_mode="percent" ('.1%')
        (
            {"Fe": 0.2512, "O": 0.7488},  # sum to 1 for easier percent calc
            "percent",
            None,
            {"Fe": "25.1%", "O": "74.9%"},
        ),
        # Custom fmt with heat_mode="percent"
        (
            {"Fe": 0.25, "O": 0.75},
            "percent",
            ".2%",
            {"Fe": "25.00%", "O": "75.00%"},
        ),
        # Integer values
        ({"C": 12, "N": 14000}, "value", None, {"C": "12.0", "N": "14.0k"}),
        ({"C": 1, "N": 3}, "percent", ".0%", {"C": "25%", "N": "75%"}),
        # Callable fmt
        (
            {"Al": 15.678, "Si": 0.02},
            "value",
            lambda x: f"{x:.1f} kg",
            {"Al": "15.7 kg", "Si": "0.0 kg"},
        ),
        (
            {"H": 0.99, "He": 12345},
            "value",
            lambda val: f"Value is {val:g}",
            {"H": "Value is 0.99", "He": "Value is 12345"},
        ),
        (
            {"O": 0.555},
            "percent",
            lambda x: f"{x * 100:.0f} out of 100",
            {"O": "100 out of 100"},
        ),
    ],
)
def test_ptable_heatmap_value_formatting(
    values: dict[str, float],
    heat_mode: Literal["value", "fraction", "percent"],
    fmt: str | Callable[[float], str] | None,
    expected_labels: dict[str, str],
) -> None:
    """Test float formatting of value labels in ptable_heatmap."""
    fig = pmv.ptable_heatmap(values, heat_mode=heat_mode, fmt=fmt)
    labels = tile_labels(fig)
    assert {elem: labels[elem] for elem in expected_labels} == expected_labels


def test_ptable_heatmap_value_edge_cases() -> None:
    """Zero values get labels+hover, NaN hidden from hover, integer counts show
    percentages, caller kwarg dicts not mutated (regressions).
    """
    border, colorbar = {"color": "blue", "width": 3}, {"len": 0.5}
    fig = pmv.ptable_heatmap({"Fe": 0, "O": 3}, border=border, colorbar=colorbar)
    assert any(">Fe</span><br>0" in anno.text for anno in fig.layout.annotations)
    hover_texts = [text for row in fig.data[-1].text for text in row if text]
    assert "Value: 0" in next(text for text in hover_texts if "Iron" in text)
    assert not any("nan" in text for text in hover_texts)  # no NaN hover leakage
    # pop()/setdefault() used to empty caller dicts, changing repeated calls
    assert border == {"color": "blue", "width": 3}
    assert colorbar == {"len": 0.5}

    def fe_hover(values: dict[str, float]) -> str:
        fig = pmv.ptable_heatmap(values)
        texts = [text for row in fig.data[-1].text for text in row]
        return next(text for text in texts if "Iron" in text)

    # integer counts show percentage in hover, non-integer data plain values
    assert fe_hover({"Fe": 2, "O": 3}) == "Iron (Fe)<br>Value: 2 (40.00%)"
    assert fe_hover({"Fe": 2.5, "O": 3}) == "Iron (Fe)<br>Value: 2.5"
