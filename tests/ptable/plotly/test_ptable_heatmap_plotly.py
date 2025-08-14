from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.exceptions import PlotlyError
from plotly.graph_objs import Figure

import pymatviz as pmv
from pymatviz.enums import ElemCountMode, Key
from pymatviz.utils import df_ptable


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal


def test_ptable_heatmap_plotly(glass_formulas: list[str]) -> None:
    fig = pmv.ptable_heatmap_plotly(glass_formulas)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) == 18 * 10, (
        "not all periodic table tiles have annotations"
    )
    assert sum(anno.text != "" for anno in fig.layout.annotations) == 118, (
        "no annotations should be empty"
    )

    # test hover_props and show_values=False
    pmv.ptable_heatmap_plotly(
        glass_formulas,
        hover_props=("atomic_mass", "atomic_number", "density"),
        show_values=False,
    )
    pmv.ptable_heatmap_plotly(
        glass_formulas,
        hover_data="density = " + df_ptable[Key.density].astype(str) + " g/cm^3",
    )
    # test label_map as dict
    fig = pmv.ptable_heatmap_plotly(
        df_ptable[Key.density], fmt=".1f", label_map={"0": "zero"}
    )
    # test label_map as callable
    pmv.ptable_heatmap_plotly(
        df_ptable[Key.density],
        fmt=".1f",
        label_map=lambda x: "meaning of life" if x == 42 else x,
    )

    pmv.ptable_heatmap_plotly(glass_formulas, heat_mode="percent")

    # test log color scale with -1, 0, 1 and random negative value
    for val in (-9.72, -1, 0, 1):
        pmv.ptable_heatmap_plotly([f"H{val}", "O2"], log=True)
        df_ptable["tmp"] = val
        fig = pmv.ptable_heatmap_plotly(df_ptable["tmp"], log=True)
        assert isinstance(fig, go.Figure)
        heatmap_trace = fig.data[-1]
        assert heatmap_trace.colorbar.title.text == "tmp"
        c_scale = heatmap_trace.colorscale
        assert isinstance(c_scale, tuple)
        assert isinstance(c_scale[0], tuple)
        assert isinstance(c_scale[0][0], float)
        assert isinstance(c_scale[0][1], str)
        assert val <= max(c[0] for c in c_scale)

    with pytest.raises(TypeError, match="should be string, list of strings or list"):
        # test that bad colorscale raises ValueError
        pmv.ptable_heatmap_plotly(glass_formulas, colorscale=lambda: "bad scale")  # type: ignore[arg-type]

    # test that unknown builtin colorscale raises ValueError
    with pytest.raises(PlotlyError, match="Colorscale foobar is not a built-in scale"):
        pmv.ptable_heatmap_plotly(glass_formulas, colorscale="foobar")

    with pytest.raises(ValueError, match="Combining log color scale and"):
        pmv.ptable_heatmap_plotly(glass_formulas, log=True, heat_mode="percent")


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
        (["Fe"], "fraction", True, False, 8, ("orange", "pink"), 2),
        (["Xe"], "percent", True, True, None, ["brown", "cyan"], 0.2),
        ([], "value", False, True, None, ["red"], 1.5),
        (["O"], "fraction", False, True, 12, ("black", "white"), 0.8),
    ],
)
def test_ptable_heatmap_plotly_kwarg_combos(
    glass_formulas: list[str],
    exclude_elements: Sequence[str],
    heat_mode: Literal["value", "fraction", "percent"],
    log: bool,
    show_scale: bool,
    font_size: int | None,
    font_colors: Sequence[str],
    scale: float,
) -> None:
    if log and heat_mode != "value":
        pytest.skip("log scale only supported for heat_mode='value'")
    fig = pmv.ptable_heatmap_plotly(
        glass_formulas,
        exclude_elements=exclude_elements,
        heat_mode=heat_mode,
        log=log,
        show_scale=show_scale,
        font_size=font_size,
        font_colors=font_colors,
        scale=scale,
    )
    assert isinstance(fig, go.Figure)

    # Additional assertions to check if the parameters are correctly applied
    if exclude_elements:
        assert all(elem not in fig.data[-1].z for elem in exclude_elements)

    if font_size:
        assert fig.layout.font.size == font_size * scale

    if len(font_colors) == 2:
        assert all(anno.font.color in font_colors for anno in fig.layout.annotations), (
            f"{font_colors=}"
        )
    elif len(font_colors) == 1:
        assert all(
            anno.font.color == font_colors[0] for anno in fig.layout.annotations
        ), f"{font_colors=}"

    # Add assertions for scale
    assert fig.layout.width == pytest.approx(900 * scale)
    assert fig.layout.height == pytest.approx(500 * scale)
    if font_size:
        assert fig.layout.font.size == pytest.approx(font_size * scale)
    else:
        assert fig.layout.font.size == pytest.approx(12 * scale)


@pytest.mark.parametrize(
    "c_scale", ["Viridis", "Jet", ("blue", "red"), ((0, "blue"), (1, "red"))]
)
def test_ptable_heatmap_plotly_colorscale(c_scale: str) -> None:
    values = {"Fe": 2, "O": 3}
    fig = pmv.ptable_heatmap_plotly(values, colorscale=c_scale)
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
def test_ptable_heatmap_plotly_color_bar(
    glass_formulas: list[str], colorbar: dict[str, Any]
) -> None:
    fig = pmv.ptable_heatmap_plotly(glass_formulas, colorbar=colorbar)
    # check color bar has expected length
    assert fig.data[-1].colorbar.len == colorbar.get("len", 0.4)
    # check color bar has expected title side
    assert fig.data[-1].colorbar.title.side == (
        "right" if colorbar.get("orientation") == "v" else "top"
    )


@pytest.mark.parametrize(
    "cscale_range", [(None, None), (None, 10), (2, None), (2, 87123)]
)
def test_ptable_heatmap_plotly_cscale_range(
    cscale_range: tuple[float | None, float | None],
) -> None:
    fig = pmv.ptable_heatmap_plotly(df_ptable[Key.density], cscale_range=cscale_range)
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


def test_ptable_heatmap_plotly_cscale_range_raises() -> None:
    cscale_range = (0, 10, 20)
    with pytest.raises(
        ValueError, match=re.escape(f"{cscale_range=} should have length 2")
    ):
        pmv.ptable_heatmap_plotly(df_ptable[Key.density], cscale_range=cscale_range)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "label_map",
    [None, False, {"1.0": "one", "2.0": "two", "3.0": "three", np.nan: "N/A"}],
)
def test_ptable_heatmap_plotly_label_map(
    label_map: dict[str, str] | Literal[False] | None,
) -> None:
    elem_vals = dict(Al=1.0, Cr=2.0, Fe=3.0, Ni=np.nan)
    fig = pmv.ptable_heatmap_plotly(elem_vals, label_map=label_map)
    assert isinstance(fig, go.Figure)

    # if label_map is not False, ensure mapped labels appear in figure annotations
    if label_map is not False:
        if label_map is None:
            # use default map
            label_map = dict.fromkeys([np.nan, None, "nan", "nan%"], " ")  # type: ignore[list-item]
        # check for non-empty intersection between label_map values and annotations
        # we use `val in anno.text` cause the labels are wrapped in non-matching
        # HTML <span> tags
        assert isinstance(label_map, dict)
        assert sum(
            any(val in anno.text for val in label_map.values())
            for anno in fig.layout.annotations
        )


@pytest.mark.parametrize("mode", ["value", "fraction", "percent"])
def test_ptable_heatmap_plotly_heat_modes(
    mode: Literal["value", "fraction", "percent"],
) -> None:
    values = {"Fe": 2, "O": 3, "H": 1}
    fig = pmv.ptable_heatmap_plotly(values, heat_mode=mode)
    assert fig.data[-1].zmax is not None
    if mode == "value":
        assert fig.data[-1].zmax == 3
    elif mode == "fraction":
        assert fig.data[-1].zmax == pytest.approx(0.5)
    elif mode == "percent":
        assert fig.data[-1].zmax == pytest.approx(50)


def test_ptable_heatmap_plotly_show_values() -> None:
    values = {"Fe": 2, "O": 3}
    fig = pmv.ptable_heatmap_plotly(values, show_values=False)
    for text in (
        "<span style='font-weight: bold; font-size: 18.0;'>O</span>",
        "<span style='font-weight: bold; font-size: 18.0;'>Fe</span>",
    ):
        assert text in [anno.text for anno in fig.layout.annotations if anno.text]


def test_ptable_heatmap_plotly_exclude_elements() -> None:
    values = {"Fe": 2, "O": 3, "H": 1}
    fig = pmv.ptable_heatmap_plotly(values, exclude_elements=["O"])
    assert "excl." in str(fig.layout.annotations)


def test_ptable_heatmap_plotly_series_input() -> None:
    values = pd.Series({"Fe": 2, "O": 3, "H": 1})
    fig = pmv.ptable_heatmap_plotly(values)
    assert isinstance(fig, Figure)


def test_ptable_heatmap_plotly_count_modes() -> None:
    values = ("FeO", "Fe2O3", "H2O")
    for mode in ElemCountMode:
        fig = pmv.ptable_heatmap_plotly(values, count_mode=mode)
        assert isinstance(fig, Figure)


def test_ptable_heatmap_plotly_hover_props() -> None:
    values = {"Fe": 2, "O": 3}
    hover_props = ["atomic_number", "atomic_mass"]
    fig = pmv.ptable_heatmap_plotly(values, hover_props=hover_props)
    assert all(prop in str(fig.data[-1].text) for prop in hover_props)


def test_ptable_heatmap_plotly_custom_label_map() -> None:
    values = {"Fe": 2, "O": 3, "H": np.nan}
    label_map = lambda label: {2: "High", 3: "Low"}.get(float(label), label)  # type: ignore[call-overload]
    fig = pmv.ptable_heatmap_plotly(values, label_map=label_map)
    annos = [
        anno.text
        for anno in fig.layout.annotations
        if anno.text.endswith(("High", "Low"))
    ]
    assert len(annos) == 2, f"{len(annos)=}"


def test_ptable_heatmap_plotly_error_cases() -> None:
    with pytest.raises(
        ValueError, match=re.escape("cscale_range=(0,) should have length 2")
    ):
        pmv.ptable_heatmap_plotly({"Fe": 2, "O": 3}, cscale_range=(0,))  # type: ignore[arg-type]

    hover_props = ("atomic_mass", bad_hover_prop := "invalid_prop")
    with pytest.raises(ValueError, match=f"Unsupported hover_props: {bad_hover_prop}"):
        pmv.ptable_heatmap_plotly({"Fe": 2, "O": 3}, hover_props=hover_props)


@pytest.mark.parametrize("heat_mode", ["value", "fraction", "percent"])
def test_ptable_heatmap_plotly_color_range(
    heat_mode: Literal["value", "fraction", "percent"],
) -> None:
    values = {"Fe": 1, "O": 50, "H": 100}
    fig = pmv.ptable_heatmap_plotly(values, heat_mode=heat_mode)
    heatmap_trace = fig.data[-1]
    non_nan_values = [v for v in heatmap_trace.z.flat if not np.isnan(v)]
    assert min(non_nan_values) == heatmap_trace.zmin
    assert max(non_nan_values) == heatmap_trace.zmax


def test_ptable_heatmap_plotly_all_elements() -> None:
    values = {elem: idx for idx, elem in enumerate(df_ptable.index)}
    fig = pmv.ptable_heatmap_plotly(values)
    heatmap_trace = fig.data[-1]
    assert not np.isnan(heatmap_trace.zmax)
    assert heatmap_trace.zmax == len(df_ptable) - 1


def test_ptable_heatmap_plotly_hover_tooltips() -> None:
    # Test with non-integer values
    float_values = {"Fe": 0.2, "O": 0.3, "H": 0.5}

    # Test value mode
    fig = pmv.ptable_heatmap_plotly(float_values, heat_mode="value")
    hover_texts = fig.data[-1].text.flat

    for elem_symb, value in float_values.items():
        elem_name = df_ptable.loc[elem_symb, "name"]
        hover_text = next(text for text in hover_texts if text.startswith(elem_name))
        assert hover_text == f"{elem_name} ({elem_symb})<br>Value: {value:.3g}"

    # Test fraction and percent modes
    for heat_mode in ["fraction", "percent"]:
        fig = pmv.ptable_heatmap_plotly(float_values, heat_mode=heat_mode)  # type: ignore[arg-type]
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
        fig = pmv.ptable_heatmap_plotly(int_values, heat_mode=heat_mode)  # type: ignore[arg-type]
        hover_texts = fig.data[-1].text.flat

        for elem_symb, value in int_values.items():
            elem_name = df_ptable.loc[elem_symb, "name"]
            hover_text = next(
                text for text in hover_texts if text.startswith(elem_name)
            )

            if heat_mode == "value":
                expected_hover_text = f"{elem_name} ({elem_symb})<br>Value: {value}"
            else:
                expected_hover_text = (
                    f"{elem_name} ({elem_symb})<br>Percentage: "
                    f"{value / sum(int_values.values()):.2%} ({value})"
                )

    # Test with excluded elements
    fig = pmv.ptable_heatmap_plotly(int_values, exclude_elements=["O"])
    hover_texts = fig.data[-1].text.flat
    o_hover_text = next(text for text in hover_texts if text.startswith("Oxygen"))
    assert "Value:" not in o_hover_text

    # Test with hover_props
    fig = pmv.ptable_heatmap_plotly(
        int_values, hover_props=["atomic_number", "atomic_mass"]
    )
    hover_texts = fig.data[-1].text.flat
    for hover_text in hover_texts:
        if not hover_text:
            continue
        assert "atomic_number" in hover_text
        assert "atomic_mass" in hover_text

    # Test with hover_data
    hover_data = {"Fe": "Custom Fe data", "O": "Custom O data"}
    fig = pmv.ptable_heatmap_plotly(int_values, hover_data=hover_data)
    hover_texts = fig.data[-1].text.flat
    fe_hover_text = next(text for text in hover_texts if text.startswith("Iron"))
    assert "Custom Fe data" in fe_hover_text


def test_ptable_heatmap_plotly_element_symbol_map() -> None:
    values = {"Fe": 2, "O": 3, "H": 1}
    element_symbol_map = {"Fe": "Iron", "O": "Oxygen", "H": "Hydrogen"}
    fig = pmv.ptable_heatmap_plotly(values, element_symbol_map=element_symbol_map)

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
    fig = pmv.ptable_heatmap_plotly(values, element_symbol_map=partial_map)
    tile_texts = [anno.text for anno in fig.layout.annotations if anno.text]
    assert any("This be Iron" in text for text in tile_texts), (
        "Custom symbol not found in tile texts"
    )
    assert any("O</span>" in text for text in tile_texts), (
        "Original symbol 'O' not found in tile texts"
    )

    # Test with None value
    fig = pmv.ptable_heatmap_plotly(values, element_symbol_map=None)
    tile_texts = [anno.text for anno in fig.layout.annotations if anno.text]
    for elem in values:
        assert any(f"{elem}</span>" in text for text in tile_texts), (
            f"Original symbol {elem} not found in tile texts for "
            "element_symbol_map=None"
        )


def test_ptable_heatmap_plotly_colorbar() -> None:
    """Test colorbar customization in ptable_heatmap_plotly."""
    data = {"Fe": 1.234, "O": 5.678}

    # Test colorbar title and formatting
    colorbar = dict(
        title="Test Title", tickformat=".2f", orientation="v", len=0.8, x=1.1
    )

    fig = pmv.ptable_heatmap_plotly(data, colorbar=colorbar)

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

    fig = pmv.ptable_heatmap_plotly(data, colorbar=h_colorbar)
    h_colorbar_trace = next(trace for trace in fig.data if hasattr(trace, "colorbar"))
    actual_h_colorbar = h_colorbar_trace.colorbar

    # Check horizontal colorbar properties
    assert actual_h_colorbar.title.text == "Horizontal Title"
    assert actual_h_colorbar.orientation == "h"
    assert actual_h_colorbar.y == 0.8

    # Test disabling colorbar
    fig = pmv.ptable_heatmap_plotly(data, show_scale=False)
    assert not any(trace.showscale for trace in fig.data)


def test_ptable_heatmap_plotly_value_formatting() -> None:
    """Test float formatting of value labels in ptable_heatmap_plotly."""
    # Test cases: (values, heat_mode, fmt, expected_labels)
    test_cases = [
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
    ]

    for values, heat_mode, fmt, expected_labels in test_cases:
        fig = pmv.ptable_heatmap_plotly(
            values,
            heat_mode=heat_mode,  # type: ignore[arg-type]
            fmt=fmt,
            font_size=10,  # smaller font for stable span style
        )
        annotations = fig.layout.annotations
        # Extract symbol and value from annotation text:
        # "<span style='font-weight: bold; font-size: 15.0;'>SYM</span><br>VALUE"
        # or "<span style='font-weight: bold; font-size: 15.0;'>SYM</span>" if no value
        # Need to handle cases where value might be missing or excluded
        found_labels = {}
        for anno in annotations:
            text = anno.text
            if not text or "<br>" not in text:
                continue  # Skip empty annotations or those without a value part

            # Extract symbol and value label
            match = re.match(
                r"<span .*?>(?P<symbol>[A-Za-z0-9]+)</span><br>(?P<label>.*)", text
            )
            if match:
                symbol = match.group("symbol")
                label = match.group("label")
                if symbol in expected_labels:  # Only check elements we provided
                    found_labels[symbol] = label

        for elem, expected_label in expected_labels.items():
            assert found_labels.get(elem) == expected_label, (
                f"Mismatch for {elem} with {values=}, {heat_mode=}, {fmt=}: "
                f"expected '{expected_label}', got '{found_labels.get(elem)}'"
            )
