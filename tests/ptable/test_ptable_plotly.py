from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.exceptions import PlotlyError
from plotly.graph_objs import Figure

from pymatviz.enums import ElemCountMode, Key
from pymatviz.ptable import ptable_heatmap_plotly, ptable_hists_plotly
from pymatviz.utils import VALID_COLOR_ELEM_STRATEGIES, ColorElemTypeStrategy, df_ptable


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal


def test_ptable_heatmap_plotly(glass_formulas: list[str]) -> None:
    fig = ptable_heatmap_plotly(glass_formulas)
    assert isinstance(fig, go.Figure)
    assert (
        len(fig.layout.annotations) == 18 * 10
    ), "not all periodic table tiles have annotations"
    assert (
        sum(anno.text != "" for anno in fig.layout.annotations) == 118
    ), "no annotations should be empty"

    # test hover_props and show_values=False
    ptable_heatmap_plotly(
        glass_formulas,
        hover_props=("atomic_mass", "atomic_number", "density"),
        show_values=False,
    )
    ptable_heatmap_plotly(
        glass_formulas,
        hover_data="density = " + df_ptable[Key.density].astype(str) + " g/cm^3",
    )
    # test label_map as dict
    fig = ptable_heatmap_plotly(
        df_ptable[Key.density], fmt=".1f", label_map={"0": "zero"}
    )
    # test label_map as callable
    ptable_heatmap_plotly(
        df_ptable[Key.density],
        fmt=".1f",
        label_map=lambda x: "meaning of life" if x == 42 else x,
    )

    ptable_heatmap_plotly(glass_formulas, heat_mode="percent")

    # test log color scale with -1, 0, 1 and random negative value
    for val in (-9.72, -1, 0, 1):
        ptable_heatmap_plotly([f"H{val}", "O2"], log=True)
        df_ptable["tmp"] = val
        fig = ptable_heatmap_plotly(df_ptable["tmp"], log=True)
        assert isinstance(fig, go.Figure)
        heatmap_trace = fig.data[-1]
        assert heatmap_trace.colorbar.title.text == "tmp<br>"
        c_scale = heatmap_trace.colorscale
        assert isinstance(c_scale, tuple)
        assert isinstance(c_scale[0], tuple)
        assert isinstance(c_scale[0][0], float)
        assert isinstance(c_scale[0][1], str)
        assert val <= max(c[0] for c in c_scale)

    with pytest.raises(TypeError, match="should be string, list of strings or list"):
        # test that bad colorscale raises ValueError
        ptable_heatmap_plotly(glass_formulas, colorscale=lambda: "bad scale")  # type: ignore[arg-type]

    # test that unknown builtin colorscale raises ValueError
    with pytest.raises(PlotlyError, match="Colorscale foobar is not a built-in scale"):
        ptable_heatmap_plotly(glass_formulas, colorscale="foobar")

    with pytest.raises(ValueError, match="Combining log color scale and"):
        ptable_heatmap_plotly(glass_formulas, log=True, heat_mode="percent")


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
    fig = ptable_heatmap_plotly(
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
        assert all(
            anno.font.color in font_colors for anno in fig.layout.annotations
        ), f"{font_colors=}"
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
    fig = ptable_heatmap_plotly(values, colorscale=c_scale)
    clr_scale_start = fig.data[-1].colorscale[0]
    assert clr_scale_start == {
        "Viridis": (0, "#440154"),
        "Jet": (0, "rgb(0,0,131)"),
        ("blue", "red"): (0, "blue"),
        ((0, "blue"), (1, "red")): (0, "blue"),
    }.get(c_scale), f"{c_scale=}, {clr_scale_start=}"


@pytest.mark.parametrize(
    "color_bar", [{}, dict(orientation="v", len=0.8), dict(orientation="h", len=0.3)]
)
def test_ptable_heatmap_plotly_color_bar(
    glass_formulas: list[str], color_bar: dict[str, Any]
) -> None:
    fig = ptable_heatmap_plotly(glass_formulas, color_bar=color_bar)
    # check color bar has expected length
    assert fig.data[-1].colorbar.len == color_bar.get("len", 0.4)
    # check color bar has expected title side
    assert fig.data[-1].colorbar.title.side == (
        "right" if color_bar.get("orientation") == "v" else "top"
    )


@pytest.mark.parametrize(
    "cscale_range", [(None, None), (None, 10), (2, None), (2, 87123)]
)
def test_ptable_heatmap_plotly_cscale_range(
    cscale_range: tuple[float | None, float | None],
) -> None:
    fig = ptable_heatmap_plotly(df_ptable[Key.density], cscale_range=cscale_range)
    trace = fig.data[-1]
    assert "colorbar" in trace
    # check for correct color bar range
    data_min, data_max = df_ptable[Key.density].min(), df_ptable[Key.density].max()
    if cscale_range == (None, None):
        # if both None, range is dynamic based on plotted data
        assert trace.zmin == pytest.approx(data_min)
        assert trace.zmax == pytest.approx(data_max)
    else:
        assert trace.zmin == pytest.approx(
            cscale_range[0] or data_min
        ), f"{cscale_range=}"
        assert trace.zmax == pytest.approx(
            cscale_range[1] or data_max
        ), f"{cscale_range=}"


def test_ptable_heatmap_plotly_cscale_range_raises() -> None:
    cscale_range = (0, 10, 20)
    with pytest.raises(
        ValueError, match=re.escape(f"{cscale_range=} should have length 2")
    ):
        ptable_heatmap_plotly(df_ptable[Key.density], cscale_range=cscale_range)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "label_map",
    [None, False, {"1.0": "one", "2.0": "two", "3.0": "three", np.nan: "N/A"}],
)
def test_ptable_heatmap_plotly_label_map(
    label_map: dict[str, str] | Literal[False] | None,
) -> None:
    elem_vals = dict(Al=1.0, Cr=2.0, Fe=3.0, Ni=np.nan)
    fig = ptable_heatmap_plotly(elem_vals, label_map=label_map)
    assert isinstance(fig, go.Figure)

    # if label_map is not False, ensure mapped labels appear in figure annotations
    if label_map is not False:
        if label_map is None:
            # use default map
            label_map = dict.fromkeys([np.nan, None, "nan", "nan%"], " ")  # type: ignore[list-item]
        # check for non-empty intersection between label_map values and annotations
        # we use `val in anno.text` cause the labels are wrapped in non-matching
        # HTML <span> tags
        assert sum(
            any(val in anno.text for val in label_map.values())
            for anno in fig.layout.annotations
        )


@pytest.mark.parametrize("mode", ["value", "fraction", "percent"])
def test_ptable_heatmap_plotly_heat_modes(
    mode: Literal["value", "fraction", "percent"],
) -> None:
    values = {"Fe": 2, "O": 3, "H": 1}
    fig = ptable_heatmap_plotly(values, heat_mode=mode)
    assert fig.data[-1].zmax is not None
    if mode == "value":
        assert fig.data[-1].zmax == 3
    elif mode == "fraction":
        assert fig.data[-1].zmax == pytest.approx(0.5)
    elif mode == "percent":
        assert fig.data[-1].zmax == pytest.approx(50)


def test_ptable_heatmap_plotly_show_values() -> None:
    values = {"Fe": 2, "O": 3}
    fig = ptable_heatmap_plotly(values, show_values=False)
    for text in (
        "<span style='font-weight: bold; font-size: 18.0;'>O</span>",
        "<span style='font-weight: bold; font-size: 18.0;'>Fe</span>",
    ):
        assert text in [anno.text for anno in fig.layout.annotations if anno.text]


def test_ptable_heatmap_plotly_exclude_elements() -> None:
    values = {"Fe": 2, "O": 3, "H": 1}
    fig = ptable_heatmap_plotly(values, exclude_elements=["O"])
    assert "excl." in str(fig.layout.annotations)


def test_ptable_heatmap_plotly_series_input() -> None:
    values = pd.Series({"Fe": 2, "O": 3, "H": 1})
    fig = ptable_heatmap_plotly(values)
    assert isinstance(fig, Figure)


def test_ptable_heatmap_plotly_list_input() -> None:
    values = ["FeO", "Fe2O3", "H2O"]
    fig = ptable_heatmap_plotly(values)
    assert isinstance(fig, Figure)


def test_ptable_heatmap_plotly_count_modes() -> None:
    values = ["FeO", "Fe2O3", "H2O"]
    for mode in ElemCountMode:
        fig = ptable_heatmap_plotly(values, count_mode=mode)
        assert isinstance(fig, Figure)


def test_ptable_heatmap_plotly_hover_props() -> None:
    values = {"Fe": 2, "O": 3}
    hover_props = ["atomic_number", "atomic_mass"]
    fig = ptable_heatmap_plotly(values, hover_props=hover_props)
    assert all(prop in str(fig.data[-1].text) for prop in hover_props)


def test_ptable_heatmap_plotly_custom_label_map() -> None:
    values = {"Fe": 2, "O": 3, "H": np.nan}
    label_map = lambda label: {2: "High", 3: "Low"}.get(float(label), label)  # type: ignore[call-overload]
    fig = ptable_heatmap_plotly(values, label_map=label_map)
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
        ptable_heatmap_plotly({"Fe": 2, "O": 3}, cscale_range=(0,))  # type: ignore[arg-type]

    hover_props = ("atomic_mass", bad_hover_prop := "invalid_prop")
    with pytest.raises(ValueError, match=f"Unsupported hover_props: {bad_hover_prop}"):
        ptable_heatmap_plotly({"Fe": 2, "O": 3}, hover_props=hover_props)


@pytest.mark.parametrize("heat_mode", ["value", "fraction", "percent"])
def test_ptable_heatmap_plotly_color_range(
    heat_mode: Literal["value", "fraction", "percent"],
) -> None:
    values = {"Fe": 1, "O": 50, "H": 100}
    fig = ptable_heatmap_plotly(values, heat_mode=heat_mode)
    heatmap_trace = fig.data[-1]
    non_nan_values = [v for v in heatmap_trace.z.flat if not np.isnan(v)]
    assert min(non_nan_values) == heatmap_trace.zmin
    assert max(non_nan_values) == heatmap_trace.zmax


def test_ptable_heatmap_plotly_all_elements() -> None:
    values = {elem: idx for idx, elem in enumerate(df_ptable.index)}
    fig = ptable_heatmap_plotly(values)
    heatmap_trace = fig.data[-1]
    assert not np.isnan(heatmap_trace.zmax)
    assert heatmap_trace.zmax == len(df_ptable) - 1


def test_ptable_heatmap_plotly_hover_tooltips() -> None:
    # Test with non-integer values
    float_values = {"Fe": 0.2, "O": 0.3, "H": 0.5}

    # Test value mode
    fig = ptable_heatmap_plotly(float_values, heat_mode="value")
    hover_texts = fig.data[-1].text.flat

    for elem_symb, value in float_values.items():
        elem_name = df_ptable.loc[elem_symb, "name"]
        hover_text = next(text for text in hover_texts if text.startswith(elem_name))
        assert hover_text == f"{elem_name} ({elem_symb})<br>Value: {value:.3g}"

    # Test fraction and percent modes
    for heat_mode in ["fraction", "percent"]:
        fig = ptable_heatmap_plotly(float_values, heat_mode=heat_mode)  # type: ignore[arg-type]
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
        fig = ptable_heatmap_plotly(int_values, heat_mode=heat_mode)  # type: ignore[arg-type]
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
    fig = ptable_heatmap_plotly(int_values, exclude_elements=["O"])
    hover_texts = fig.data[-1].text.flat
    o_hover_text = next(text for text in hover_texts if text.startswith("Oxygen"))
    assert "Value:" not in o_hover_text

    # Test with hover_props
    fig = ptable_heatmap_plotly(
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
    fig = ptable_heatmap_plotly(int_values, hover_data=hover_data)
    hover_texts = fig.data[-1].text.flat
    fe_hover_text = next(text for text in hover_texts if text.startswith("Iron"))
    assert "Custom Fe data" in fe_hover_text


def test_ptable_heatmap_plotly_element_symbol_map() -> None:
    values = {"Fe": 2, "O": 3, "H": 1}
    element_symbol_map = {"Fe": "Iron", "O": "Oxygen", "H": "Hydrogen"}
    fig = ptable_heatmap_plotly(values, element_symbol_map=element_symbol_map)

    # Check if custom symbols are used in tile texts
    tile_texts = [anno.text for anno in fig.layout.annotations if anno.text]
    for custom_symbol in element_symbol_map.values():
        assert any(
            custom_symbol in text for text in tile_texts
        ), f"Custom symbol {custom_symbol} not found in tile texts"

    # Check if original element symbols are still used in hover texts
    hover_texts = fig.data[-1].text.flat
    for elem in values:
        hover_text = next(
            text for text in hover_texts if text.startswith(df_ptable.loc[elem, "name"])
        )
        assert (
            f"({elem})" in hover_text
        ), f"Original symbol {elem} not found in hover text"

    # Test with partial mapping
    partial_map = {"Fe": "This be Iron"}
    fig = ptable_heatmap_plotly(values, element_symbol_map=partial_map)
    tile_texts = [anno.text for anno in fig.layout.annotations if anno.text]
    assert any(
        "This be Iron" in text for text in tile_texts
    ), "Custom symbol not found in tile texts"
    assert any(
        "O</span>" in text for text in tile_texts
    ), "Original symbol 'O' not found in tile texts"

    # Test with None value
    fig = ptable_heatmap_plotly(values, element_symbol_map=None)
    tile_texts = [anno.text for anno in fig.layout.annotations if anno.text]
    for elem in values:
        assert any(
            f"{elem}</span>" in text for text in tile_texts
        ), f"Original symbol {elem} not found in tile texts for element_symbol_map=None"


@pytest.mark.parametrize(
    ("data", "bins", "x_range", "log", "colorscale"),
    [
        # Basic case
        ({"Fe": [1, 2], "O": [3, 4]}, 10, None, False, "RdBu"),
        # Single element, log scale
        (pd.DataFrame({"Fe": [0.1, 0.2]}), 5, (0, 1), True, "Viridis"),
        ({"H": [0.1, 0.2], "He": [0.3, 0.4]}, 5, None, True, "Turbo"),
    ],
)
def test_ptable_hists_plotly_basic(
    data: pd.DataFrame | dict[str, list[float]],
    bins: int,
    x_range: tuple[float | None, float | None] | None,
    log: bool,
    colorscale: str,
) -> None:
    fig = ptable_hists_plotly(
        data, bins=bins, x_range=x_range, log=log, colorscale=colorscale
    )
    assert isinstance(fig, go.Figure)
    # Count only histogram traces (exclude colorbar trace)
    n_hist_traces = sum(1 for trace in fig.data if isinstance(trace, go.Histogram))
    n_elements = len(data if isinstance(data, dict) else data.columns)
    assert n_hist_traces == n_elements


@pytest.mark.parametrize(
    ("font_size", "scale", "symbol_kwargs", "anno_kwargs"),
    [
        (12, 1.0, dict(x=0, y=1), dict(x=1, y=1)),  # Most common case
        (None, 0.5, dict(x=0.5, y=0.5), dict(x=0.5, y=0.5)),  # Test None font_size
    ],
)
def test_ptable_hists_plotly_layout(
    font_size: int | None,
    scale: float,
    symbol_kwargs: dict[str, Any],
    anno_kwargs: dict[str, Any],
) -> None:
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}
    fig = ptable_hists_plotly(
        data,
        font_size=font_size,
        scale=scale,
        symbol_kwargs=symbol_kwargs,
        anno_kwargs=anno_kwargs,
    )
    assert isinstance(fig, go.Figure)
    if font_size:
        assert any(
            anno.font.size == font_size * scale
            for anno in fig.layout.annotations
            if anno.font is not None
        )


@pytest.mark.parametrize(
    "color_elem_strategy",
    VALID_COLOR_ELEM_STRATEGIES,
)
def test_ptable_hists_plotly_element_colors(
    color_elem_strategy: ColorElemTypeStrategy,
) -> None:
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}
    fig = ptable_hists_plotly(data, color_elem_strategy=color_elem_strategy)
    assert isinstance(fig, go.Figure)


def test_ptable_hists_plotly_custom_annotations() -> None:
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}
    anno_text = {"Fe": "Iron", "O": "Oxygen"}
    anno_kwargs = {"font": {"size": 14, "color": "red"}}
    fig = ptable_hists_plotly(data, anno_text=anno_text, anno_kwargs=anno_kwargs)
    assert isinstance(fig, go.Figure)


def test_ptable_hists_plotly_error_cases() -> None:
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}

    # Test invalid color_elem_strategy
    with pytest.raises(
        ValueError, match="color_elem_strategy='invalid' must be one of"
    ):
        ptable_hists_plotly(data, color_elem_strategy="invalid")  # type: ignore[arg-type]

    # Test invalid scale
    with pytest.raises(ValueError, match="Invalid value of type ") as exc:
        ptable_hists_plotly(data, scale=-1.0)
    assert "An int or float in the interval [1, inf]" in str(exc.value)

    # Test invalid bins
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        ptable_hists_plotly(data, bins=0)


@pytest.mark.parametrize(
    ("element_symbol_map", "symbol_kwargs"),
    [
        ({"Fe": "Iron", "O": "Oxygen"}, {"font": {"size": 14}}),
        (None, None),
        ({"Fe": "Fe*"}, {"font": {"color": "red"}}),
    ],
)
def test_ptable_hists_plotly_symbol_customization(
    element_symbol_map: dict[str, str] | None,
    symbol_kwargs: dict[str, Any] | None,
) -> None:
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}
    fig = ptable_hists_plotly(
        data,
        element_symbol_map=element_symbol_map,
        symbol_kwargs=symbol_kwargs,
    )
    assert isinstance(fig, go.Figure)


def test_ptable_hists_plotly_subplot_kwargs() -> None:
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}
    sub_titles = ["Iron", "Oxygen"]
    subplot_kwargs = {
        "horizontal_spacing": 0.05,
        "vertical_spacing": 0.05,
        "subplot_titles": sub_titles,
    }
    fig = ptable_hists_plotly(data, subplot_kwargs=subplot_kwargs)
    assert isinstance(fig, go.Figure)
    # Verify subplot titles are set
    assert [anno.text for anno in fig.layout.annotations][:2] == sub_titles


def test_ptable_hists_plotly_hover_tooltips() -> None:
    """Test that hover tooltips show element info and histogram values."""
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}
    element_symbol_map = {"Fe": "Iron"}

    fig = ptable_hists_plotly(data, element_symbol_map=element_symbol_map)

    # Get hover templates for each histogram trace
    hover_templates = [trace.hovertemplate for trace in fig.data]

    expected_fe_template = (
        "<b>Iron</b> (Fe)<br>Range: %{x}<br>Count: %{y}<extra></extra>"
    )
    expected_o_template = "<b>O</b><br>Range: %{x}<br>Count: %{y}<extra></extra>"

    assert expected_fe_template in hover_templates
    assert expected_o_template in hover_templates


@pytest.mark.parametrize(
    ("colorbar", "expected"),
    [
        # Test default settings
        (None, dict(orientation="h", len=0.4, thickness=15)),
        # Test no colorbar
        (False, None),
        # Test horizontal colorbar with custom length
        (dict(orientation="h", len=0.4), dict(orientation="h", len=0.4)),
        # Test vertical colorbar with custom length
        (dict(orientation="v", len=0.8), dict(orientation="v", len=0.8)),
        # Test title formatting for horizontal orientation
        (
            dict(title="Test Title", orientation="h"),
            dict(title="Test Title<br>", orientation="h"),
        ),
        # Test comprehensive custom settings
        (
            dict(
                orientation="v",
                len=0.6,
                thickness=20,
                title="Custom",
                x=1.1,
                y=0.5,
            ),
            dict(
                orientation="v",
                len=0.6,
                thickness=20,
                title="<br><br>Custom",
                x=1.1,
                y=0.5,
            ),
        ),
    ],
)
def test_ptable_hists_plotly_colorbar(
    colorbar: dict[str, Any] | Literal[False] | None,
    expected: dict[str, Any] | None,
) -> None:
    """Test colorbar customization in ptable_hists_plotly including range and
    visibility."""
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}
    x_range = (-1, 5)  # Test custom range
    fig = ptable_hists_plotly(data, colorbar=colorbar, x_range=x_range)

    if colorbar is False:
        assert all(not hasattr(trace, "colorbar") for trace in fig.data)
        return

    # Find the scatter trace with the colorbar (should be last trace)
    colorbar_trace = fig.data[-1]
    assert isinstance(colorbar_trace, go.Scatter)
    assert colorbar_trace.hoverinfo == "none"
    assert colorbar_trace.showlegend is False

    # Check colorbar properties
    marker = colorbar_trace.marker
    assert marker.showscale is True

    # Check colorbar range matches x_range
    assert marker.cmin == x_range[0]
    assert marker.cmax == x_range[1]

    # Check that axes for this trace are hidden
    xaxis_num = len(fig.data)  # Last trace's axis number
    assert not fig.layout[f"xaxis{xaxis_num}"].visible
    assert not fig.layout[f"yaxis{xaxis_num}"].visible

    # Check colorbar settings
    if expected is not None:
        for key, value in expected.items():
            if key == "title":
                assert marker.colorbar.title.text == value
            else:
                assert getattr(marker.colorbar, key) == value
