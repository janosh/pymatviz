from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pymatgen.core import Element

import pymatviz as pmv


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal


@pytest.mark.parametrize("on_empty", ["hide", "show"])
def test_ptable_heatmap_splits_basic(on_empty: Literal["hide", "show"]) -> None:
    """Test basic functionality of ptable_heatmap_splits."""
    # Test grid orientation separately with 4 splits
    data_4_split = {"Fe": [1, 2, 3, 4]}
    fig = pmv.ptable_heatmap_splits(
        data_4_split, orientation="grid", split_value_fmt=".0f", on_empty=on_empty
    )
    if on_empty == "hide":
        assert len(fig.data) == 5
    else:
        assert len(fig.data) > 5

    # grid sections are in reading order and each value annotation sits inside
    # its own section (regression: sections were placed bottom-row first,
    # putting every annotation in the vertically-opposite quadrant)
    fe_ref = next(anno.xref for anno in fig.layout.annotations if anno.text == "1")
    sections = [
        trace for trace in fig.data if trace.fill == "toself" and trace.xaxis == fe_ref
    ]
    anno_xy = {a.text: (a.x, a.y) for a in fig.layout.annotations if a.text.isdigit()}
    for idx, trace in enumerate(sections):
        anno_x, anno_y = anno_xy[str(idx + 1)]
        assert min(trace.x) <= anno_x <= max(trace.x), f"split {idx} x mismatch"
        assert min(trace.y) <= anno_y <= max(trace.y), f"split {idx} y mismatch"
    # first two splits are the top row (y in upper half), last two the bottom
    assert [min(trc.y) for trc in sections] == [0.5, 0.5, 0, 0]


def test_ptable_heatmap_splits_f_block() -> None:
    """Test that f-block elements are only hidden in 'auto' mode when there's no data
    for any f-block element.
    """
    # Data with no f-block elements
    data_no_f = {"Fe": [1, 2], "O": [3, 4], "H": [0.5, 1.5]}

    rows_per_mode: dict[bool | Literal["auto"], int] = {True: 7, "auto": 7, False: 10}

    for hide_f_block, expected_n_rows in rows_per_mode.items():
        fig = pmv.ptable_heatmap_splits(data_no_f, hide_f_block=hide_f_block)
        assert fig._grid_ref is not None
        n_rows = len(fig._grid_ref)
        assert n_rows == expected_n_rows, f"{n_rows=}, {hide_f_block=}"

    # Data including f-block elements
    data_with_f = {"Fe": [1, 2], "La": [3, 4], "U": [5, 6]}

    rows_per_mode["auto"] = 10
    for hide_f_block, expected_n_rows in rows_per_mode.items():
        fig = pmv.ptable_heatmap_splits(data_with_f, hide_f_block=hide_f_block)
        assert fig._grid_ref is not None
        n_rows = len(fig._grid_ref)
        assert n_rows == expected_n_rows, f"{n_rows=}, {hide_f_block=}"

    # Check that f-block elements are shown when hide_f_block="auto" and data includes
    # f-block elements
    fig_with_f = pmv.ptable_heatmap_splits(data_with_f, hide_f_block="auto")
    anno_texts = [
        anno.text
        for anno in fig_with_f.layout.annotations
        if hasattr(anno, "text")
        # by default, filter out individual split value annotations if
        # show_split_values=True
        and not (anno.text.isnumeric() or anno.text.replace(".", "", 1).isdigit())
    ]
    expected = [
        "Fe",
        "<b>Iron</b><br>Split 1: 1<br>Split 2: 2",
        "La",
        "<b>Lanthanum</b><br>Split 1: 3<br>Split 2: 4",
        "U",
        "<b>Uranium</b><br>Split 1: 5<br>Split 2: 6",
    ]
    assert anno_texts == expected, f"{anno_texts=}"

    # Check that f-block elements are hidden when hide_f_block=True
    fig_with_f_hidden = pmv.ptable_heatmap_splits(data_with_f, hide_f_block=True)
    anno_texts_hidden = [
        anno.text
        for anno in fig_with_f_hidden.layout.annotations
        if hasattr(anno, "text")
    ]

    # Fe should be present, but La and U should be hidden
    assert "Fe" in anno_texts_hidden
    assert "La" not in anno_texts_hidden
    assert "U" not in anno_texts_hidden


@pytest.mark.parametrize(
    ("orientation", "colorscale", "font_size", "scale"),
    [
        ("diagonal", "Viridis", 12, 1.0),
        ("horizontal", "RdBu", None, 0.8),
        ("vertical", "Spectral", 14, 1.2),
        ("diagonal", "Plasma", 10, 0.9),
    ],
)
def test_ptable_heatmap_splits_display_options(
    orientation: Literal["diagonal", "horizontal", "vertical", "grid"],
    colorscale: str,
    font_size: int | None,
    scale: float,
) -> None:
    """Test various display options for ptable_heatmap_splits."""
    subplot_kwargs = {
        "horizontal_spacing": 0.05,
        "vertical_spacing": 0.05,
        "subplot_titles": ["Split 1", "Split 2", "Split 3", "Split 4"],
    }
    # Test custom element symbols
    element_symbol_map = {"Fe": "Iron", "O": "Oxygen"}
    symbol_kwargs = {"font": {"size": 14, "color": "red"}}

    fig = pmv.ptable_heatmap_splits(
        {"Fe": [1, 2], "O": [3, 4], "H": [0.5, 1.5], "He": [1.5, 2.5]},
        orientation=orientation,
        colorscale=colorscale,
        font_size=font_size,
        scale=scale,
        subplot_kwargs=subplot_kwargs,
        element_symbol_map=element_symbol_map,
        symbol_kwargs=symbol_kwargs,
    )

    assert fig.layout.width == 45 * 18 * scale
    assert fig.layout.height == 45 * 7 * scale + 40
    anno_font_sizes = [
        anno.font.size for anno in fig.layout.annotations if anno.font is not None
    ]
    # Check if annotations have expected font size (symbol_kwargs.font.size takes
    # precedence over font_size)
    assert symbol_kwargs.get("font", {}).get("size") in anno_font_sizes


@pytest.mark.parametrize(
    "colorbar",
    [
        None,
        False,
        dict(orientation="v", len=0.8),
        dict(orientation="h", len=0.3),
        [{}, {}],
    ],
)
@pytest.mark.parametrize("as_dataframe", [False, True])
def test_ptable_heatmap_splits_colorbar(
    as_dataframe: bool,
    colorbar: dict[str, Any] | list[dict[str, Any]] | Literal[False] | None,
) -> None:
    """Test colorbar customization in ptable_heatmap_splits."""
    data = {"H": [1, 2], "He": [2, 3]}
    fig = pmv.ptable_heatmap_splits(
        pd.DataFrame.from_dict(data, orient="index") if as_dataframe else data,
        colorbar=colorbar,
    )
    markers = [trace.marker for trace in fig.data if trace.marker.showscale]
    fills = [trace.fillcolor for trace in fig.data if trace.fill == "toself"]
    if as_dataframe or isinstance(colorbar, list):
        assert [(marker.cmin, marker.cmax) for marker in markers] == (
            [] if colorbar is False else [(1, 2), (2, 3)]
        )
        assert fills[1] != fills[2]
    else:
        assert fills[1] == fills[2], "Equal values must share a color with one scale"
        assert [(marker.cmin, marker.cmax) for marker in markers] == (
            [] if colorbar is False else [(1, 3)]
        )


def test_ptable_heatmap_splits_annotations() -> None:
    """Test custom annotations in ptable_heatmap_splits."""
    data = {"Fe": [1, 2], "O": [3, 4], "H": [0.5, 1.5], "He": [1.5, 2.5]}

    # Test with dict annotations
    annotations: dict[str, str | dict[str, Any]] = {
        "Fe": {"text": "Iron", "font": {"size": 14, "color": "red"}},
        "O": {"text": "Oxygen", "font": {"size": 14, "color": "blue"}},
    }

    fig = pmv.ptable_heatmap_splits(data, annotations=annotations)

    # user font color wins, size comes from the default font_size
    custom_fonts = {
        anno.text: (anno.font.color, anno.font.size)
        for anno in fig.layout.annotations
        if anno.text in ("Iron", "Oxygen")
    }
    assert custom_fonts == {"Iron": ("red", 8.0), "Oxygen": ("blue", 8.0)}

    # Test with callable annotations
    def annotation_func(value: list[float] | np.ndarray) -> dict[str, Any]:
        return {"text": f"Value: {np.mean(value):.1f}"}

    fig = pmv.ptable_heatmap_splits(data, annotations=annotation_func)

    # Check for annotations with the expected text format
    value_annotations = [
        anno
        for anno in fig.layout.annotations
        if hasattr(anno, "text") and "Value:" in anno.text
    ]

    # There should be at least one value annotation
    assert len(value_annotations) == 4

    # Check that the annotations have the expected values
    expected_values = ["Value: 1.5", "Value: 3.5", "Value: 1.0", "Value: 2.0"]
    for value in expected_values:
        assert any(value in anno.text for anno in value_annotations)


@pytest.mark.parametrize(
    ("data", "kwargs", "message"),
    [
        ({}, {}, r"ptable_heatmap_splits: data={} must not be empty"),
        ({"Fe": [1]}, {}, "Number of splits 1 must be 2, 3, or 4"),
        ({"Fe": [1] * 5}, {}, "must be 2, 3, or 4"),
        ({"Fe": [1, 2], "O": [3, 4, 5]}, {}, "O.*3.*expected 2"),
        ({"Fe": [1, float("inf")]}, {}, "Fe.*finite"),
        ({"Fe": [1, 2]}, dict(orientation="typo"), "orientation"),
        (
            {"Fe": [1, 2]},
            dict(orientation="grid"),
            "orientation='grid' is only supported for n_splits=4, got n_splits=2",
        ),
        (
            {"Fe": [1, 2]},
            dict(colorscale=["Viridis", "Plasma", "Inferno"]),
            "Number of colorscales .* must match",
        ),
        (
            {"Fe": [1, 2]},
            dict(colorbar=[dict(title="1"), dict(title="2"), dict(title="3")]),
            r"Number of colorbars \(3\) must match n_splits=2",
        ),
        (
            {"Fe": [1, 2]},
            dict(scale=-1.0),
            "received for the 'size' property of layout.annotation.font",
        ),
        (
            {"Fe": [1, 2]},
            dict(colorscale="invalid_colorscale"),
            (
                "Invalid value of type 'builtins.str' received for the "
                "'colorscale' property of scatter.marker"
            ),
        ),
    ],
)
def test_ptable_heatmap_splits_error_cases(
    data: dict[str, list[float]], kwargs: dict[str, Any], message: str
) -> None:
    """Reject invalid split arrays, geometry, and colors before rendering."""
    with pytest.raises(ValueError, match=message):
        pmv.ptable_heatmap_splits(data, **kwargs)


@pytest.mark.parametrize("as_dataframe", [False, True])
@pytest.mark.parametrize(
    ("colorscale", "n_colorbars"),
    [
        ("Viridis", 1),
        (["Viridis", "Plasma_r"], 2),
        ([[(0, "rgba(255,0,0,0.1)"), (1, "rgba(0,0,255,0.5)")]] * 2, 2),
        (["#ff0000", "#0000ff"], 1),
        (["#f00", "#00f"], 1),
        (["red", "blue"], 1),
        (["lightblue", "royalblue"], 1),
        ([(0, "rebeccapurple"), (1, "darkslategray")], 1),
        ([(0, "rgb(255,0,0)"), (0.5, "rgb(255,255,0)"), (1, "rgb(0,0,255)")], 1),
        ([(0, "red"), (1, "blue")], 1),
        ([["#ff0000", "#0000ff"], ["#000000", "#ffffff"]], 2),
        (["Viridis", [(0, "rgb(0,0,0)"), (1, "rgb(255,255,255)")]], 2),
    ],
)
def test_ptable_heatmap_splits_colorscales(
    colorscale: Any, n_colorbars: int, as_dataframe: bool
) -> None:
    """Normalize individual and per-split scales without confusing color stops."""
    data = {"H": [1, 3], "He": [2, 2], "Li": [3, 1]}
    fig = pmv.ptable_heatmap_splits(
        pd.DataFrame.from_dict(data, orient="index") if as_dataframe else data,
        colorscale=colorscale,
    )
    colorbars = [trace.marker for trace in fig.data if trace.marker.showscale]
    assert len(colorbars) == (2 if as_dataframe else n_colorbars)
    for colorbar in colorbars:
        assert (colorbar.cmin, colorbar.cmax) == (1, 3)
    if "rgba(" in str(colorscale):
        assert all(
            colorbar.colorscale == ((0, "rgba(255,0,0,0.1)"), (1, "rgba(0,0,255,0.5)"))
            for colorbar in colorbars
        )
    fills = [trace.fillcolor for trace in fig.data if trace.fill == "toself"]
    assert len(fills) == 6
    assert fills[0] != fills[2] != fills[4]
    if n_colorbars == 1:
        assert fills[0] == fills[5]
        assert fills[1] == fills[4]
        assert fills[2] == fills[3]
    if colorscale in (["red", "blue"], [(0, "red"), (1, "blue")]):
        assert fills == [
            "rgb(255, 0, 0)",
            "rgb(0, 0, 255)",
            "rgb(128, 0, 128)",
            "rgb(128, 0, 128)",
            "rgb(0, 0, 255)",
            "rgb(255, 0, 0)",
        ]


@pytest.mark.parametrize("magnitude", [4.0, 1e308, np.finfo(float).max])
def test_ptable_heatmap_splits_extreme_range(magnitude: float) -> None:
    """Normalize finite opposite-sign extrema without overflowing their span."""
    fig = pmv.ptable_heatmap_splits(
        {"H": [-magnitude, magnitude], "He": [-magnitude / 2, magnitude / 2]},
        colorscale=["#000000", "#ffffff"],
    )
    fills = [trace.fillcolor for trace in fig.data if trace.fill == "toself"]
    assert fills == [
        "rgb(0, 0, 0)",
        "rgb(255, 255, 255)",
        "rgb(64, 64, 64)",
        "rgb(191, 191, 191)",
    ]
    colorbar = next(trace.marker for trace in fig.data if trace.marker.showscale)
    assert (colorbar.cmin, colorbar.cmax) == (-magnitude, magnitude)


@pytest.mark.parametrize("per_split", [False, True])
def test_ptable_heatmap_splits_callable_colors(per_split: bool) -> None:
    """Callable tile and colorbar colors receive the actual split index."""

    def color_func(_symbol: str, value: float, split_idx: int) -> str:
        """Encode the split index in a valid RGB color."""
        return f"rgb({split_idx * 100}, 0, {value * 10:.0f})"

    fig = pmv.ptable_heatmap_splits(
        {"H": [1, 2], "He": [3, 4]},
        colorscale=[color_func, color_func] if per_split else color_func,
        colorbar=[{}, {}],
    )
    assert [trace.fillcolor for trace in fig.data if trace.fill == "toself"] == [
        "rgb(0, 0, 10)",
        "rgb(100, 0, 20)",
        "rgb(0, 0, 30)",
        "rgb(100, 0, 40)",
    ]
    colorbars = [trace.marker for trace in fig.data if trace.marker.showscale]
    assert colorbars[0].colorscale == ((0, "rgb(0, 0, 10)"), (1, "rgb(0, 0, 30)"))
    assert colorbars[1].colorscale == ((0, "rgb(100, 0, 20)"), (1, "rgb(100, 0, 40)"))


@pytest.mark.parametrize(
    ("orientations", "positions"),
    [
        (("v", "v"), [(0.8, 0.7), (1.0, 0.7)]),
        (("h", "v"), [(None, None), (None, None)]),
    ],
)
def test_ptable_heatmap_splits_colorbars(
    orientations: tuple[str, str], positions: list[tuple[float | None, float | None]]
) -> None:
    """Per-split colorbar dicts keep custom titles, orientations and positions."""
    data = {str(elem): [1, 2] for elem in list(Element)[:5]}
    colorbar = [
        dict(title=title, orientation=orient)
        | (dict(x=x_pos, y=y_pos) if x_pos is not None else {})
        for title, orient, (x_pos, y_pos) in zip(
            ("First", "Second"), orientations, positions, strict=True
        )
    ]
    fig = pmv.ptable_heatmap_splits(data, colorbar=colorbar)

    cbars = [trace.marker.colorbar for trace in fig.data if trace.marker.showscale]
    assert [cbar.title.text for cbar in cbars] == ["First", "Second"]
    assert tuple(cbar.orientation for cbar in cbars) == orientations
    if positions[0][0] is not None:
        assert [(cbar.x, cbar.y) for cbar in cbars] == positions


def hover_texts(fig: go.Figure) -> list[str]:
    """Hover texts of the invisible per-element hover annotations."""
    return [
        anno.hovertext
        for anno in fig.layout.annotations
        if isinstance(anno.hovertext, str)
    ]


def _low_high(val: float) -> str:
    """Format a value with a low/high suffix."""
    return f"{val:.1f} ({'low' if val < 2 else 'high'})"


TEMPLATE = "{name} ({symbol}) - {split_name}: {value}"


@pytest.mark.parametrize(
    ("values", "kwargs", "expected"),
    [
        ([1.0, 2.0], {}, "<b>Hydrogen</b><br>Split 1: 1<br>Split 2: 2"),
        (
            [1.0, 2.0],
            dict(hover_template=TEMPLATE, hover_fmt=".0f"),
            "Hydrogen (H) - Split 1: 1<br>Hydrogen (H) - Split 2: 2",
        ),
        (
            [1.23456, 2.34567],
            dict(hover_template=TEMPLATE, hover_fmt=".2f"),
            "Hydrogen (H) - Split 1: 1.23<br>Hydrogen (H) - Split 2: 2.35",
        ),
        (
            [1.23456, 2.34567],
            dict(hover_template=TEMPLATE, hover_fmt=_low_high),
            "Hydrogen (H) - Split 1: 1.2 (low)<br>Hydrogen (H) - Split 2: 2.3 (high)",
        ),
        (
            [1.23456, 2.34567],
            dict(hover_data={"H": "Custom data for H"}),
            "<b>Hydrogen</b><br>Split 1: 1.23<br>Split 2: 2.35<br>Custom data for H",
        ),
        (
            [1.0, np.nan],
            dict(
                hover_template="{symbol}: {value} {hover_data}", hover_data={"H": "x"}
            ),
            "H: 1 x",
        ),
    ],
    ids=[
        "default",
        "fmt_str_int",
        "fmt_str_float",
        "fmt_callable",
        "hover_data",
        "template_hover_data_nan",
    ],
)
def test_ptable_heatmap_splits_hover_tooltips(
    values: list[float], kwargs: dict[str, Any], expected: str
) -> None:
    """Hover text honors hover_template, hover_fmt and hover_data, skipping NaN."""
    fig = pmv.ptable_heatmap_splits({"H": values, "He": [1.0, 2.0]}, **kwargs)
    assert hover_texts(fig)[0] == expected


def test_ptable_heatmap_splits_dataframe_input() -> None:
    """Test that ptable_heatmap_splits correctly handles DataFrame input."""
    # Create test DataFrame with meaningful column names
    data = pd.DataFrame(
        {
            "Formation Energy": [1.23, 3.45, 5.67],
            "Band Gap": [2.34, 4.56, 6.78],
        },
        index=pd.Index(["Fe", "O", "H"]),
    )

    # column names are used as split labels in hover text
    fig = pmv.ptable_heatmap_splits(data=data, orientation="diagonal")
    fe_hover_text = next(text for text in hover_texts(fig) if "Iron" in text)
    assert fe_hover_text == "<b>Iron</b><br>Formation Energy: 1.23<br>Band Gap: 2.34"

    # Test custom hover template with DataFrame
    custom_template = "{name} ({symbol}): {split_name} = {value} eV"
    fig = pmv.ptable_heatmap_splits(
        data=data,
        orientation="diagonal",
        hover_template=custom_template,
        hover_fmt=".2f",
    )
    fe_hover_text = next(text for text in hover_texts(fig) if "Iron" in text)
    assert fe_hover_text == (
        "Iron (Fe): Formation Energy = 1.23 eV<br>Iron (Fe): Band Gap = 2.34 eV"
    )

    # Test that colorbar titles include column names when using multiple colorbars
    fig = pmv.ptable_heatmap_splits(
        data=data,
        orientation="diagonal",
        colorbar=[
            dict(title="First"),
            dict(title="Second"),
        ],
    )
    titles = [
        trace.marker.colorbar.title.text for trace in fig.data if trace.marker.showscale
    ]
    assert titles == ["First", "Second"]


def test_ptable_heatmap_splits_special_colors() -> None:
    """Test nan_color and zero_color customization."""
    data = {"Fe": [1.0, np.nan], "O": [0.0, 2.0]}  # Test NaN and zero values

    # Test both custom and default colors
    fig_default = pmv.ptable_heatmap_splits(data)
    fig_custom = pmv.ptable_heatmap_splits(data, nan_color="#f00", zero_color="#0f0")

    for fig, nan_color, zero_color in (
        (fig_default, "#eff", "#aaa"),
        (fig_custom, "#f00", "#0f0"),
    ):
        fill_colors = [
            trace.fillcolor for trace in fig.data if hasattr(trace, "fillcolor")
        ]
        assert nan_color in fill_colors, f"{nan_color=} not found in figure"
        assert zero_color in fill_colors, f"{zero_color=} not found in figure"


@pytest.mark.parametrize(
    ("symbol_kwargs", "expected_color"),
    [
        (None, "white"),  # auto contrast against dark mid-Viridis background
        ({"font": {"color": "red", "size": 16}}, "red"),  # explicit color wins
    ],
)
def test_ptable_heatmap_splits_symbol_font_color(
    symbol_kwargs: dict[str, Any] | None, expected_color: str
) -> None:
    """Symbol font color auto-contrasts with the tile unless set in symbol_kwargs."""
    fig = pmv.ptable_heatmap_splits({"Fe": [0.5, 0.5]}, symbol_kwargs=symbol_kwargs)
    symbol_colors = [
        anno.font.color for anno in fig.layout.annotations if anno.text == "Fe"
    ]
    assert symbol_colors == [expected_color]


@pytest.mark.parametrize(
    "build_fig",
    [
        lambda syms: pmv.ptable_heatmap_splits(
            {sym: [1, 2] for sym in syms}, annotations=dict.fromkeys(syms, "x")
        ),
        lambda syms: pmv.ptable_hists(
            {sym: [1.0, 2.0, 3.0] for sym in syms}, color_elem_strategy="background"
        ),
        lambda syms: pmv.ptable_scatter(
            {sym: ([1, 2, 3], [4, 5, 6]) for sym in syms},
            color_elem_strategy="background",
        ),
    ],
    ids=["heatmap_splits", "hists", "scatter"],
)
def test_ptable_builders_batch_layout_items(
    build_fig: Callable[[list[str]], go.Figure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-tile annotations/shapes are batch-assigned, never added one at a time.

    Adding them via fig.add_annotation()/fig.add_shape() inside the element loop is
    O(n^2) (plotly re-validates the whole layout array on each call, a full periodic
    table took ~10 min in CI), so builders must collect dicts and assign
    fig.layout.{annotations,shapes} in a single update.
    """
    counts = {"add_annotation": 0, "add_shape": 0}

    def make_spy(method: str) -> Callable[..., object]:
        orig = getattr(go.Figure, method)

        def spy(self: go.Figure, *args: object, **kwargs: object) -> object:
            counts[method] += 1
            return orig(self, *args, **kwargs)

        return spy

    for method in counts:
        monkeypatch.setattr(go.Figure, method, make_spy(method))

    symbols = [elem.symbol for elem in Element][:40]
    fig = build_fig(symbols)

    # builders must use batched fig.layout.{annotations,shapes}, never per-tile calls
    assert counts == {"add_annotation": 0, "add_shape": 0}
    # batching kept content: at least one layout annotation per rendered tile
    assert len(fig.layout.annotations) >= len(symbols)
