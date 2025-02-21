from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pymatgen.core import Element

import pymatviz as pmv


if TYPE_CHECKING:
    from typing import Any, Literal


def test_ptable_heatmap_splits_plotly_basic() -> None:
    """Test basic functionality of ptable_heatmap_splits_plotly."""
    data: dict[str, list[float]] = {
        "Fe": [1, 2],
        "O": [3, 4],
        "H": [0.5, 1.5],
        "He": [1.5, 2.5],
    }

    # Test each orientation
    for orientation in ["diagonal", "horizontal", "vertical"]:
        fig = pmv.ptable_heatmap_splits_plotly(data, orientation=orientation)  # type: ignore[arg-type]
        assert isinstance(fig, go.Figure)
        # Each split should have its own subplot
        assert len(fig.data) == sum(len(v) for v in data.values()) + 1

    # Test grid orientation separately with 4 splits
    data_4_split = {"Fe": [1, 2, 3, 4]}
    fig = pmv.ptable_heatmap_splits_plotly(data_4_split, orientation="grid")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == sum(len(v) for v in data_4_split.values()) + 1


def test_ptable_heatmap_splits_plotly_f_block() -> None:
    """Test that f-block elements are only hidden in 'auto' mode when there's no data
    for any f-block element."""
    # Data with no f-block elements
    data_no_f = {"Fe": [1, 2], "O": [3, 4], "H": [0.5, 1.5]}

    rows_per_mode: dict[bool | Literal["auto"], int] = {True: 7, "auto": 7, False: 10}

    for hide_f_block, expected_n_rows in rows_per_mode.items():
        fig = pmv.ptable_heatmap_splits_plotly(data_no_f, hide_f_block=hide_f_block)
        n_rows = len(fig._grid_ref)
        assert n_rows == expected_n_rows, f"{n_rows=}, {hide_f_block=}"

    # Data including f-block elements
    data_with_f = {"Fe": [1, 2], "La": [3, 4], "U": [5, 6]}

    rows_per_mode["auto"] = 10
    for hide_f_block, expected_n_rows in rows_per_mode.items():
        fig = pmv.ptable_heatmap_splits_plotly(data_with_f, hide_f_block=hide_f_block)
        n_rows = len(fig._grid_ref)
        assert n_rows == expected_n_rows, f"{n_rows=}, {hide_f_block=}"

    fig_with_f = pmv.ptable_heatmap_splits_plotly(data_with_f, hide_f_block="auto")
    anno_texts = [anno.text for anno in fig_with_f.layout.annotations]
    expected = [
        "Fe",
        "<b>Iron</b><br>Split 1: 1<br>Split 2: 2",
        "La",
        "<b>Lanthanum</b><br>Split 1: 3<br>Split 2: 4",
        "U",
        "<b>Uranium</b><br>Split 1: 5<br>Split 2: 6",
    ]
    assert anno_texts == expected, f"{anno_texts=}"

    fig_with_f = pmv.ptable_heatmap_splits_plotly(data_with_f, hide_f_block=True)
    anno_texts = [anno.text for anno in fig_with_f.layout.annotations]
    expected = ["Fe", "<b>Iron</b><br>Split 1: 1<br>Split 2: 2"]
    assert anno_texts == expected, f"{anno_texts=}"


@pytest.mark.parametrize(
    ("orientation", "colorscale", "font_size", "scale"),
    [
        ("diagonal", "Viridis", 12, 1.0),
        ("horizontal", "RdBu", None, 0.8),
        ("vertical", "Spectral", 14, 1.2),
        ("diagonal", "Plasma", 10, 0.9),
    ],
)
def test_ptable_heatmap_splits_plotly_display_options(
    orientation: Literal["diagonal", "horizontal", "vertical", "grid"],
    colorscale: str,
    font_size: int | None,
    scale: float,
) -> None:
    """Test various display options for ptable_heatmap_splits_plotly."""
    subplot_kwargs = {
        "horizontal_spacing": 0.05,
        "vertical_spacing": 0.05,
        "subplot_titles": ["Split 1", "Split 2", "Split 3", "Split 4"],
    }
    # Test custom element symbols
    element_symbol_map = {"Fe": "Iron", "O": "Oxygen"}
    symbol_kwargs = {"font": {"size": 14, "color": "red"}}

    fig = pmv.ptable_heatmap_splits_plotly(
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
    [None, False, dict(orientation="v", len=0.8), dict(orientation="h", len=0.3)],
)
def test_ptable_heatmap_splits_plotly_colorbar(
    colorbar: dict[str, Any] | Literal[False] | None,
) -> None:
    """Test colorbar customization in ptable_heatmap_splits_plotly."""
    data = {"Fe": [1, 2], "O": [3, 4], "H": [0.5, 1.5], "He": [1.5, 2.5]}

    fig = pmv.ptable_heatmap_splits_plotly(data, colorbar=colorbar)

    hidden_scatter_trace = [trace for trace in fig.data if trace.x[0] is None]
    assert (len(hidden_scatter_trace) == 0) == (colorbar is False)


def test_ptable_heatmap_splits_plotly_annotations() -> None:
    """Test custom annotations in ptable_heatmap_splits_plotly."""
    data = {"Fe": [1, 2], "O": [3, 4], "H": [0.5, 1.5], "He": [1.5, 2.5]}

    # Test with dict annotations
    annotations = {
        "Fe": {"text": "Iron", "font": {"size": 14, "color": "red"}},
        "O": {"text": "Oxygen", "font": {"size": 14, "color": "blue"}},
    }

    fig = pmv.ptable_heatmap_splits_plotly(data, annotations=annotations)  # type: ignore[arg-type]
    assert isinstance(fig, go.Figure)

    # Test with callable annotations
    def annotation_func(value: list[float] | np.ndarray) -> dict[str, Any]:
        return {"text": f"Value: {np.mean(value):.1f}"}

    fig = pmv.ptable_heatmap_splits_plotly(data, annotations=annotation_func)
    # check annotations are present
    anno_texts = [anno.text for anno in fig.layout.annotations]
    assert "Value: 1.5" in anno_texts
    assert "Value: 3.5" in anno_texts


def test_ptable_heatmap_splits_plotly_error_cases() -> None:
    """Test error cases for ptable_heatmap_splits_plotly."""
    data = {"Fe": [1, 2], "O": [3, 4]}

    # Test invalid n_splits
    with pytest.raises(ValueError, match="n_splits=1 must be 2, 3, or 4"):
        pmv.ptable_heatmap_splits_plotly({"Fe": [1]})

    # Test invalid orientation
    with pytest.raises(
        ValueError,
        match="orientation='grid' is only supported for n_splits=4, got n_splits=2",
    ):
        pmv.ptable_heatmap_splits_plotly(data, orientation="grid")

    # Test invalid scale
    with pytest.raises(
        ValueError, match="received for the 'size' property of layout.annotation.font"
    ):
        pmv.ptable_heatmap_splits_plotly(data, scale=-1.0)

    # Test invalid colorscale
    with pytest.raises(
        ValueError,
        match="Invalid value of type 'builtins.str' received for the "
        "'colorscale' property of scatter.marker",
    ):
        pmv.ptable_heatmap_splits_plotly(data, colorscale="invalid_colorscale")


def test_ptable_heatmap_splits_plotly_colorscales() -> None:
    """Test different colorscale configurations."""
    # Create test data with 2 values per element
    data = {str(elem): [1, 2] for elem in list(Element)[:5]}

    # Test single colorscale
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        colorscale="Viridis",
        colorbar=dict(title="Test"),  # Add colorbar to trigger split_names check
    )
    assert isinstance(fig, go.Figure)

    # Test multiple colorscales as list of strings
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        colorscale=["Viridis", "Plasma"],  # One colorscale per split
        colorbar=[
            dict(title="First"),
            dict(title="Second"),
        ],  # Add colorbars to trigger split_names check
    )
    assert isinstance(fig, go.Figure)

    # Test custom colorscale as list of RGB tuples
    custom_colorscale = [
        (0.0, "rgb(255,0,0)"),
        (0.5, "rgb(255,255,0)"),
        (1.0, "rgb(0,0,255)"),
    ]
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        colorscale=[custom_colorscale, custom_colorscale],  # One per split
    )
    assert isinstance(fig, go.Figure)

    # Test invalid number of splits
    with pytest.raises(ValueError, match="must be 2, 3, or 4"):
        pmv.ptable_heatmap_splits_plotly(
            data={str(elem): [1] * 5 for elem in list(Element)[:5]},  # 5 splits
            orientation="diagonal",
        )

    # Test mismatched colorscales and data splits
    with pytest.raises(ValueError, match="Number of colorscales .* must match"):
        pmv.ptable_heatmap_splits_plotly(
            data=data,  # 2 splits
            orientation="diagonal",
            colorscale=["Viridis", "Plasma", "Inferno"],  # 3 colorscales
        )


def test_ptable_heatmap_splits_plotly_colorbars() -> None:
    """Test different colorbar configurations."""
    # Create test data with 2 values per element
    data = {str(elem): [1, 2] for elem in list(Element)[:5]}

    # Test single colorbar dict
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        colorbar=dict(title="Test"),
    )
    assert isinstance(fig, go.Figure)

    # Test multiple colorbars with custom positions
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        colorbar=[
            dict(title="First", orientation="v", x=0.8, y=0.7),  # Custom position
            dict(title="Second", orientation="v", x=1.0, y=0.7),  # Custom position
        ],
    )
    assert isinstance(fig, go.Figure)
    # Check that last two traces used the custom positions (dummy traces for colorbars)
    colorbar_traces = [
        trace
        for trace in fig.data
        if hasattr(trace, "marker") and hasattr(trace.marker, "colorbar")
    ]
    assert len(colorbar_traces) >= 2
    cbar1 = colorbar_traces[-2].marker.colorbar
    cbar2 = colorbar_traces[-1].marker.colorbar
    assert (cbar1.x, cbar1.y) == (0.8, 0.7)
    assert (cbar2.x, cbar2.y) == (1.0, 0.7)

    # Test mixed orientations
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        colorbar=[
            dict(title="First", orientation="h"),
            dict(title="Second", orientation="v"),
        ],
    )
    assert isinstance(fig, go.Figure)
    colorbar_traces = [
        trace
        for trace in fig.data
        if hasattr(trace, "marker") and hasattr(trace.marker, "colorbar")
    ]
    assert len(colorbar_traces) >= 2
    assert colorbar_traces[-2].marker.colorbar.orientation == "h"
    assert colorbar_traces[-1].marker.colorbar.orientation == "v"

    # Test mismatched colorbars and splits
    with pytest.raises(ValueError, match="Number of colorbars .* must match"):
        pmv.ptable_heatmap_splits_plotly(
            data=data,  # 2 splits
            orientation="diagonal",
            colorbar=[dict(title="1"), dict(title="2"), dict(title="3")],  # 3 colorbars
        )


def test_ptable_heatmap_splits_plotly_split_names() -> None:
    """Test that split names are correctly assigned based on orientation."""
    # Create test data with 2 values per element
    data = {str(elem): [1, 2] for elem in list(Element)[:5]}

    # Test diagonal orientation split names
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        colorscale=["Viridis", "Plasma"],  # One colorscale per split
        colorbar=[dict(title="First"), dict(title="Second")],
    )
    # Check that colorbar titles contain the correct split names
    colorbar_traces = [
        trace
        for trace in fig.data
        if hasattr(trace, "marker") and hasattr(trace.marker, "colorbar")
    ]
    assert len(colorbar_traces) >= 2
    titles = [trace.marker.colorbar.title.text for trace in colorbar_traces[-2:]]
    assert any("bottom-left" in title for title in titles), f"{titles=}"
    assert any("top-right" in title for title in titles), f"{titles=}"

    # Test horizontal orientation split names
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="horizontal",
        colorscale=["Viridis", "Plasma"],
        colorbar=[dict(title="First"), dict(title="Second")],
    )
    colorbar_traces = [
        trace
        for trace in fig.data
        if hasattr(trace, "marker") and hasattr(trace.marker, "colorbar")
    ]
    assert len(colorbar_traces) >= 2
    titles = [trace.marker.colorbar.title.text for trace in colorbar_traces[-2:]]
    assert any("bottom" in title for title in titles)
    assert any("top" in title for title in titles)

    # Test vertical orientation split names
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="vertical",
        colorscale=["Viridis", "Plasma"],
        colorbar=[dict(title="First"), dict(title="Second")],
    )
    colorbar_traces = [
        trace
        for trace in fig.data
        if hasattr(trace, "marker") and hasattr(trace.marker, "colorbar")
    ]
    assert len(colorbar_traces) >= 2
    titles = [trace.marker.colorbar.title.text for trace in colorbar_traces[-2:]]
    assert any("left" in title for title in titles)
    assert any("right" in title for title in titles)


def test_ptable_heatmap_splits_plotly_hover_tooltips() -> None:
    """Test hover tooltip customization."""
    # Create test data with 2 values per element
    data = {str(elem): [1.0, 2.0] for elem in list(Element)[:5]}

    # Test default hover tooltip
    fig = pmv.ptable_heatmap_splits_plotly(data=data, orientation="diagonal")
    hover_texts = [
        anno.hovertext
        for anno in fig.layout.annotations
        if hasattr(anno, "hovertext") and isinstance(anno.hovertext, str)
    ]
    assert len(hover_texts) > 0, "No hover texts found"
    split_texts = [text for text in hover_texts if "Split 1" in text]
    assert len(split_texts) > 0, "No split texts found"
    for text in split_texts:
        assert "Split 1" in text, "Default split name not in hover text"
        assert "Split 2" in text, "Default split name not in hover text"

    # Test custom hover template with string formatter
    custom_template = "{name} ({symbol}) - {split_name}: {value}"
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        hover_template=custom_template,
        hover_fmt=".0f",  # Test integer format
    )
    hover_texts = [
        anno.hovertext
        for anno in fig.layout.annotations
        if hasattr(anno, "hovertext") and isinstance(anno.hovertext, str)
    ]
    assert len(hover_texts) > 0, "No hover texts found"
    split_texts = [text for text in hover_texts if "Split 1" in text]
    assert len(split_texts) > 0, "No split texts found"
    for text in split_texts:
        assert " - Split 1: 1" in text, f"Value format wrong in text: {text}"
        assert " - Split 2: 2" in text, f"Value format wrong in text: {text}"

    # Test with non-integer values and string formatter
    data = {str(elem): [1.23456, 2.34567] for elem in list(Element)[:5]}
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        hover_template=custom_template,
        hover_fmt=".2f",  # Test float format with 2 decimals
    )
    hover_texts = [
        anno.hovertext
        for anno in fig.layout.annotations
        if hasattr(anno, "hovertext") and isinstance(anno.hovertext, str)
    ]
    assert len(hover_texts) > 0, "No hover texts found"
    split_texts = [text for text in hover_texts if "Split 1" in text]
    assert len(split_texts) > 0, "No split texts found"
    for text in split_texts:
        assert " - Split 1: 1.23" in text, (
            f"String format not applied correctly in text: {text}"
        )
        assert " - Split 2: 2.35" in text, (
            f"String format not applied correctly in text: {text}"
        )

    # Test with callable formatter
    def custom_formatter(val: float) -> str:
        if val < 2:
            return f"{val:.1f} (low)"
        return f"{val:.1f} (high)"

    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        hover_template=custom_template,
        hover_fmt=custom_formatter,
    )
    hover_texts = [
        anno.hovertext
        for anno in fig.layout.annotations
        if hasattr(anno, "hovertext") and isinstance(anno.hovertext, str)
    ]
    assert len(hover_texts) > 0, "No hover texts found"
    split_texts = [text for text in hover_texts if "Split 1" in text]
    assert len(split_texts) > 0, "No split texts found"
    for text in split_texts:
        assert " - Split 1: 1.2 (low)" in text, (
            f"Callable format not applied correctly in text: {text}"
        )
        assert " - Split 2: 2.3 (high)" in text, (
            f"Callable format not applied correctly in text: {text}"
        )

    # Test hover tooltip with DataFrame input
    df_data = pd.DataFrame(data).T
    df_data.columns = ["First Value", "Second Value"]
    fig = pmv.ptable_heatmap_splits_plotly(
        data=df_data,
        orientation="diagonal",
        hover_fmt=".2f",
    )
    hover_texts = [
        anno.hovertext
        for anno in fig.layout.annotations
        if hasattr(anno, "hovertext") and isinstance(anno.hovertext, str)
    ]
    assert len(hover_texts) > 0, "No hover texts found"
    split_texts = [text for text in hover_texts if "First Value" in text]
    assert len(split_texts) > 0, "No split texts found"
    for text in split_texts:
        assert "First Value" in text, "DataFrame column name not in hover text"
        assert "Second Value" in text, "DataFrame column name not in hover text"

    # Test hover tooltip with custom hover data
    hover_data = {str(elem): f"Custom data for {elem}" for elem in list(Element)[:5]}
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        hover_data=hover_data,
    )
    hover_texts = [
        anno.hovertext
        for anno in fig.layout.annotations
        if hasattr(anno, "hovertext") and isinstance(anno.hovertext, str)
    ]
    assert len(hover_texts) > 0, "No hover texts found"
    custom_texts = [text for text in hover_texts if "Custom data" in text]
    assert len(custom_texts) > 0, "No custom hover data found"
    for text in custom_texts:
        assert "Custom data for" in text, "Custom hover data not in hover text"


def test_ptable_heatmap_splits_plotly_dataframe_input() -> None:
    """Test that ptable_heatmap_splits_plotly correctly handles DataFrame input."""
    # Create test DataFrame with meaningful column names
    data = pd.DataFrame(
        {
            "Formation Energy": [1.23, 3.45, 5.67],
            "Band Gap": [2.34, 4.56, 6.78],
        },
        index=["Fe", "O", "H"],
    )

    # Test default hover tooltip with DataFrame
    fig = pmv.ptable_heatmap_splits_plotly(data=data, orientation="diagonal")
    hover_texts = [
        anno.hovertext
        for anno in fig.layout.annotations
        if hasattr(anno, "hovertext") and isinstance(anno.hovertext, str)
    ]
    assert len(hover_texts) > 0, "No hover texts found"

    # Check that column names are used as split labels in hover text
    fe_texts = [text for text in hover_texts if "Iron" in text]
    assert len(fe_texts) > 0, "No hover text found for Iron"
    fe_hover_text = fe_texts[0]
    assert "Formation Energy: 1.23" in fe_hover_text, (
        "DataFrame column name 'Formation Energy' not used in hover text"
    )
    assert "Band Gap: 2.34" in fe_hover_text, (
        "DataFrame column name 'Band Gap' not used in hover text"
    )

    # Test custom hover template with DataFrame
    custom_template = "{name} ({symbol}): {split_name} = {value} eV"
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        hover_template=custom_template,
        hover_fmt=".2f",
    )
    hover_texts = [
        anno.hovertext
        for anno in fig.layout.annotations
        if hasattr(anno, "hovertext") and isinstance(anno.hovertext, str)
    ]
    assert len(hover_texts) > 0, "No hover texts found with custom template"

    # Check that column names are used with custom template
    fe_texts = [text for text in hover_texts if "Iron" in text]
    assert len(fe_texts) > 0, "No hover text found for Iron"
    fe_hover_text = fe_texts[0]
    assert "Formation Energy = 1.23 eV" in fe_hover_text, (
        "Column name not used correctly with custom template"
    )
    assert "Band Gap = 2.34 eV" in fe_hover_text, (
        "Column name not used correctly with custom template"
    )

    # Test that colorbar titles include column names when using multiple colorbars
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data,
        orientation="diagonal",
        colorbar=[
            dict(title="First"),
            dict(title="Second"),
        ],
    )
    colorbar_traces = [
        trace
        for trace in fig.data
        if hasattr(trace, "marker")
        and hasattr(trace.marker, "colorbar")
        and all(val is None for val in trace.x)  # Only dummy traces for colorbars
        and trace.marker.colorbar.title is not None
    ]
    assert len(colorbar_traces) == 2, "Expected 2 colorbars"
    # TODO make sure dataframe column names make their way into colorbar titles
    # titles = [trace.marker.colorbar.title.text for trace in colorbar_traces]
    # assert any("Formation Energy" in title for title in titles), f"{titles=}"
    # assert any("Band Gap" in title for title in titles), f"{titles=}"


def test_ptable_heatmap_splits_plotly_special_colors() -> None:
    """Test nan_color and zero_color customization."""
    data = {"Fe": [1.0, np.nan], "O": [0.0, 2.0]}  # Test NaN and zero values

    # Test both custom and default colors
    fig_default = pmv.ptable_heatmap_splits_plotly(data)
    fig_custom = pmv.ptable_heatmap_splits_plotly(
        data, nan_color="#f00", zero_color="#0f0"
    )

    for fig, nan_color, zero_color in (
        (fig_default, "#eff", "#aaa"),
        (fig_custom, "#f00", "#0f0"),
    ):
        fill_colors = [
            trace.fillcolor for trace in fig.data if hasattr(trace, "fillcolor")
        ]
        assert nan_color in fill_colors, f"{nan_color=} not found in figure"
        assert zero_color in fill_colors, f"{zero_color=} not found in figure"
