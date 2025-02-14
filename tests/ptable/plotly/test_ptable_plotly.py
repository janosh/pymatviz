from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.exceptions import PlotlyError

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
        PlotlyError, match="Colorscale invalid_colorscale is not a built-in scale"
    ):
        pmv.ptable_heatmap_splits_plotly(data, colorscale="invalid_colorscale")


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
    assert (
        actual_h_colorbar.title.text == "Horizontal Title<br>"
    )  # Horizontal title has break after
    assert actual_h_colorbar.orientation == "h"
    assert actual_h_colorbar.y == 0.8

    # Test disabling colorbar
    fig = pmv.ptable_heatmap_plotly(data, show_scale=False)
    assert not any(trace.showscale for trace in fig.data)
