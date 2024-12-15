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

    assert fig.layout.width == 850 * scale
    assert fig.layout.height == 500 * scale
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


def test_ptable_heatmap_splits_plotly_callable_colorscale() -> None:
    """Test that ptable_heatmap_splits_plotly accepts a callable colorscale."""
    data = {"Fe": [1, 2], "O": [3, 4]}

    def custom_colorscale(_element_symbol: str, _value: float, split_idx: int) -> str:
        # Return a color based on element symbol, value and split index
        if split_idx == 0:
            return "rgb(255,0,0)"  # Red for first split
        if split_idx == 1:
            return "rgb(0,0,255)"  # Blue for second split
        return "rgb(255,255,255)"  # White else (not used in this test)

    fig = pmv.ptable_heatmap_splits_plotly(data, colorscale=custom_colorscale)
    assert isinstance(fig, go.Figure)

    # Check that colorscale was applied correctly to heatmap traces
    for trace in fig.data:
        # Each element tile should have a color array with custom colors
        assert trace.fillcolor in {"rgb(255,0,0)", "rgb(0,0,255)"}
