from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import pymatviz as pmv
from pymatviz.typing import VALID_COLOR_ELEM_STRATEGIES, ColorElemTypeStrategy


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal


@pytest.mark.parametrize(
    ("elem_data", "bins", "x_range", "log", "colorscale"),
    [
        # Basic case
        ({"Fe": [1, 2], "O": [3, 4]}, 10, None, False, "RdBu"),
        # Single element, log scale
        (pd.DataFrame({"Fe": [0.1, 0.2]}), 5, (0, 1), True, "Viridis"),
        ({"H": [0.1, 0.2], "He": [0.3, 0.4]}, 5, None, True, "Turbo"),
    ],
)
def test_ptable_hists_plotly_basic(
    elem_data: pd.DataFrame | dict[str, list[float]],
    bins: int,
    x_range: tuple[float | None, float | None] | None,
    log: bool,
    colorscale: str,
) -> None:
    fig = pmv.ptable_hists_plotly(
        elem_data, bins=bins, x_range=x_range, log=log, colorscale=colorscale
    )
    assert isinstance(fig, go.Figure)
    # Count only histogram traces (exclude colorbar trace)
    n_hist_traces = sum(isinstance(trace, go.Histogram) for trace in fig.data)
    if isinstance(elem_data, dict):
        n_elements = len(elem_data)
    else:
        n_elements = len(elem_data.columns)
    assert n_hist_traces == n_elements


@pytest.mark.parametrize(
    ("font_size", "scale", "symbol_kwargs", "annotations"),
    [
        (12, 1.0, dict(x=0, y=1), dict(x=1, y=1)),  # Most common case
        (None, 0.5, dict(x=0.5, y=0.5), dict(x=0.5, y=0.5)),  # Test None font_size
    ],
)
def test_ptable_hists_plotly_layout(
    font_size: int | None,
    scale: float,
    symbol_kwargs: dict[str, Any],
    annotations: dict[str, Any],
) -> None:
    fig = pmv.ptable_hists_plotly(
        {"Fe": [1, 2, 3], "O": [2, 3, 4]},
        font_size=font_size,
        scale=scale,
        symbol_kwargs=symbol_kwargs,
        annotations=annotations,
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
    fig = pmv.ptable_hists_plotly(data, color_elem_strategy=color_elem_strategy)
    assert isinstance(fig, go.Figure)


def test_ptable_hists_plotly_annotations() -> None:
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}
    anno_kwargs: dict[str, str | int] = {"font_size": 14, "font_color": "red"}

    annotations = {
        # Test multiple annotations per element
        "Fe": [
            {"text": "Iron", **anno_kwargs},
            {"text": "<br>Fe", "font_size": 12, "font_color": "gray"},
        ],
        "O": {"text": "Oxygen", **anno_kwargs},
    }
    fig = pmv.ptable_hists_plotly(data, annotations=annotations)  # type: ignore[arg-type]
    assert isinstance(fig, go.Figure)

    # Check multiple annotations are present
    anno_texts = [anno.text for anno in fig.layout.annotations]
    assert len(anno_texts) == 5
    assert "Iron" in anno_texts
    assert "<br>Fe" in anno_texts
    assert "Oxygen" in anno_texts
    assert "O" in anno_texts

    # Test with callable annotations returning multiple annotations
    def annotation_func(value: Sequence[float]) -> list[dict[str, Any]]:
        mean_val = np.mean(value)
        return [
            {"text": f"Mean: {mean_val:.1f}"},
            {"text": f"<br>Range: {max(value) - min(value):.1f}"},
        ]

    fig = pmv.ptable_hists_plotly(data, annotations=annotation_func)
    # check multiple annotations are present
    anno_texts = [anno.text for anno in fig.layout.annotations]
    assert len(anno_texts) == 6
    assert "Mean: 2.0" in anno_texts
    assert "Mean: 3.0" in anno_texts
    assert "<br>Range: 2.0" in anno_texts


def test_ptable_hists_plotly_error_cases() -> None:
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}

    # Test invalid color_elem_strategy
    with pytest.raises(
        ValueError, match="color_elem_strategy='invalid' must be one of"
    ):
        pmv.ptable_hists_plotly(data, color_elem_strategy="invalid")  # type: ignore[arg-type]

    # Test invalid scale
    with pytest.raises(ValueError, match="Invalid value of type ") as exc:
        pmv.ptable_hists_plotly(data, scale=-1.0)
    assert "An int or float in the interval [1, inf]" in str(exc.value)

    # Test invalid bins
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        pmv.ptable_hists_plotly(data, bins=0)


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
    fig = pmv.ptable_hists_plotly(
        {"Fe": [1, 2, 3], "O": [2, 3, 4]},
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
    fig = pmv.ptable_hists_plotly(data, subplot_kwargs=subplot_kwargs)
    assert isinstance(fig, go.Figure)
    # Verify subplot titles are set
    assert [anno.text for anno in fig.layout.annotations][:2] == sub_titles


def test_ptable_hists_plotly_hover_tooltips() -> None:
    """Test that hover tooltips show element info and histogram values."""
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}
    element_symbol_map = {"Fe": "Iron"}

    fig = pmv.ptable_hists_plotly(data, element_symbol_map=element_symbol_map)

    # Get hover templates for each histogram trace
    hover_templates = [trace.hovertemplate for trace in fig.data]

    expected_fe_template = (
        "<b>Iron</b> (Fe)<br>Range: %{x}<br>Count: %{y}<extra></extra>"
    )
    expected_o_template = "<b>O</b><br>Range: %{x}<br>Count: %{y}<extra></extra>"

    assert expected_fe_template in hover_templates
    assert expected_o_template in hover_templates


@pytest.mark.parametrize(
    "colorbar",
    [
        # Test default settings
        None,
        # Test no colorbar
        False,
        # Test horizontal colorbar with custom length
        dict(orientation="h", len=0.4),
        # Test vertical colorbar with custom length
        dict(orientation="v", len=0.8),
        # Test title formatting for horizontal orientation
        dict(title="Test Title", orientation="h"),
        # Test comprehensive custom settings
        dict(orientation="v", len=0.6, thickness=20, title="Custom", x=1.1, y=0.5),
    ],
)
def test_ptable_hists_plotly_colorbar(
    colorbar: dict[str, Any] | Literal[False] | None,
) -> None:
    """Test colorbar customization in pmv.ptable_hists_plotly including range and
    visibility.
    """
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}
    x_range = (-1, 5)  # Test custom range
    fig = pmv.ptable_hists_plotly(data, colorbar=colorbar, x_range=x_range)

    if colorbar is False:
        assert all(not hasattr(trace, "colorbar") for trace in fig.data)
        return

    # Find the scatter trace with the colorbar (should be last trace)
    cbar_trace = fig.data[-1]
    assert isinstance(cbar_trace, go.Scatter)
    assert cbar_trace.hoverinfo == "none"
    assert cbar_trace.showlegend is False

    # Check colorbar properties
    marker = cbar_trace.marker
    assert marker.showscale is True

    # Check colorbar range matches x_range
    assert marker.cmin == x_range[0]
    assert marker.cmax == x_range[1]

    # Check that axes for this trace are hidden
    x_axis_num = len(fig.data)  # Last trace's axis number
    assert not fig.layout[f"xaxis{x_axis_num}"].visible
    assert not fig.layout[f"yaxis{x_axis_num}"].visible

    # Check colorbar settings
    cbar_defaults = dict(len=0.87, thickness=15)
    for key, expect in (colorbar or cbar_defaults).items():
        if key == "title":
            assert marker.colorbar.title.text == expect
        else:
            actual = getattr(marker.colorbar, key)
            assert actual == expect, f"{key=}: {actual=} != {expect=}"


def test_ptable_hists_plotly_x_axis_kwargs() -> None:
    """Test that x_axis_kwargs properly modifies histogram x-axes."""
    data = {"Fe": [1, 2, 3], "O": [2, 3, 4]}

    # Test various x-axis customizations
    x_axis_kwargs = {
        "tickangle": 45,
        "tickformat": ".3f",
        "nticks": 5,
        "showticklabels": False,
        "tickfont": dict(size=14, color="red"),
        "linecolor": "blue",
        "linewidth": 2,
    }

    fig = pmv.ptable_hists_plotly(data, x_axis_kwargs=x_axis_kwargs)

    xaxis = fig.layout.xaxis

    for key, expected in x_axis_kwargs.items():
        actual = getattr(xaxis, key)
        if isinstance(expected, dict):
            for sub_key, sub_expected in expected.items():
                assert getattr(actual, sub_key) == sub_expected, f"{key}.{sub_key}"
        else:
            assert actual == expected, f"{key}: {actual} != {expected}"
