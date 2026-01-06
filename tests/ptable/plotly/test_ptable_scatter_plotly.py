"""Test ptable_scatter_plotly function."""

import re
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import plotly.graph_objects as go
import pytest
from pymatgen.core import Element

import pymatviz as pmv


SampleData: TypeAlias = dict[str, tuple[Sequence[float], Sequence[float]]]
ColorData: TypeAlias = dict[
    str,
    tuple[Sequence[float], Sequence[float]]
    | tuple[Sequence[float], Sequence[float], Sequence[float | str]],
]


@pytest.fixture
def sample_data() -> SampleData:
    """Create sample data for testing."""
    return {"Fe": ([1, 2, 3], [4, 5, 6]), "O": ([0, 1, 2], [3, 4, 5])}


@pytest.fixture
def color_data() -> ColorData:
    """Create sample data with color information."""
    return {
        "Fe": ([1, 2], [4, 5], ["red", "blue"]),  # discrete colors
        "Cu": ([1, 2], [7, 8], [0.1, 0.9]),  # numeric colors
        "O": ([1, 2], [3, 4]),  # no colors
    }


def test_basic_scatter_plot(sample_data: SampleData) -> None:
    """Test basic scatter plot creation."""
    fig = pmv.ptable_scatter_plotly(sample_data)

    # Check basic figure properties
    assert isinstance(fig, go.Figure)
    assert fig.layout.width == 850  # default scale=1.0
    assert fig.layout.height == 500
    assert fig.layout.showlegend is False
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"

    # Check that we have scatter traces for each element
    scatter_traces = [trace for trace in fig.data if isinstance(trace, go.Scatter)]
    assert len(scatter_traces) == len(sample_data)

    # Verify data points match input
    for trace in scatter_traces:
        elem_name = trace.hovertemplate.split("<br>")[0]  # extract element name
        elem_symbol = {"Iron": "Fe", "Oxygen": "O"}[elem_name]
        x_data, y_data = sample_data[elem_symbol]
        assert list(trace.x) == x_data
        assert list(trace.y) == y_data
        # Check trace properties
        assert trace.mode == "markers"  # default mode
        assert trace.showlegend is False
        assert trace.marker.size == 6
        assert trace.marker.line.width == 0

    # Check axis properties
    for axis in (fig.layout.xaxis, fig.layout.yaxis):
        assert axis.showgrid is False
        assert axis.showline is True
        assert axis.linewidth == 1
        assert axis.mirror is False
        assert axis.ticks == "inside"
        assert axis.tickwidth == 1
        assert axis.nticks == 2
        assert axis.tickmode == "auto"

    # Check that zeroline is False for both axes
    assert fig.layout.xaxis.zeroline is False
    assert fig.layout.yaxis.zeroline is False

    # Check element annotations
    symbol_annotations = [
        ann for ann in fig.layout.annotations if ann.text in {e.name for e in Element}
    ]
    assert len(symbol_annotations) == len(sample_data)

    for ann in symbol_annotations:
        assert ann.showarrow is False
        assert ann.xanchor == "right"
        assert ann.yanchor == "top"
        assert ann.x == 1
        assert ann.y == 1
        assert ann.font.size == 11  # default font size
        # Check that text is either Fe or O
        assert ann.text in ("Fe", "O")


@pytest.mark.parametrize(
    ("mode", "expected_color", "expected_width"),
    [
        ("markers", "blue", None),
        ("lines", "red", 1.5),
        ("lines+markers", "green", 3),
    ],
)
def test_plot_modes(
    sample_data: SampleData,
    mode: Literal["markers", "lines", "lines+markers"],
    expected_color: str,
    expected_width: float | None,
) -> None:
    """Test different plot modes with their styling."""
    line_kwargs: dict[str, Any] = {"color": expected_color}
    if expected_width is not None:
        line_kwargs["width"] = expected_width

    fig = pmv.ptable_scatter_plotly(sample_data, mode=mode, line_kwargs=line_kwargs)

    scatter_traces = [trace for trace in fig.data if isinstance(trace, go.Scatter)]
    for trace in scatter_traces:
        assert trace.mode == mode
        assert trace.line.color == expected_color
        if expected_width is not None:
            assert trace.line.width == expected_width


def test_axis_ranges(sample_data: SampleData) -> None:
    """Test x and y axis range settings."""
    # Test setting both x and y ranges
    x_range = (0, 10)
    y_range = (-5, 5)
    fig = pmv.ptable_scatter_plotly(sample_data, x_range=x_range, y_range=y_range)

    # Check ranges were set correctly
    assert fig.layout.xaxis.range == x_range
    assert fig.layout.yaxis.range == y_range


AnnotationCallable = Callable[
    [Sequence[float]], str | dict[str, Any] | list[dict[str, Any]]
]
Annotations = dict[str, str | dict[str, Any]] | AnnotationCallable


@pytest.mark.parametrize(
    ("annotations", "expected_count"),
    [
        ({"Fe": "Iron note", "O": "Oxygen note"}, 2),  # dict annotations
        (lambda vals: f"Max: {max(vals[1]):.1f}", 2),  # annotate with callable
        ({"Fe": {"text": "Iron", "font_size": 12}}, 1),  # dict with styling
    ],
)
def test_annotations(
    sample_data: SampleData,
    annotations: Annotations,
    expected_count: int,
) -> None:
    """Test different types of annotations."""
    fig = pmv.ptable_scatter_plotly(sample_data, annotations=annotations)  # type: ignore[arg-type]

    if callable(annotations):
        # For callable annotations, check format
        anno_texts = [
            ann.text
            for ann in fig.layout.annotations
            if isinstance(ann.text, str) and ann.text.startswith("Max: ")
        ]
    elif isinstance(annotations, dict):
        # For dict annotations, check exact matches
        anno_texts = [
            ann.text
            for ann in fig.layout.annotations
            if any(
                (isinstance(v, str) and ann.text == v)
                or (isinstance(v, dict) and ann.text == v["text"])
                for v in annotations.values()
            )
        ]

    assert len(anno_texts) == expected_count


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_scaling(
    sample_data: SampleData,
    scale: float,
) -> None:
    """Test figure scaling with different values."""
    fig = pmv.ptable_scatter_plotly(sample_data, scale=scale)

    assert fig.layout.width == 850 * scale
    assert fig.layout.height == 500 * scale
    # Check font scaling
    symbol_annotations = [
        ann for ann in fig.layout.annotations if ann.text in {e.name for e in Element}
    ]
    assert all(ann.font.size == 11 * scale for ann in symbol_annotations)


def test_ptable_scatter_plotly_invalid_input() -> None:
    """Test that invalid input raises appropriate errors."""
    # Invalid mode should raise ValueError
    err_msg = "Invalid value of type 'builtins.str' received for the 'mode' property"
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        pmv.ptable_scatter_plotly({"Fe": ([1], [1])}, mode="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "axis_kwargs",
    [
        dict(title="Test", tickangle=45, showgrid=True),
        dict(nticks=5, tickmode="auto", zeroline=False),
    ],
)
def test_axis_styling(
    sample_data: SampleData,
    axis_kwargs: dict[str, Any],
) -> None:
    """Test axis styling options."""
    fig = pmv.ptable_scatter_plotly(
        sample_data, x_axis_kwargs=axis_kwargs, y_axis_kwargs=axis_kwargs
    )

    for key, val in axis_kwargs.items():
        # Handle nested properties like "title.text"
        key_path = dict(title="title.text").get(key, key)
        obj = fig.layout.xaxis
        for part in key_path.split("."):
            obj = getattr(obj, part)
        assert obj == val


def test_marker_styling(sample_data: SampleData) -> None:
    """Test marker and line customization."""
    marker_kwargs = dict(size=10, symbol="diamond", line=dict(width=2, color="white"))
    line_kwargs = dict(width=3, dash="dot")

    fig = pmv.ptable_scatter_plotly(
        sample_data,
        mode="lines+markers",
        marker_kwargs=marker_kwargs,
        line_kwargs=line_kwargs,
    )

    for trace in fig.data:
        if trace.x is not None:  # Skip empty traces
            assert trace.marker.size == 10
            assert trace.marker.symbol == "diamond"
            assert trace.marker.line.width == 2
            assert trace.marker.line.color == "white"
            assert trace.line.width == 3
            assert trace.line.dash == "dot"


def test_color_data_handling(color_data: ColorData) -> None:
    """Test handling of color data in the third sequence."""
    fig = pmv.ptable_scatter_plotly(color_data)

    # Find traces with data and map them by their y-values
    data_traces = [trace for trace in fig.data if trace.x]
    traces = {}
    for trace in data_traces:
        if trace.x[0] == 1:  # First x value should be 1 for all traces
            elem_name = trace.hovertemplate.split("<br>")[0]
            traces[elem_name] = trace

    # Check discrete colors for Fe
    assert traces["Iron"].marker.color == ("red", "blue")
    # Check numeric colors for Cu
    assert traces["Copper"].marker.color == (0.1, 0.9)
    # Check default color for O
    assert traces["Oxygen"].marker.color == "green"

    # Check line colors are removed when using color data
    assert traces["Iron"].line.color is None
    assert traces["Copper"].line.color is None


def test_color_precedence(color_data: ColorData) -> None:
    """Test color precedence: color_data > marker_kwargs > element_type_colors."""
    marker_kwargs = dict(color="yellow")

    fig = pmv.ptable_scatter_plotly(
        color_data, marker_kwargs=marker_kwargs, color_elem_strategy="symbol"
    )

    # Find traces with data and map them by their y-values
    data_traces = [trace for trace in fig.data if trace.x]
    traces = {}
    for trace in data_traces:
        if trace.x[0] == 1:  # First x value should be 1 for all traces
            elem_name = trace.hovertemplate.split("<br>")[0]
            traces[elem_name] = trace

    # Color data should override marker_kwargs
    assert traces["Iron"].marker.color == ("red", "blue")
    # marker_kwargs should override element type colors
    assert traces["Oxygen"].marker.color == "yellow"


def test_colorbar_with_numeric_colors(color_data: ColorData) -> None:
    """Test colorbar is added when numeric color values are provided."""
    # Create data with only numeric colors
    numeric_data = {
        "Fe": ([1, 2], [4, 5], [0.1, 0.5]),
        "Cu": ([1, 2], [7, 8], [0.3, 0.7]),
        "O": ([1, 2], [3, 4], [0.2, 0.6]),
    }

    # Test default colorbar
    fig = pmv.ptable_scatter_plotly(numeric_data)
    colorbar_traces = [
        trace
        for trace in fig.data
        if trace.x == (None,) and trace.marker.showscale is True
    ]
    assert len(colorbar_traces) == 1
    cbar = colorbar_traces[0].marker
    assert cbar.colorscale[0] == (0.0, "#440154")
    assert cbar.cmin == 0.1  # min of all color values
    assert cbar.cmax == 0.7  # max of all color values

    # Test custom colorbar settings
    fig = pmv.ptable_scatter_plotly(
        numeric_data,
        colorscale="RdYlBu",
        colorbar=dict(
            title="Test Title",
            orientation="v",
            thickness=20,
        ),
    )
    cbar = next(trace for trace in fig.data if trace.x == (None,)).marker
    assert cbar.colorscale[0] == (0.0, "rgb(165,0,38)")
    assert cbar.colorbar.title.text == "<br><br>Test Title"  # vertical title
    assert cbar.colorbar.thickness == 20
    assert cbar.colorbar.orientation == "v"

    # Test colorbar can be hidden
    fig = pmv.ptable_scatter_plotly(numeric_data, colorbar=False)
    colorbar_traces = [
        trace
        for trace in fig.data
        if trace.x == (None,) and trace.marker.showscale is True
    ]
    assert len(colorbar_traces) == 0

    # Test mixed numeric and string colors (should still show colorbar)
    fig = pmv.ptable_scatter_plotly(color_data)
    colorbar_traces = [
        trace
        for trace in fig.data
        if trace.x == (None,) and trace.marker.showscale is True
    ]
    assert len(colorbar_traces) == 1


def test_ptable_scatter_plotly_multi_line() -> None:
    """Test plotting multiple lines per element with a legend."""
    # Create test data with 2 lines per element for a few elements
    data = {
        "Fe": {"line1": ([1, 2, 3], [4, 5, 6]), "line2": ([1, 2, 3], [7, 8, 9])},
        "Co": {"line1": ([1, 2, 3], [5, 6, 7]), "line2": ([1, 2, 3], [8, 9, 10])},
    }

    fig = pmv.ptable_scatter_plotly(data, mode="lines")

    # Check that legend is shown
    assert fig.layout.showlegend is True

    # Check that we have the correct number of traces
    # 2 elements x 2 lines per element = 4 traces
    assert len(fig.data) == 4

    # Check that lines have correct names in legend
    line_names = {trace.name for trace in fig.data if trace.name}
    assert line_names == {"line1", "line2"}

    # Check that only first element shows in legend (to avoid duplicates)
    legend_traces = [trace for trace in fig.data if trace.showlegend]
    assert len(legend_traces) == 2

    # Check legend position
    assert fig.layout.legend.orientation == "h"
    assert fig.layout.legend.y == 0.74
    assert fig.layout.legend.x == 0.4

    dev_fig = fig.full_figure_for_development(warn=False)

    # Check that lines have consistent colors across elements
    line1_colors = {trace.line.color for trace in dev_fig.data if trace.name == "line1"}
    line2_colors = {trace.line.color for trace in dev_fig.data if trace.name == "line2"}
    # Each line type should have exactly one color
    assert len(line1_colors) == 1
    assert len(line2_colors) == 1
    # Colors should be different
    assert line1_colors != line2_colors


def test_ptable_scatter_plotly_hover_text() -> None:
    """Test that hover text is correctly formatted for multi-line plots."""
    data = {"Fe": {"line1": ([1], [4]), "line2": ([1], [7])}}

    fig = pmv.ptable_scatter_plotly(data, mode="lines")

    # Check hover text format
    for trace in fig.data:
        if trace.name == "line1":
            assert "Iron - line1" in trace.hovertemplate
        elif trace.name == "line2":
            assert "Iron - line2" in trace.hovertemplate


def test_mixed_length_lines() -> None:
    """Test plotting lines of different lengths in the same element."""
    data = {
        "Fe": {
            "short": ([1, 2], [3, 4]),
            "long": ([1, 2, 3, 4], [5, 6, 7, 8]),
        }
    }
    fig = pmv.ptable_scatter_plotly(data, mode="lines")

    # Check that both lines are plotted correctly
    assert len(fig.data) == 2
    assert len(fig.data[0].x) != len(fig.data[1].x)


def test_empty_element_data() -> None:
    """Test handling of empty sequences in element data."""
    data = {
        "H": {},
        "Fe": {"line1": ([], [])},  # Empty sequences
        "Cu": {"line1": ([1, 2], [3, 4])},  # Normal data
    }
    fig = pmv.ptable_scatter_plotly(data, mode="lines")

    # Check that empty element is skipped
    assert len(fig.data) == 2, "H should not have a trace"
    assert fig.data[0].hovertemplate.startswith("Iron")
    assert fig.data[1].hovertemplate.startswith("Copper")


def test_single_point_lines() -> None:
    """Test plotting lines with just one point."""
    data = {
        "Fe": {
            "single": ([1], [2]),
            "multi": ([1, 2, 3], [4, 5, 6]),
        }
    }
    fig = pmv.ptable_scatter_plotly(data, mode="lines")

    # Both should be plotted even though one has a single point
    assert len(fig.data) == 2


def test_duplicate_line_names() -> None:
    """Test handling of duplicate line names across elements."""
    data = {
        "Fe": {"common": ([1, 2], [3, 4])},
        "Cu": {"common": ([1, 2], [5, 6])},
    }
    fig = pmv.ptable_scatter_plotly(data, mode="lines")

    # Check that lines with same name have same color
    colors = {trace.line.color for trace in fig.data if trace.name == "common"}
    assert len(colors) == 1  # All lines named "common" should have same color


def test_mixed_input_types() -> None:
    """Test handling of mixed input types (lists, arrays, tuples)."""
    data = {
        "Fe": {
            "list": ([1, 2], [3, 4]),  # Python lists
            "array": (np.array([1, 2]), np.array([3, 4])),  # NumPy arrays
            "tuple": ((1, 2), (3, 4)),  # Python tuples
        }
    }
    fig = pmv.ptable_scatter_plotly(data, mode="lines")  # type: ignore[arg-type]

    # All should be plotted correctly
    assert len(fig.data) == 3
    assert {trace.name for trace in fig.data} == {"list", "array", "tuple"}
