"""Test ptable_scatter_plotly function."""

import re
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import plotly.graph_objects as go
import pytest

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
        assert axis.ticks == "outside"
        assert axis.tickwidth == 1
        assert axis.nticks == 2
        assert axis.tickmode == "auto"

    # Check that zeroline is False for both axes
    assert fig.layout.xaxis.zeroline is False
    assert fig.layout.yaxis.zeroline is False

    # Check element annotations
    symbol_annotations = [
        ann for ann in fig.layout.annotations if "<b>" in str(ann.text)
    ]
    assert len(symbol_annotations) == len(sample_data)

    for ann in symbol_annotations:
        assert ann.showarrow is False
        assert ann.xanchor == "right"
        assert ann.yanchor == "top"
        assert ann.x == 1
        assert ann.y == 1
        assert ann.font.size == 12  # default font size
        # Check that text is either <b>Fe</b> or <b>O</b>
        assert ann.text in ("<b>Fe</b>", "<b>O</b>")


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
    line_kwargs: dict[str, Any] = dict(color=expected_color)
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
        (lambda vals: f"Max: {max(vals):.1f}", 2),  # callable annotations
        ({"Fe": {"text": "Iron", "font_size": 12}}, 1),  # dict with styling
    ],
)
def test_annotations(
    sample_data: SampleData,
    annotations: Annotations,
    expected_count: int,
) -> None:
    """Test different types of annotations."""
    fig = pmv.ptable_scatter_plotly(sample_data, annotations=annotations)

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
        ann for ann in fig.layout.annotations if "<b>" in str(ann.text)
    ]
    assert all(ann.font.size == 12 * scale for ann in symbol_annotations)


def test_invalid_modes() -> None:
    """Test invalid plot modes."""
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
    assert traces["Oxygen"].marker.color == "white"

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
