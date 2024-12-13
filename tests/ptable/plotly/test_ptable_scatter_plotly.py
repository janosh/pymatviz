"""Test ptable_scatter_plotly function."""

import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

import plotly.graph_objects as go
import pytest

import pymatviz as pmv


@pytest.fixture
def sample_data() -> dict[str, tuple[list[float], list[float]]]:
    """Create sample data for testing."""
    return {
        "Fe": ([1, 2, 3], [4, 5, 6]),
        "O": ([0, 1, 2], [3, 4, 5]),
    }


def test_basic_scatter_plot(
    sample_data: dict[str, tuple[list[float], list[float]]],
) -> None:
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
        assert trace.marker.size == 3
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
    sample_data: dict[str, tuple[list[float], list[float]]],
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


def test_axis_ranges(sample_data: dict[str, tuple[list[float], list[float]]]) -> None:
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
    sample_data: dict[str, tuple[list[float], list[float]]],
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
    sample_data: dict[str, tuple[list[float], list[float]]],
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


def test_empty_data() -> None:
    """Test handling of empty data."""
    with pytest.raises(ValueError, match=re.escape("min() iterable argument is empty")):
        pmv.ptable_scatter_plotly({})


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
    sample_data: dict[str, tuple[list[float], list[float]]],
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
