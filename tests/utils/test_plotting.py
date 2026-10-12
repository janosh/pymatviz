from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pytest
from plotly.subplots import make_subplots

import pymatviz as pmv
from pymatviz.utils.plotting import annotated_heatmap, contrast_ratio


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    from pymatviz.typing import ColorType, RgbColorType


@pytest.mark.parametrize("labels", [list, np.array, pd.Index])
@pytest.mark.parametrize("reversescale", [False, True])
@pytest.mark.parametrize(
    ("colorscale", "font_colors", "expected_colors"),
    [
        ("Plasma", ["red", "blue"], ("red", "blue")),
        (None, None, ("#000000", "#000000")),
        (["black", "white"], None, ("#FFFFFF", "#000000")),
        (["rebeccapurple", "lightyellow"], None, ("#FFFFFF", "#000000")),
        ([[0, "white"], [1, "black"]], None, ("#000000", "#FFFFFF")),
        ("viridis", None, ("#FFFFFF", "#000000")),
        ("Viridis_r", None, ("#000000", "#FFFFFF")),
        ("Blues", None, ("#000000", "#FFFFFF")),
        ("RdBu", None, ("#FFFFFF", "#FFFFFF")),
    ],
)
def test_annotated_heatmap_array_labels(
    labels: type,
    reversescale: bool,
    colorscale: str | list | None,
    font_colors: list[str] | None,
    expected_colors: tuple[str, str],
) -> None:
    """Array-like labels and font colors retain their annotation positions."""
    fig = annotated_heatmap(
        [[0, 1], [2, 3]],
        x=labels(["left", "right"]),
        y=labels(["bottom", "top"]),
        colorscale=colorscale,
        font_colors=np.array(font_colors) if font_colors is not None else None,
        reversescale=reversescale,
    )
    if reversescale and font_colors is None:
        expected_colors = expected_colors[::-1]
    low_color, high_color = expected_colors
    assert [(anno.x, anno.y, anno.font.color) for anno in fig.layout.annotations] == [
        ("left", "bottom", low_color),
        ("right", "bottom", low_color),
        ("left", "top", high_color),
        ("right", "top", high_color),
    ]
    assert list(fig.data[0].x) == ["left", "right"]


@pytest.mark.parametrize(
    ("name", "channels"),
    [
        ("lightblue", (173, 216, 230)),
        ("RoyalBlue", (65, 105, 225)),
        (" rebeccapurple ", (102, 51, 153)),
        ("darkslategray", (47, 79, 79)),
    ],
)
def test_css_named_colors(name: str, channels: tuple[int, int, int]) -> None:
    """Support every Plotly CSS name with the standard RGB values."""
    from _plotly_utils.basevalidators import ColorValidator

    from pymatviz.utils.plotting import rgb_components

    np.testing.assert_array_equal(rgb_components(name), np.array(channels) / 255)
    for color in ColorValidator.named_colors:
        assert 0 <= pmv.utils.luminance(color) <= 1


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ((0, 0, 0), 0),  # Black
        ((1, 1, 1), 1),  # White
        ((0.5, 0.5, 0.5), 0.21404114048223255),  # Gray
        ((1, 0, 0), 0.2126),  # Red
        ((0, 1, 0), 0.7152),  # Green
        ((0, 0, 1, 0.3), 0.0722),  # Blue with alpha (should be ignored)
        ("#FF0000", 0.2126),  # Red
        ("#00FF00", 0.7152),  # Green
        ("#0000FF", 0.0722),  # Blue
        ("red", 0.2126),
        ("green", 0.1543834296814607),
        ("gray", 0.21586050011389926),
        (" GREY ", 0.21586050011389926),
        ("#abc", 0.4844632879252147),
        ("blue", 0.0722),
        # RGB color string tests
        ("rgb(255, 0, 0)", 0.2126),  # Red in RGB format
        ("rgb(0, 255, 0)", 0.7152),  # Green in RGB format
        ("rgb(0, 0, 255)", 0.0722),  # Blue in RGB format
        ("rgb(128, 128, 128)", 0.21586050011389926),  # Gray in RGB format
        ("rgb(255, 255, 255)", 1.0),  # White in RGB format
        ("rgb(0, 0, 0)", 0.0),  # Black in RGB format
        ("rgb(255, 0, 0, 0.5)", 0.2126),  # Red with alpha
        # Edge cases
        ("rgb(255,0,0)", 0.2126),  # No spaces
        ("rgb( 255, 0, 0 )", 0.2126),  # Extra spaces
        ("rgb(127.5, 127.5, 127.5)", 0.21404114048223255),  # Decimal values
        ("rgb(1, 0, 0)", 0.2126 / (255 * 12.92)),
        ("rgb(0, 1, 0)", 0.7152 / (255 * 12.92)),
        ("rgb(0, 0, 1)", 0.0722 / (255 * 12.92)),
        ("rgb(1, 1, 1)", 1 / (255 * 12.92)),
        (" RGB(255,0,0) ", 0.2126),
        ("rgba(255,0,0,0.5)", 0.2126),
        # NumPy scalar channels
        ((np.int32(128), np.int32(128), np.int32(128)), 0.21586050011389926),
    ],
)
def test_luminance(color: RgbColorType, expected: float) -> None:
    """RGB strings use CSS byte channels; tuples may use normalized channels."""
    assert pmv.utils.luminance(color) == pytest.approx(
        expected, rel=1e-12, abs=1e-15
    ), f"{color=}"


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ((1.0, 1.0, 1.0), "black"),  # White
        ((0, 0, 0), "white"),  # Black
        ((0.5, 0.5, 0.5), "white"),  # Gray
        ((1, 0, 0, 0.3), "white"),  # Red with alpha (should be ignored)
        ((0, 1, 0), "black"),  # Green
        ((0, 0, 1.0), "white"),  # Blue
        # Standard color names
        ("black", "white"),
        ("white", "black"),
        ("red", "white"),
        ("yellow", "black"),
        ("green", "white"),
        # Hex colors
        ("#000080", "white"),  # Navy blue (dark)
        ("#FFFF00", "black"),  # Yellow (light)
        ("#8B0000", "white"),  # Dark red
        ("#90EE90", "black"),  # Light green
        # RGB format
        ("rgb(1, 1, 1)", "white"),
        ("rgb(255, 255, 255)", "black"),  # White
        ("rgb(128, 0, 0)", "white"),  # Maroon
        ("rgb(0, 128, 0)", "white"),  # Green
        ("rgb(0, 0, 128)", "white"),  # Navy
        ("rgb(200, 200, 200)", "black"),  # Light gray
        ("rgb(100, 100, 100)", "white"),  # Medium gray
        ("rgb( 50, 50, 50 )", "white"),  # Very dark gray with spaces
        # Edge cases near the threshold
        ("rgb(76, 76, 76)", "white"),  # Just below threshold
        ("rgb(77, 77, 77)", "white"),  # Near threshold
    ],
)
def test_pick_max_contrast_color(
    color: RgbColorType,
    expected: Literal["black", "white"],
) -> None:
    """Dark backgrounds get white text and light backgrounds black text."""
    assert pmv.utils.pick_max_contrast_color(color) == expected


@pytest.mark.parametrize(
    "color",
    [
        "not_a_color",
        "rgb(255,0,0",
        "rgb(255,0,0))",
        "rgb(1,2)",
        "rgb(1,2,3,4,5)",
        "rgb(255,0,0,nan)",
        "rgba(255,0,0,2)",
        "rgb(256,0,0)",
        "rgb(-1,0,0)",
        "rgb(nan,0,0)",
        "rgb(inf,0,0)",
        "rgb(red,0,0)",
        "##fff",
        "#ff",
        "#-ff",
        "#ggg",
        (-1, 0, 0),
        (256, 0, 0),
        (float("nan"), 0, 0),
        (float("inf"), 0, 0),
        ("red", 0, 0),
        (0, 0),
        (0, 0, 0, 0, 0),
    ],
)
def test_luminance_with_edge_cases(color: ColorType) -> None:
    """Malformed and out-of-range colors fail before producing invalid luminance."""
    with pytest.raises(ValueError, match=r"color|channels"):
        pmv.utils.luminance(color)


@pytest.mark.parametrize(
    ("bg_color", "colors", "min_contrast_ratio", "expected"),
    [
        ("rgb(128, 128, 128)", ("white", "black"), 3.0, "white"),
        ("rgb(128, 128, 128)", ("white", "black"), 1.0, "white"),
        ("black", ("red", "blue"), 2.0, "red"),
        ("black", ("yellow", "green"), 2.0, "yellow"),
        ("white", ("red", "blue"), 10, "blue"),
        ("white", ("yellow", "green"), 10, "green"),
    ],
)
def test_pick_max_contrast_color_options(
    bg_color: str,
    colors: tuple[str, str],
    min_contrast_ratio: float,
    expected: str,
) -> None:
    """Custom text colors and contrast thresholds pick the expected color."""
    picked = pmv.utils.pick_max_contrast_color(
        bg_color, colors=colors, min_contrast_ratio=min_contrast_ratio
    )
    assert picked == expected


def test_contrast_ratio() -> None:
    """Test the contrast_ratio function with various color combinations."""
    # Test black and white (should be 21:1)
    assert abs(contrast_ratio("black", "white") - 21.0) < 0.1
    assert abs(contrast_ratio("white", "black") - 21.0) < 0.1

    # Test same colors (should be 1:1)
    assert abs(contrast_ratio("red", "red") - 1.0) < 0.1
    assert abs(contrast_ratio("blue", "blue") - 1.0) < 0.1

    # Test some common combinations
    assert 3.5 < contrast_ratio("red", "white") < 4.5  # Around 4.0:1
    assert 8.0 < contrast_ratio("blue", "white") < 9.0  # Around 8.59:1
    assert 15.0 < contrast_ratio("yellow", "black") < 20.0  # Around 19.56:1

    # Test that order doesn't matter
    assert abs(contrast_ratio("red", "white") - contrast_ratio("white", "red")) < 0.1


@pytest.mark.parametrize("color", ["red", "blue", "#FF0000"])
def test_annotate(color: str, plotly_scatter: go.Figure) -> None:
    text = "Test annotation"

    fig_plotly = pmv.utils.annotate(text, plotly_scatter, color=color)
    assert fig_plotly.layout.annotations[-1].text == text
    assert fig_plotly.layout.annotations[-1].font.color == color


@pytest.mark.parametrize(
    ("func", "kwargs", "err_msg"),
    [
        (pmv.utils.annotate, {"text": "test"}, "Expected plotly Figure"),
        (pmv.utils.get_fig_xy_range, {}, "Expected plotly Figure"),
        (pmv.utils.get_font_color, {}, "Input must be plotly Figure, got type(fig)="),
    ],
)
def test_non_figure_input_raises(
    func: Callable[..., Any], kwargs: dict[str, Any], err_msg: str
) -> None:
    """Figure helpers reject non-figure input with a TypeError."""
    with pytest.raises(TypeError, match=re.escape(err_msg)):
        func(fig="invalid", **kwargs)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (["A1", "A2"], [("A1", "x domain"), ("A2", "x2 domain")]),
        (["A1", ""], [("A1", "x domain")]),  # empty strings skip a facet
        ("Single", [("Single", "x domain"), ("Single", "x2 domain")]),
    ],
    ids=["per-facet", "skip-empty", "broadcast-str"],
)
def test_annotate_faceted_plotly(
    plotly_faceted_scatter: go.Figure,
    text: str | list[str],
    expected: list[tuple[str, str]],
) -> None:
    """Faceted figures get one annotation per non-empty text in subplot domains."""
    fig = pmv.utils.annotate(text, plotly_faceted_scatter)
    assert [(anno.text, anno.xref) for anno in fig.layout.annotations] == expected


def test_annotate_non_faceted_plotly_with_list_raises(
    plotly_scatter: go.Figure,
) -> None:
    text = ["Annotation 1", "Annotation 2"]
    text_type = type(text).__name__
    with pytest.raises(
        ValueError,
        match=re.escape(f"Unexpected {text_type=} for non-faceted plot, must be str"),
    ):
        pmv.utils.annotate(text, plotly_scatter)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": 0.5, "y": 0.5},
        {"font": dict(size=20, color="green")},
        {"showarrow": True, "arrowhead": 2},
    ],
)
def test_annotate_kwargs(plotly_scatter: go.Figure, kwargs: dict[str, Any]) -> None:
    fig: go.Figure = pmv.utils.annotate("Test", plotly_scatter, **kwargs)

    for key, val in kwargs.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                assert getattr(fig.layout.annotations[-1][key], sub_key) == sub_val
        else:
            assert getattr(fig.layout.annotations[-1], key) == val


@pytest.mark.parametrize("selector", [[0, 1], slice(None), lambda _trace: True])
@pytest.mark.parametrize("axis_type", ["linear", "log"])
def test_get_fig_xy_range(selector: Any, axis_type: str) -> None:
    """Ranges pad finite data of selected traces; fixed layout ranges win."""
    fig = go.Figure().add_scatter(x=[1, 10, np.nan], y=[2, 20, 5])
    fig.add_scatter(x=[100, 1000], y=[np.inf, 200])
    fig.add_histogram(x=[1e6])  # no y data, so never used for ranges
    fig.update_xaxes(type=axis_type).update_yaxes(type=axis_type)

    def check(expected: tuple[tuple[float, float], ...], **kwargs: Any) -> None:
        """Compare get_fig_xy_range output to expected ranges."""
        actual = pmv.utils.get_fig_xy_range(fig, **kwargs)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)

    if axis_type == "log":  # pad 5% of the log10 span: 3 and 2 decades
        expected = ((10**-0.15, 10**3.15), (2 * 10**-0.1, 200 * 10**0.1))
        single = ((10**1.95, 10**3.05), (20, 2000))  # zero span pads 1 decade
    else:
        expected = ((1 - 49.95, 1000 + 49.95), (2 - 9.9, 200 + 9.9))
        single = ((100 - 45, 1000 + 45), (199, 201))  # zero span pads 1
    check(expected, traces=selector)
    check(single, traces=1)
    check(((1, 1000), (2, 200)), traces=selector, pad_frac=0)

    fig.update_xaxes(range=[0, 3])  # log axis ranges are in log10 units
    fixed_max = 1000 if axis_type == "log" else 3
    check(((1 if axis_type == "log" else 0, fixed_max), expected[1]), traces=selector)
    fig.update_xaxes(range=[None, 3])
    check(((expected[0][0], fixed_max), expected[1]), traces=selector)

    with pytest.raises(ValueError, match="No valid traces with required data found"):
        pmv.utils.get_fig_xy_range(fig, traces=lambda _trace: False)


def test_get_fig_xy_range_subplots_and_errors() -> None:
    """Subplot traces share one range; mixed axis types and bad data raise."""
    fig = make_subplots(rows=1, cols=2)
    fig.add_scatter(x=[1, 2], y=[10, 20], row=1, col=1)
    fig.add_scatter(x=[3, 5], y=[30, 50], row=1, col=2)
    ranges = pmv.utils.get_fig_xy_range(fig, traces=slice(None), pad_frac=0)
    assert ranges == ((1, 5), (10, 50))

    fig.update_xaxes(type="log", row=1, col=2)
    with pytest.raises(ValueError, match="span x-axes of different types"):
        pmv.utils.get_fig_xy_range(fig, traces=slice(None))

    fig_neg = go.Figure().add_scatter(x=[1, 2], y=[-1, 0]).update_yaxes(type="log")
    with pytest.raises(ValueError, match="No finite positive data for y-axis"):
        pmv.utils.get_fig_xy_range(fig_neg)

    fig_cat = go.Figure().add_scatter(x=["a", "b"], y=[1, 2])
    with pytest.raises(ValueError, match="Non-numeric x data"):
        pmv.utils.get_fig_xy_range(fig_cat)


@pytest.mark.parametrize(
    ("layout_color", "template_color", "default_template", "expected"),
    [
        ("red", "blue", "plotly", "red"),
        ("#00FF00", "blue", "plotly", "#00FF00"),
        ("rgb(0, 0, 255)", "blue", "plotly", "rgb(0, 0, 255)"),
        (None, "blue", "plotly", "blue"),
        (None, None, "plotly", "#2a3f5f"),
        (None, None, "none", "black"),
    ],
)
def test_get_font_color(
    monkeypatch: pytest.MonkeyPatch,
    layout_color: str | None,
    template_color: str | None,
    default_template: str,
    expected: str,
) -> None:
    """Font colors prefer layout, figure template, then global template."""
    monkeypatch.setattr(pio.templates, "default", default_template)
    fig = go.Figure().update_layout(
        font_color=layout_color,
        template=go.layout.Template(layout=dict(font_color=template_color)),
    )
    assert pmv.utils.get_font_color(fig) == expected
