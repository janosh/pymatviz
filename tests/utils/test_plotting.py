from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest

import pymatviz as pmv
from pymatviz.utils.plotting import annotated_heatmap, contrast_ratio


if TYPE_CHECKING:
    from typing import Literal

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

    from pymatviz.utils.plotting import _rgb_components

    np.testing.assert_array_equal(_rgb_components(name), np.array(channels) / 255)
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
    ],
)
def test_pick_max_contrast_color(
    color: RgbColorType,
    expected: Literal["black", "white"],
) -> None:
    assert pmv.utils.pick_max_contrast_color(color) == expected


def test_text_color_contrast() -> None:
    """Test that pick_max_contrast_color returns the correct text color for contrast."""
    test_cases = [
        # Standard color names
        ("black", "white"),  # Dark color should get white text
        ("white", "black"),  # Light color should get black text
        ("red", "white"),  # Red gets white text
        ("yellow", "black"),  # Light color should get black text
        # Hex colors
        ("#000000", "white"),  # Black
        ("#FFFFFF", "black"),  # White
        ("#000080", "white"),  # Navy blue (dark)
        ("#FFFF00", "black"),  # Yellow (light)
        ("#8B0000", "white"),  # Dark red
        ("#90EE90", "black"),  # Light green
        # RGB format
        ("rgb(0, 0, 0)", "white"),  # Black
        ("rgb(1, 1, 1)", "white"),
        ("green", "white"),
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
    ]

    for color, expected in test_cases:
        actual = pmv.utils.pick_max_contrast_color(color)
        assert actual == expected, f"For {color}, expected {expected}, got {actual}"


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


def test_luminance_accepts_numpy_scalar_rgb_tuple() -> None:
    """Luminance accepts NumPy scalar channels in RGB tuples."""
    color = (np.int32(128), np.int32(128), np.int32(128))
    assert pmv.utils.luminance(cast("ColorType", color)) == pytest.approx(
        0.21586, abs=0.001
    )


def test_pick_max_contrast_color_with_min_contrast_ratio() -> None:
    """Test pick_max_contrast_color with custom min_contrast_ratio."""
    # Medium gray
    color = "rgb(128, 128, 128)"

    # Test with default min_contrast_ratio and custom min_contrast_ratio
    assert pmv.utils.pick_max_contrast_color(color) == "white"
    assert pmv.utils.pick_max_contrast_color(color, min_contrast_ratio=3.0) == "white"
    assert pmv.utils.pick_max_contrast_color(color, min_contrast_ratio=1.0) == "white"


def test_pick_max_contrast_color_with_custom_colors() -> None:
    """Test pick_max_contrast_color with custom contrast colors."""
    # Dark background tests
    dark_color = "black"
    assert pmv.utils.pick_max_contrast_color(dark_color) == "white"
    assert (
        pmv.utils.pick_max_contrast_color(dark_color, colors=("red", "blue")) == "red"
    )
    assert (
        pmv.utils.pick_max_contrast_color(dark_color, colors=("yellow", "green"))
        == "yellow"
    )

    # Light background tests
    light_color = "white"
    assert pmv.utils.pick_max_contrast_color(light_color) == "black"
    assert (
        pmv.utils.pick_max_contrast_color(
            light_color, colors=("red", "blue"), min_contrast_ratio=10
        )
        == "blue"
    )
    assert (
        pmv.utils.pick_max_contrast_color(
            light_color, colors=("yellow", "green"), min_contrast_ratio=10
        )
        == "green"
    )


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
