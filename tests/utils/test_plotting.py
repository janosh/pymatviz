from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import pymatviz as pmv
from pymatviz.utils.plotting import contrast_ratio


if TYPE_CHECKING:
    from typing import Literal

    from pymatviz.typing import RgbColorType


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ((0, 0, 0), 0),  # Black
        ((1, 1, 1), 1),  # White
        ((0.5, 0.5, 0.5), 0.21404),  # Gray
        ((1, 0, 0), 0.2126),  # Red
        ((0, 1, 0), 0.7152),  # Green
        ((0, 0, 1, 0.3), 0.0722),  # Blue with alpha (should be ignored)
        ("#FF0000", 0.2126),  # Red
        ("#00FF00", 0.7152),  # Green
        ("#0000FF", 0.0722),  # Blue
        ("red", 0.2126),
        ("green", 0.15438),  # Matplotlib's green
        ("blue", 0.0722),
        # RGB color string tests
        ("rgb(255, 0, 0)", 0.2126),  # Red in RGB format
        ("rgb(0, 255, 0)", 0.7152),  # Green in RGB format
        ("rgb(0, 0, 255)", 0.0722),  # Blue in RGB format
        ("rgb(128, 128, 128)", 0.21586),  # Gray in RGB format
        ("rgb(255, 255, 255)", 1.0),  # White in RGB format
        ("rgb(0, 0, 0)", 0.0),  # Black in RGB format
        ("rgb(255, 0, 0, 0.5)", 0.2126),  # Red with alpha
        # Edge cases
        ("rgb(255,0,0)", 0.2126),  # No spaces
        ("rgb( 255, 0, 0 )", 0.2126),  # Extra spaces
        ("rgb(127.5, 127.5, 127.5)", 0.21404),  # Decimal values
        # Values already in [0,1] range
        ("rgb(1, 0, 0)", 0.2126),  # Red with values in [0,1] range
        ("rgb(0, 1, 0)", 0.7152),  # Green with values in [0,1] range
        ("rgb(0, 0, 1)", 0.0722),  # Blue with values in [0,1] range
    ],
)
def test_luminance(color: RgbColorType, expected: float) -> None:
    assert pmv.utils.luminance(color) == pytest.approx(expected, 0.001), f"{color=}"


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


def test_luminance_with_edge_cases() -> None:
    """Test the luminance function with edge cases."""
    # Test with standard color names
    assert abs(pmv.utils.luminance("black") - 0.0) < 0.01
    assert abs(pmv.utils.luminance("white") - 1.0) < 0.01
    assert abs(pmv.utils.luminance("red") - 0.2126) < 0.01

    # The actual value for "green" in matplotlib is different from pure green
    green_lum = pmv.utils.luminance("green")
    assert 0.15 < green_lum < 0.16

    # Test with blue
    assert abs(pmv.utils.luminance("blue") - 0.0722) < 0.01

    # Test with invalid color
    with pytest.raises(ValueError, match="Invalid RGBA argument: 'not_a_color'"):
        pmv.utils.luminance("not_a_color")


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
