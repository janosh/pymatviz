from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import pymatviz as pmv


if TYPE_CHECKING:
    from typing import Literal


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ((0, 0, 0), 0),  # Black
        ((1, 1, 1), 1),  # White
        ((0.5, 0.5, 0.5), 0.5),  # Gray
        ((1, 0, 0), 0.2126),  # Red
        ((0, 1, 0), 0.7152),  # Green
        ((0, 0, 1, 0.3), 0.0722),  # Blue with alpha (should be ignored)
        ("#FF0000", 0.2126),  # Red
        ("#00FF00", 0.7152),  # Green
        ("#0000FF", 0.0722),  # Blue
        ("red", 0.2126),
        ("green", 0.35900235),
        ("blue", 0.0722),
        # RGB color string tests
        ("rgb(255, 0, 0)", 0.2126),  # Red in RGB format
        ("rgb(0, 255, 0)", 0.7152),  # Green in RGB format
        ("rgb(0, 0, 255)", 0.0722),  # Blue in RGB format
        (
            "rgb(128, 128, 128)",
            0.5019607843137255,
        ),  # Gray in RGB format (128/255 ≈ 0.502)
        ("rgb(255, 255, 255)", 1.0),  # White in RGB format
        ("rgb(0, 0, 0)", 0.0),  # Black in RGB format
        # RGBA color string tests (alpha should be ignored)
        ("rgb(255, 0, 0, 0.5)", 0.2126),  # Red with alpha
        # Edge cases
        ("rgb(255,0,0)", 0.2126),  # No spaces
        ("rgb( 255, 0, 0 )", 0.2126),  # Extra spaces
        # Mixed values (decimal and integer)
        ("rgb(127.5, 127.5, 127.5)", 0.5),  # Decimal values
        # Values already in [0,1] range
        ("rgb(1, 0, 0)", 0.2126),  # Red with values in [0,1] range
        ("rgb(0, 1, 0)", 0.7152),  # Green with values in [0,1] range
        ("rgb(0, 0, 1)", 0.0722),  # Blue with values in [0,1] range
    ],
)
def test_luminance(color: tuple[float, float, float], expected: float) -> None:
    assert pmv.utils.luminance(color) == pytest.approx(expected, 0.001), f"{color=}"


@pytest.mark.parametrize(
    ("color", "luminance_threshold", "expected"),
    [
        ((1.0, 1.0, 1.0), 0.7, "black"),  # White
        ((0, 0, 0), 0.7, "white"),  # Black
        ((0.5, 0.5, 0.5), 0.7, "white"),  # Gray
        ((0.5, 0.5, 0.5), 0, "black"),  # Gray with low threshold
        ((1, 0, 0, 0.3), 0.7, "white"),  # Red with alpha (should be ignored)
        ((0, 1, 0), 0.7, "black"),  # Green
        ((0, 0, 1.0), 0.4, "white"),  # Blue with low threshold
    ],
)
def test_pick_max_contrast_color(
    color: tuple[float, float, float],
    luminance_threshold: float,
    expected: Literal["black", "white"],
) -> None:
    assert pmv.utils.pick_max_contrast_color(color, luminance_threshold) == expected
