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
        ((1, 0, 0), 0.299),  # Red
        ((0, 1, 0), 0.587),  # Green
        ((0, 0, 1, 0.3), 0.114),  # Blue with alpha (should be ignored)
        ("#FF0000", 0.299),  # Red
        ("#00FF00", 0.587),  # Green
        ("#0000FF", 0.114),  # Blue
        ("red", 0.299),
        ("green", 0.294650),
        ("blue", 0.114),
    ],
)
def test_luminance(color: tuple[float, float, float], expected: float) -> None:
    assert pmv.utils.luminance(color) == pytest.approx(expected, 0.001), f"{color=}"


@pytest.mark.parametrize(
    ("color", "text_color_threshold", "expected"),
    [
        ((1.0, 1.0, 1.0), 0.7, "black"),  # White
        ((0, 0, 0), 0.7, "white"),  # Black
        ((0.5, 0.5, 0.5), 0.7, "white"),  # Gray
        ((0.5, 0.5, 0.5), 0, "black"),  # Gray with low threshold
        ((1, 0, 0, 0.3), 0.7, "white"),  # Red with alpha (should be ignored)
        ((0, 1, 0), 0.7, "white"),  # Green
        ((0, 0, 1.0), 0.4, "white"),  # Blue with low threshold
    ],
)
def test_pick_bw_for_contrast(
    color: tuple[float, float, float],
    text_color_threshold: float,
    expected: Literal["black", "white"],
) -> None:
    assert pmv.utils.pick_bw_for_contrast(color, text_color_threshold) == expected
