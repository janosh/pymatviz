"""Test color schemes defined in pymatviz.colors."""

from __future__ import annotations

import numpy as np
import pytest

import pymatviz.colors as pmv_colors


@pytest.mark.parametrize(
    ("color_scheme", "expected_type", "value_range"),
    [
        (pmv_colors.ELEM_TYPE_COLORS, str, None),  # String color names
        (pmv_colors.ELEM_COLORS_JMOL_256, tuple, (0, 255)),  # 8-bit RGB
        (pmv_colors.ELEM_COLORS_VESTA_256, tuple, (0, 255)),
        (pmv_colors.ELEM_COLORS_ALLOY_256, tuple, (0, 255)),
        (pmv_colors.ELEM_COLORS_JMOL, tuple, (0, 1)),  # Normalized RGB
        (pmv_colors.ELEM_COLORS_VESTA, tuple, (0, 1)),
        (pmv_colors.ELEM_COLORS_ALLOY, tuple, (0, 1)),
    ],
)
def test_color_schemes(
    color_scheme: dict[str, str | tuple[float | int, ...]],
    expected_type: type,
    value_range: tuple[int | float, int | float] | None,
) -> None:
    """Test color scheme types and value ranges."""
    assert isinstance(color_scheme, dict)
    for key, rgb_color in color_scheme.items():
        assert isinstance(key, str)
        assert isinstance(rgb_color, expected_type)
        if value_range:
            assert isinstance(rgb_color, tuple)
            assert len(rgb_color) == 3  # RGB tuple
            assert all(isinstance(val, int | float) for val in rgb_color)
            assert all(value_range[0] <= val <= value_range[1] for val in rgb_color)
            assert sum(np.isnan(val) for val in rgb_color) == 0
