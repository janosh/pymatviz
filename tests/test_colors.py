"""Test public color palette invariants."""

from __future__ import annotations

import pytest

import pymatviz.colors as pmv_colors


COLOR_PALETTE_PAIRS = {
    "jmol": (pmv_colors.ELEM_COLORS_JMOL_256, pmv_colors.ELEM_COLORS_JMOL),
    "vesta": (pmv_colors.ELEM_COLORS_VESTA_256, pmv_colors.ELEM_COLORS_VESTA),
    "alloy": (pmv_colors.ELEM_COLORS_ALLOY_256, pmv_colors.ELEM_COLORS_ALLOY),
    "pastel": (pmv_colors.ELEM_COLORS_PASTEL_256, pmv_colors.ELEM_COLORS_PASTEL),
}


def test_complete_color_palettes_cover_same_elements() -> None:
    """Complete public RGB palettes expose the same element symbols."""
    complete_palettes = ("jmol", "vesta", "alloy")
    key_sets = {
        palette_name: set(COLOR_PALETTE_PAIRS[palette_name][0])
        for palette_name in complete_palettes
    }

    assert key_sets["jmol"] == key_sets["vesta"] == key_sets["alloy"]
    assert len(key_sets["jmol"]) == 109
    assert {"H", "C", "Fe", "U", "Mt"} <= key_sets["jmol"]


@pytest.mark.parametrize(
    ("palette_name", "colors_256", "colors_normalized"),
    [(name, *palettes) for name, palettes in COLOR_PALETTE_PAIRS.items()],
)
def test_normalized_color_palettes_match_256_sources(
    palette_name: str,
    colors_256: dict[str, tuple[int, int, int]],
    colors_normalized: dict[str, tuple[float, float, float]],
) -> None:
    """Normalized RGB palettes are derived from their 256-channel sources."""
    assert set(colors_normalized) == set(colors_256), palette_name

    for element_symbol, rgb_256 in colors_256.items():
        assert len(rgb_256) == 3
        assert all(isinstance(channel, int) for channel in rgb_256)
        assert all(0 <= channel <= 255 for channel in rgb_256)

        rgb_normalized = colors_normalized[element_symbol]
        assert len(rgb_normalized) == 3
        assert all(0 <= channel <= 1 for channel in rgb_normalized)
        assert rgb_normalized == pytest.approx(
            tuple(channel / 255 for channel in rgb_256)
        )
