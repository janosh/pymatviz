"""Tools related to image processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib
import matplotlib.colors


if TYPE_CHECKING:
    from typing import Literal


def luminance(color: str | tuple[float, float, float]) -> float:
    """Compute the luminance of a color as in https://stackoverflow.com/a/596243.

    Args:
        color (tuple[float, float, float]): RGB color tuple with values in [0, 1].

    Returns:
        float: Luminance of the color.
    """
    # raises ValueError if color invalid
    red, green, blue = matplotlib.colors.to_rgb(color)
    return 0.299 * red + 0.587 * green + 0.114 * blue


def pick_bw_for_contrast(
    color: tuple[float, float, float] | str,
    text_color_threshold: float = 0.7,
) -> Literal["black", "white"]:
    """Choose black or white text color for a given background color based on luminance.

    Args:
        color (tuple[float, float, float] | str): RGB color tuple with values in [0, 1].
        text_color_threshold (float, optional): Luminance threshold for choosing
            black or white text color. Defaults to 0.7.

    Returns:
        "black" | "white": depending on the luminance of the background color.
    """
    if isinstance(color, str):
        color = matplotlib.colors.to_rgb(color)

    light_bg = luminance(cast(tuple[float, float, float], color)) > text_color_threshold
    return "black" if light_bg else "white"
