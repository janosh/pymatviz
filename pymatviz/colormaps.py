"""Generate custom colormaps."""
# TODO: min_value/max_value need better doc


from __future__ import annotations

import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap


def blend_two(
    cmap_above: Colormap,
    cmap_below: Colormap,
    min_value: float,
    max_value: float,
    separator: float = None,
) -> Colormap:
    """Blend two matplotlib colormaps above and below a specified separator value.

    Parameters:
    - cmap_above (Colormap): The colormap for values above the separator.
    - cmap_below (Colormap): The colormap for values below the separator.
    - min_value (float): The minimum value for the color scale.
    - max_value (float): The maximum value for the color scale.
    - separator (float, optional): The separator value. Must be between min_value and max_value.
        Defaults to None, would calculate the average of min/max values.

    Returns:
    - Colormap: A blended colormap.

    Raises:
    - AssertionError: If the separator is not within the valid range.

    References:
    - https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps/31052741#31052741
    - https://stackoverflow.com/questions/22128166/two-different-color-colormaps-in-the-same-imshow-matplotlib
    """
    # Calculate separator position as a ratio between min and max values
    assert min_value < separator < max_value
    if separator is None:
        separator = (min_value + max_value) / 2
    separator_position = 1 - ((separator - min_value) / (max_value - min_value))

    # Load separate colormaps
    colors_cmap_above = cmap_above(np.linspace(0, 1, int(separator_position * 256)))
    colors_cmap_below = cmap_below(
        np.linspace(0, 1, int((1 - separator_position) * 256))
    )

    # Blend the colors
    blended_colors = np.vstack((colors_cmap_below, colors_cmap_above))

    return LinearSegmentedColormap.from_list("custom_colormap", blended_colors, N=256)
