"""Generate custom colormaps."""


from __future__ import annotations

import numpy as np
import matplotlib
from matplotlib.colors import Colormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap


def composite_colormaps(
    cmap_above: str | Colormap,  # TODO: double-check type hint
    cmap_below: str | Colormap,
    min_value: float,  # TODO: what is this?
    max_value: float,  # TODO: what is this?
    separator: float = None,
    N: int = 256,
) -> Colormap:
    """Create a composite matplotlib Colormap by merging two colormaps
        above and below a specified separator value.

    Parameters:
    - cmap_above (str | Colormap): The colormap for values above the separator.
    - cmap_below (str | Colormap): The colormap for values below the separator.
    - separator (float, optional): The separator value between min and max values.  # TODO:
        Defaults to the average of min/max values.
    - N (int): The number of RGB quantization levels. Defaults to 256.

    Returns:
    - Colormap: A composite colormap created by merging the provided colormaps.

    Raises:
    - AssertionError: If the separator is not within the valid range.

    References:
    - https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps/31052741#31052741

    TODO: generalize to composite "multiple" Colormaps?
    """
    # Check and load Colormaps
    if isinstance(cmap_above, str):
        cmap_above = matplotlib.colormaps[cmap_above]
    elif not isinstance(cmap_above, Colormap):
        raise TypeError(f"Expect type Colormap or str for cmap_above, got {type(cmap_above)}.")

    if isinstance(cmap_below, str):
        cmap_below = matplotlib.colormaps[cmap_below]
    elif not isinstance(cmap_below, Colormap):
        raise TypeError(f"Expect type Colormap or str for cmap_below, got {type(cmap_below)}.")

    # Merge two Colormaps
    colors_cmap_above = cmap_above(np.linspace(0, 1, int(separator_position * N)))
    colors_cmap_below = cmap_below(
        np.linspace(0, 1, int((1 - separator_position) * N))
    )
    blended_colors = np.vstack((colors_cmap_below, colors_cmap_above))

    return LinearSegmentedColormap.from_list("custom_colormap", blended_colors, N=N)


def truncate(
    cmap: str | Colormap,
    start: float,
    end: float,
    N: int = 256,
) -> Colormap:
    """Truncate a matplotlib Colormap to a specified range.

    This function takes a colormap and truncates it to a specified range
    defined by the start and end points, creating a new colormap.

    Parameters:
    - cmap (str or Colormap): The original Colormap.
    - start (float): The starting point of the truncated colormap (in range [0, 1]).
    - end (float): The ending point of the truncated colormap (in range [0, 1]).
    - N (int, optional): The number of RGB quantization levels. Defaults to 256.

    Returns:
    - Colormap: The truncated colormap.

    Raises:
    - TypeError: If cmap is neither a string nor a Colormap.
    - ValueError: If start or end points are not within the valid range [0, 1]
        or if N is not a positive integer.

    Examples:
    >>> truncated_cmap = truncate('viridis', 0.2, 0.8, N=128)
    >>> plot_colormap(truncated_cmap)  # Custom function to visualize colormaps.

    References:
    - https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html
    """
    # Check args
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    elif not isinstance(cmap, Colormap):
        raise TypeError(f"Expect type Colormap or str for cmap, got {type(cmap)}.")

    if not 0 <= start < end <= 1:
        raise ValueError("Invalid Colormap start or end point.")

    if not isinstance(N, int) or N <= 0:
        raise ValueError(f"Invalid number of RGB quantization levels {N}.")

    return ListedColormap(cmap(np.linspace(start, end, N)))
