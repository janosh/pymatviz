"""Generate custom colormaps."""


from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap


def combine_two(
    cmaps: list[Colormap | str],
    node: float = 0.5,
    n_rgb_levels: int = 256,
    reverse: bool = False,
) -> Colormap:
    """Create a composite matplotlib Colormap by combining two color maps.

    This function takes a list of two color maps (or their names as strings)
    and creates a composite colormap by blending them together.

    Parameters:
    - cmaps (List[Colormap | str]): A list containing two color maps or
        their names as strings.
    - node (float, optional): The blending point between the two color maps,
        in range [0, 1]. Defaults to 0.5.
    - N (int, optional): The number of RGB quantization levels.
        Defaults to 256.
    - reverse (bool, optional): Reverse the resulting colormap.

    Returns:
    - Colormap: The composite colormap.

    Raises:
    - TypeError: If the list contains mixed datatypes or
        invalid colormap names.

    Note:
    - The color maps are combined from bottom to top by default.

    References:
    - https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html
    - https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps/31052741#31052741

    Examples:
    >>> combined_cmap = combine_two_colormaps(['viridis', 'plasma'],
            node=0.3, N=128, reverse=True)

    Todo:
    - Generalize to a list of color maps.
    """
    # Check cmap datatype and convert to List[Colormap]
    if all(isinstance(cmap, str) for cmap in cmaps):
        cmaps = [matplotlib.colormaps[s] for s in cmaps]
    elif any(not isinstance(cmap, Colormap) for cmap in cmaps):
        raise TypeError(
            "Invalid datatype. Expect either all color maps or all strings."
        )

    # Sample two source colormaps
    cmap_0 = cmaps[0](np.linspace(0, 1, int(n_rgb_levels * node)))
    cmap_1 = cmaps[1](np.linspace(0, 1, n_rgb_levels - int(n_rgb_levels * node)))

    # Merge color maps
    if reverse:
        return LinearSegmentedColormap.from_list(
            "composite_cmap_r", np.vstack((cmap_0, cmap_1))
        ).reversed()
    return LinearSegmentedColormap.from_list(
        "composite_cmap", np.vstack((cmap_0, cmap_1))
    )
