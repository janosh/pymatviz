"""Generate custom colormaps."""


from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap


def combine_two(
    cmaps: list[Colormap | str],
    node: float = 0.5,
    N: int = 256,
    reverse: bool = False,
) -> Colormap:
    """Create a composite matplotlib Colormap by combining two colormaps.

    This function takes a list of two colormaps (or their names as strings)
    and creates a composite colormap by blending them together.

    Parameters:
    - cmaps (List[Colormap | str]): A list containing two colormaps or
        their names as strings.
    - node (float, optional): The blending point between the two colormaps,
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
    - The colormaps are combined from bottom to top by default.

    References:
    - https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html
    - https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps/31052741#31052741

    Examples:
    >>> combined_cmap = combine_two_colormaps(['viridis', 'plasma'],
            node=0.3, N=128, reverse=True)

    Todo:
    - Generalize to a list of Colormaps.
    """
    # Check cmap datatype and convert to List[Colormap]
    if all(isinstance(cmap, str) for cmap in cmaps):
        cmaps = [matplotlib.colormaps[s] for s in cmaps]
    elif any(not isinstance(cmap, Colormap) for cmap in cmaps):
        raise TypeError("Invalid datatype. Expect either all Colormaps or all strings.")

    # Sample two source colormaps
    cmap_0 = cmaps[0](np.linspace(0, 1, int(N * node)))
    cmap_1 = cmaps[1](np.linspace(0, 1, N - int(N * node)))

    # Merge Colormaps
    if reverse:
        return LinearSegmentedColormap.from_list(
            "composite_cmap_r", np.vstack((cmap_0, cmap_1))
        ).reversed()
    else:
        return LinearSegmentedColormap.from_list(
            "composite_cmap", np.vstack((cmap_0, cmap_1))
        )


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
    - start (float): The starting point of the truncated colormap,
        in range [0, 1].
    - end (float): The ending point of the truncated colormap,
        in range [0, 1].
    - N (int, optional): The number of RGB quantization levels.
        Defaults to 256.

    Returns:
    - Colormap: The truncated colormap.

    Raises:
    - TypeError: If cmap is neither a string nor a Colormap.
    - ValueError: If start or end points are not within the valid range [0, 1]
        or if N is not a positive integer.

    Examples:
    >>> truncated_cmap = truncate('viridis', 0.2, 0.8, N=128)

    References:
    - https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html
    """
    # Check args
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    elif not isinstance(cmap, Colormap):
        raise TypeError(
            f"Expect type Colormap or str for cmap, \
            got {type(cmap)}."
        )

    if not 0 <= start < end <= 1:
        raise ValueError("Invalid Colormap start or end point.")

    if not isinstance(N, int) or N <= 0:
        raise ValueError(f"Invalid number of RGB quantization levels {N}.")

    return ListedColormap(cmap(np.linspace(start, end, N)))
