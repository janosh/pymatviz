"""Heatmap plotter."""


from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Colormap


def heatmap(
    df: pd.DataFrame,
    cmap: Colormap,
    add_cbar: bool = True,
    cbar_label: str = "Colorbar label",
    ax: plt.Axes = None,
    x_label: str = "X label",
    y_label: str = "Y label",
    **kwargs: Any,
) -> plt.Axes:
    """Create a heatmap from a 2D Pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data to visualize. The index is used as tick labels.
    - cmap (Colormap): Matplotlib colormap to be used for coloring the heatmap.
    - add_cbar (bool, optional): Whether to add a colorbar. Default is True.
    - cbar_label (str, optional): Label for the colorbar. Default is "Colorbar label".
    - ax (plt.Axes, optional): Matplotlib Axes on which to plot the heatmap. If not provided, the current Axes is used.
    - x_label (str, optional): Label for the x-axis. Default is "X label".
    - y_label (str, optional): Label for the y-axis. Default is "Y label".

    Returns:
    - plt.Axes: Matplotlib Axes containing the heatmap.

    Raises:
    - AssertionError: If input types or values are not as expected.

    Example:
    ```python
    # Create a sample DataFrame
    data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    # Plot the heatmap
    fig, ax = plt.subplots()
    heatmap(data_df, cmap, ax=ax, x_label='X-Axis', y_label='Y-Axis')
    ax.set_title('Heatmap Title')

    plt.show()
    ```
    """
    # Type and value checks
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Heatmap plotter expect a pd.DataFrame as input.")
    if not isinstance(cmap, Colormap):
        raise TypeError(
            f"Heatmap plotter expect a Colormap instance, got {type(cmap)}."
        )
    if not isinstance(add_cbar, bool):
        raise TypeError("add_cbar should be bool.")

    if df.ndim != 2:
        raise ValueError("Heatmap plotter expect a 2D DataFrame.")

    # Initialize ax for plotting
    ax = ax or plt.gca()

    # Generate heatmap
    im = plt.imshow(df, cmap)

    # Set x/y axes tick labels
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation="vertical")

    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)

    # Set x/y axes labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add colorbar
    if add_cbar:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    return ax
