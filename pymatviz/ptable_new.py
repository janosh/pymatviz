"""Various periodic table heatmaps with matplotlib and plotly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from pymatgen.core import Element

from pymatviz.utils import df_ptable


if TYPE_CHECKING:
    from typing import Any, Callable, Literal

    from matplotlib.colors import Colormap


class PTableProjector:
    """Project (nest) a custom plot into a periodic table."""

    def __init__(
        self,
        colormap: str | Colormap,
        data: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize a ptable projector.

        Default figsize is set to (0.75 * n_groups, 0.75 * n_periods).

        Args:
            colormap (str | Colormap): The colormap to use.
            data (Any): The data to be visualized.
            **kwargs (Any): Additional keyword arguments to
                be passed to the matplotlib subplots function.
        """
        # Get colormap
        self.cmap = colormap

        # Preprocess data
        self.data = data

        # Initialize periodic table canvas
        n_periods = df_ptable.row.max()
        n_groups = df_ptable.column.max()

        # Set figure size
        kwargs.setdefault("figsize", (0.75 * n_groups, 0.75 * n_periods))

        self.fig, self.axes = plt.subplots(n_periods, n_groups, **kwargs)

    @property
    def cmap(self) -> Colormap:
        """The global Colormap used.

        Returns:
            Colormap: The Colormap used.
        """
        return self._cmap

    @cmap.setter
    def cmap(self, colormap: str | Colormap) -> None:
        """Args:
        colormap (str | Colormap): The colormap to use.
        """
        self._cmap = plt.get_cmap(colormap)

    @property
    def data(self) -> pd.DataFrame:
        """The preprocessed data.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        return self._data

    @data.setter
    def data(self, data: Any) -> None:
        """Set and preprocess the data.

        TODO: add more details

        Parameters:
            data (Any): The data to be used.
        """
        # TODO: add data preprocessing function
        self._data = data

        # Get vmin/vmax from metadata  # TODO
        vmin = "TODO"
        vmax = "TODO"

        # Normalize data for colorbar
        self._norm = Normalize(vmin=vmin, vmax=vmax)

    def set_style(self) -> None:
        """Set global styles.

        Todo:
        """

    def add_child_plots(
        self,
        child_plotter: Callable,
        child_args: dict,
        on_empty: Literal["hide", "show"] = "hide",
        hide_axis: bool = True,
    ) -> None:
        """Add selected custom child plots."""
        for element in Element:
            # Get axis by element symbol
            symbol = element.symbol
            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax = self.axes[row - 1][column - 1]

            # Check tile data
            plot_data = np.array(self.data.get(symbol, []))
            if len(plot_data) == 0 and on_empty == "hide":
                continue

            # Call child plotter
            if plot_data is not None:
                child_plotter(ax, **child_args)

            # Hide axis
            if hide_axis:
                ax.axis("off")

    def add_ele_symbols(
        self,
        text: Callable[[Element], str] = lambda elem: elem.symbol,
        pos: tuple[float, float] = (0.5, 0.5),
        kwargs: dict | None = None,
    ) -> None:
        """Add element symbols for each tile."""
        kwargs = kwargs or {}
        kwargs.setdefault("fontsize", 18)

        # Add symbol for each element
        for element in Element:
            # Get axes by element symbol
            symbol = element.symbol
            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax = self.axes[row - 1][column - 1]

            ax.text(
                *pos,
                text(element) if callable(text) else text.format(elem=element),
                ha="center",
                va="center",
                transform=ax.transAxes,
                **kwargs,
            )

    def add_colorbar(
        self,
        title: str,
        coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.02),
        cbar_kwds: dict | None = None,
        title_kwds: dict | None = None,
    ) -> None:
        """Add a global colorbar."""
        # Update colorbar keyword args
        cbar_kwds = {"orientation": "horizontal"} | (cbar_kwds or {})

        # Add colorbar
        cbar_ax = self.fig.add_axes(coords)

        self.fig.colorbar(
            plt.cm.ScalarMappable(norm=self._norm, cmap=self.cmap),
            cax=cbar_ax,
            **cbar_kwds,
        )

        # Set colorbar title
        title_kwds = title_kwds or {}
        title_kwds.setdefault("fontsize", 12)
        title_kwds.setdefault("pad", 10)
        title_kwds["label"] = title

        cbar_ax.set_title(**title_kwds)


# Test area
if __name__ == "__main__":
    # Def test child plotter
    from matplotlib.patches import Rectangle

    def plot_split_rectangle(
        ax: plt.axes,
        colors: list[tuple[float, float, float]],
        start_angle: float,
    ) -> None:
        """Helper function to plot an evenly-split rectangle.

        Args:
            colors (list): A list of colors to fill each split of the rectangle.
            start_angle (float): The starting angle for the splits in degrees,
                and the split proceeds counter-clockwise (0 refers to the x-axis).
        """
        # Plot the pie chart
        ax.pie(
            np.ones(len(colors)),
            colors=colors,
            startangle=start_angle,
            wedgeprops=dict(clip_on=True),
        )

        # Crop a central rectangle from the pie chart
        rect = Rectangle((-0.5, -0.5), 1, 1, fc="none", ec="none")
        ax.set_clip_path(rect)

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)

    # Generate test data
    import random

    test_data = {
        elem.symbol: [
            random.randint(0, 10),
            random.randint(10, 20),
        ]
        for elem in Element
    }

    # Test projector
    plotter = PTableProjector(
        colormap="coolwarm",
        data=test_data,
    )

    # Call child plotter
    child_args = {"start_angle": 135}

    # TODO: need "colors" mapper

    plotter.add_child_plots(plot_split_rectangle, child_args)

    plotter.add_ele_symbols()
    plotter.add_colorbar(title="Test Colorbar")
