"""Various periodic table heatmaps with matplotlib and plotly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        return self._cmap

    @cmap.setter
    def cmap(self, colormap: str | Colormap) -> None:
        self._cmap = plt.get_cmap(colormap)

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, data: Any) -> None:
        # TODO: add data preprocessing function
        self._data = data

    def set_style(self) -> None:
        """Set global styles."""

    def add_child_plots(
        self,
        child_plotter: Callable,
        child_args: dict,
        on_empty: Literal["hide", "show"] = "hide",
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

            # Call child plotter for each tile
            if plot_data is not None:
                # Call child plotter
                child_plotter(ax, **child_args)

            # Hide axis boarders
            # TODO: Make this an arg too?
            for side in ("right", "top", "left", "bottom"):
                ax.spines[side].set_visible(b=False)

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
            # TODO: fix norm
            plt.cm.ScalarMappable(norm=norm, cmap=self.cmap),
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

        Parameters:
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

        # Hide axes
        ax.axis("off")

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

    plotter.add_child_plots(plot_split_rectangle, child_args)

    plotter.add_ele_symbols()
    plotter.add_colorbar(title="Test Colorbar")
