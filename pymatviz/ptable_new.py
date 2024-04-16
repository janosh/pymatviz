"""Various periodic table heatmaps with matplotlib and plotly."""

from __future__ import annotations

from collections.abc import Sequence
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

# ----------------------------------------------------------
# This block might need to be relocated.
# Data types supported by ptable plotters
PTableSupportedDataType = (
    dict[str, float | Sequence[float] | np.ndarray] | pd.DataFrame | pd.Series
)


def _data_preprocessor(data: PTableSupportedDataType) -> pd.DataFrame:
    """Preprocess input data, including:
        - Convert all data types to pd.DataFrame.
        - Impute missing values.
        - Handle anomalies such as NaN, infinity.
        - Write vmin/vmax as metadata into the DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with element names as index
            and values as columns. The DataFrame also contains metadata for
            vmin and vmax.

    Example:
        >>> data: dict = {"H": 1.0, "He": [2.0, 4.0]}
        OR
        >>> data: pd.DataFrame = pd.DataFrame({"Element": ["H", "He"], "Value": [1.0, [2.0, 4.0]]})
        OR
        >>> data: pd.Series = pd.Series({"H": 1.0, "He": [2.0, 4.0]})
        TODO: wrong example: Series need same length values

        >>> preprocess_data(data)

        Element   Value
        H         1.0
        He        [2.0, 4.0]

        Metadata:
            vmin: 1.0
            vmax: 4.0
    """
    if isinstance(data, pd.DataFrame):
        df = data

    elif isinstance(data, pd.Series):
        df = data.to_frame()

    elif isinstance(data, dict):
        df = (
            pd.DataFrame(data)
            # .explode("Value")  # TODO: double-check
            .reset_index()
            .rename(columns={"index": "Element"})
        )

    else:
        raise TypeError(
            f"Unsupported data type, choose from: {PTableSupportedDataType}."
        )

    # Get and write vmin/vmax into metadata
    df.attrs["vmin"] = df["Value"].min()
    df.attrs["vmax"] = df["Value"].max()

    return df


# ----------------------------------------------------------


class PTableProjector:
    """Project (nest) a custom plot into a periodic table."""

    def __init__(
        self,
        colormap: str | Colormap,
        data: PTableSupportedDataType,  # TODO: narrow supported data type
        **kwargs: Any,
    ) -> None:
        """Initialize a ptable projector.

        Default figsize is set to (0.75 * n_groups, 0.75 * n_periods).

        Args:
            colormap (str | Colormap): The colormap to use.
            data (PTableSupportedDataType): The data to be visualized.
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

        TODO: add more details about data

        Parameters:
            data (Any): The data to be used.
        """
        # Preprocess data
        self._data = _data_preprocessor(data)

        # Normalize data for colorbar
        self._norm = Normalize(
            vmin=self._data.attrs.get("vmin"), vmax=self._data.attrs.get("vmax")
        )

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
