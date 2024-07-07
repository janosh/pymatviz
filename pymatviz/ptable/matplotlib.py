"""Periodic table plots powered by matplotlib."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatter, LogLocator
from matplotlib.typing import ColorType
from pymatgen.core import Element

from pymatviz.colors import ELEM_COLORS_JMOL, ELEM_COLORS_VESTA, ELEM_TYPE_COLORS
from pymatviz.enums import ElemColorMode, ElemColors, ElemCountMode, Key
from pymatviz.process_data import count_elements
from pymatviz.ptable._process_data import (
    PTableData,
    SupportedDataType,
    SupportedValueType,
)
from pymatviz.utils import (
    ElemValues,
    df_ptable,
    get_cbar_label_formatter,
    pick_bw_for_contrast,
)


if TYPE_CHECKING:
    from typing import Any, Callable


class PTableProjector:
    """Project (nest) a custom plot into a periodic table.

    Attributes:
        cmap (Colormap | None): Colormap.
        data (pd.DataFrame): Data for plotting.
        ptable_data (PTableData): Internal data container.
        norm (Normalize): Data min-max normalizer.
        hide_f_block (bool): Whether to hide f-block.
        elem_types (set[str]): Types of elements present.
        elem_type_colors (dict[str, str]): Element typed based colors.
        elem_colors (dict): Element based colors.

    Scopes mentioned in this plotter:
        plot: Refers to the global Figure.
        axis: Refers to the Axis where child plotter would plot.
        child: Refers to the child plotter, for example, ax.plot().
    """

    def __init__(
        self,
        *,
        data: SupportedDataType | PTableData,
        log: bool = False,
        colormap: str | Colormap = "viridis",
        tile_size: tuple[float, float] = (0.75, 0.75),
        plot_kwargs: dict[str, Any] | None = None,
        exclude_elements: Sequence[str] = (),
        on_empty: Literal["hide", "show"] = "hide",
        hide_f_block: bool | Literal["AUTO"] = "AUTO",
        elem_type_colors: dict[str, str] | None = None,
        elem_colors: ElemColors | dict[str, ColorType] = ElemColors.vesta,
    ) -> None:
        """Initialize a ptable projector.

        Default figsize is set to (0.75 * n_groups, 0.75 * n_periods),
        and axes would be turned off by default.

        Args:
            data (SupportedDataType | PTableData): The data to be visualized.
            log (bool): Whether to show colorbar in log scale.
            colormap (str | Colormap): The colormap to use. Defaults to "viridis".
            tile_size (tuple[float, float]): The relative tile height and width.
            plot_kwargs (dict): Additional keyword arguments to
                pass to the plt.subplots function call.
            exclude_elements (Sequence[str]): Elements to exclude.
            on_empty ("hide" | "show"): Hide or show tile if no data provided.
            hide_f_block (bool | "AUTO"): Hide f-block (Lanthanum and Actinium series).
                Defaults to "AUTO", meaning hide if no data present.
            elem_type_colors (dict | None): Element typed based colors.
            elem_colors (dict | ElemColors): Element-specific colors.
        """
        # Set colors
        self.cmap: Colormap = colormap
        self._elem_type_colors = ELEM_TYPE_COLORS | (elem_type_colors or {})
        self.elem_colors = elem_colors  # type: ignore[assignment]

        # Preprocess data
        self.log = log
        self.data: pd.DataFrame = data

        # Remove excluded element from internal data to avoid metadata pollution
        self.ptable_data.drop_elements(exclude_elements)

        self.on_empty = on_empty
        self.hide_f_block = hide_f_block  # type: ignore[assignment]

        # Initialize periodic table canvas
        n_periods = df_ptable.row.max()
        n_groups = df_ptable.column.max()

        if self.hide_f_block:
            n_periods -= 3

        # Set figure size
        plot_kwargs = plot_kwargs or {}
        height, width = tile_size
        plot_kwargs.setdefault("figsize", (height * n_groups, width * n_periods))

        self.fig, self.axes = plt.subplots(n_periods, n_groups, **plot_kwargs)

        # Turn off all axes
        for ax in self.axes.flat:
            ax.axis("off")

    @property
    def cmap(self) -> Colormap | None:
        """The periodic table's Colormap."""
        return self._cmap

    @cmap.setter
    def cmap(self, colormap: str | Colormap | None) -> None:
        """Set the periodic table's Colormap."""
        self._cmap = None if colormap is None else plt.get_cmap(colormap)

    @property
    def data(self) -> pd.DataFrame:
        """The preprocessed data."""
        return self.ptable_data.data

    @data.setter
    def data(self, data: SupportedDataType) -> None:
        """Preprocess and set the data. Also set normalizer."""
        if not isinstance(data, PTableData):
            data = PTableData(data)

        self.ptable_data = data

        # Update norm when data is updated
        self.norm = (None, None)

    @property
    def norm(self) -> Normalize | LogNorm:
        """Data normalizer."""
        return self._norm

    @norm.setter
    def norm(self, value_range: tuple[float | None, float | None]) -> None:
        """Set normalizer by value range.

        Args:
            value_range (tuple[float | None, float | None]): The upper and
                lower bound of values, use None for auto detect from metadata.
        """
        value_range = (
            value_range[0] or self.ptable_data.data.attrs["vmin"],
            value_range[1] or self.ptable_data.data.attrs["vmax"],
        )
        self._norm = LogNorm(*value_range) if self.log else Normalize(*value_range)

    @property
    def anomalies(self) -> dict[str, set[Literal["nan", "inf"]]] | Literal["NA"]:
        """Element symbol to anomalies ("nan/inf") mapping."""
        return self.ptable_data.anomalies

    @property
    def hide_f_block(self) -> bool:
        """Whether to hide f-block in plots."""
        return self._hide_f_block

    @hide_f_block.setter
    def hide_f_block(self, hide_f_block: bool | Literal["AUTO"]) -> None:
        """If hide_f_block is "AUTO", would detect if data is present."""
        if hide_f_block == "AUTO":
            f_block_elements_has_data = {
                atom_num
                for atom_num in [*range(57, 72), *range(89, 104)]  # rare earths
                if (elem := Element.from_Z(atom_num).symbol) in self.data.index
                and len(self.data.loc[elem, Key.heat_val]) > 0
            }

            self._hide_f_block = not bool(f_block_elements_has_data)

        else:
            self._hide_f_block = hide_f_block

    @property
    def elem_types(self) -> set[str]:
        """Set of element types present in data."""
        return set(df_ptable.loc[self.data.index, "type"])

    @property
    def elem_type_colors(self) -> dict[str, str]:
        """Element type based colors mapping.

        Example:
            dict(Nonmetal="green", Halogen="teal", Metal="lightblue")
        """
        return self._elem_type_colors or {}

    @elem_type_colors.setter
    def elem_type_colors(self, elem_type_colors: dict[str, str]) -> None:
        self._elem_type_colors |= elem_type_colors or {}

    @property
    def elem_colors(self) -> dict[str, ColorType]:
        """Element-based colors."""
        return self._elem_colors

    @elem_colors.setter
    def elem_colors(
        self,
        elem_colors: ElemColors | dict[str, ColorType] = ElemColors.vesta,
    ) -> None:
        """Set elem_colors.

        Args:
            elem_colors ("vesta" | "jmol" | dict[str, ColorType]): Use VESTA
                or Jmol color mapping, or a custom {"element": Color} mapping.
                Defaults to "vesta".
        """
        if elem_colors == "vesta":
            self._elem_colors = ELEM_COLORS_VESTA
        elif elem_colors == "jmol":
            self._elem_colors = ELEM_COLORS_JMOL
        elif isinstance(elem_colors, dict):
            self._elem_colors = elem_colors
        else:
            raise ValueError(
                f"elem_colors must be 'vesta', 'jmol', or a custom dict, "
                f"got {elem_colors=}"
            )

    def get_elem_type_color(
        self,
        elem_symbol: str,
        default: ColorType = "white",
    ) -> ColorType:
        """Get element type color by element symbol.

        Args:
            elem_symbol (str): The element symbol.
            default (str): Default color if not found in mapping.
        """
        elem_type = df_ptable.loc[elem_symbol].get("type", None)
        return self.elem_type_colors.get(elem_type, default)

    def add_child_plots(
        self,
        child_plotter: Callable[..., None],
        *,
        child_kwargs: dict[str, Any] | None = None,
        tick_kwargs: dict[str, Any] | None = None,
        ax_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Add custom child plots to the periodic table grid.

        Args:
            child_plotter (Callable): The child plotter.
            child_kwargs (dict): Arguments to pass to the child plotter call.
            tick_kwargs (dict): Keyword arguments to pass to ax.tick_params().
            ax_kwargs (dict): Keyword arguments to pass to ax.set().
        """
        # Update kwargs
        child_kwargs = child_kwargs or {}
        ax_kwargs = ax_kwargs or {}
        tick_kwargs = {"axis": "both", "which": "major", "labelsize": 8} | (
            tick_kwargs or {}
        )

        for element in Element:
            # Hide f-block
            if self.hide_f_block and (element.is_lanthanoid or element.is_actinoid):
                continue

            # Get axis index by element symbol
            symbol: str = element.symbol
            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax: plt.Axes = self.axes[row - 1][column - 1]

            # Get and check tile data
            try:
                plot_data: np.ndarray | Sequence[float] = self.data.loc[
                    symbol, Key.heat_val
                ]
            except KeyError:  # skip element without data
                plot_data = None

            if (plot_data is None or len(plot_data) == 0) and self.on_empty == "hide":
                continue

            # Call child plotter
            child_plotter(ax, plot_data, tick_kwargs=tick_kwargs, **child_kwargs)

            # Pass axis kwargs
            if ax_kwargs:
                ax.set(**ax_kwargs)

    def add_elem_symbols(
        self,
        text: str | Callable[[Element], str] = lambda elem: elem.symbol,
        *,
        pos: tuple[float, float] = (0.5, 0.5),
        text_color: ColorType
        | dict[str, ColorType]
        | Literal[ElemColorMode.element_types] = "black",
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Add element symbols for each tile.

        Args:
            text (str | Callable): The text to add to the tile.
                If a callable, it should accept a pymatgen Element and return a
                string. If a string, it can contain a format
                specifier for an `elem` variable which will be replaced by the element.
            pos (tuple): The position of the text relative to the axes.
            text_color (ColorType | dict[str, ColorType]): The color of the text.
                Defaults to "black". Could take the following type:
                    - ColorType: The same color for all elements.
                    - dict[str, ColorType]: An element to color mapping.
                    - "element-types": Use color from self.elem_type_colors.
            kwargs (dict): Additional keyword arguments to pass to the `ax.text`.
        """
        # Update symbol kwargs
        kwargs = kwargs or {}
        kwargs.setdefault("fontsize", 12)

        # Add symbol for each element
        for element in Element:
            # Hide f-block
            if self.hide_f_block and (element.is_lanthanoid or element.is_actinoid):
                continue

            # Get axis index by element symbol
            symbol: str = element.symbol
            if symbol not in self.data.index and self.on_empty == "hide":
                continue

            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax: plt.Axes = self.axes[row - 1][column - 1]

            content = text(element) if callable(text) else text.format(elem=element)

            # Generate symbol text color
            if text_color == ElemColorMode.element_types:
                symbol_color = self.get_elem_type_color(symbol, "black")
            elif isinstance(text_color, dict):
                symbol_color = text_color.get(symbol, "black")
            else:
                symbol_color = text_color

            ax.text(
                *pos,
                content,
                color=symbol_color,
                ha="center",
                va="center",
                transform=ax.transAxes,
                **kwargs,
            )

    def add_colorbar(
        self,
        title: str,
        *,
        coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.02),
        cbar_range: tuple[float | None, float | None] = (None, None),
        cbar_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Add colorbar.

        Args:
            title (str): Title for the colorbar.
            coords (tuple): Coordinates of the colorbar (left, bottom, width, height).
                Defaults to (0.18, 0.8, 0.42, 0.02).
            cbar_range (tuple): Colorbar values display range, use None for auto
                detection for the corresponding boundary.
            cbar_kwargs (dict): Additional keyword arguments to pass to fig.colorbar().
            title_kwargs (dict): Additional keyword arguments for the colorbar title.
        """
        # Update colorbar kwargs
        cbar_kwargs = {"orientation": "horizontal"} | (cbar_kwargs or {})
        title_kwargs = {"fontsize": 12, "pad": 10, "label": title} | (
            title_kwargs or {}
        )

        # Format for log scale
        if self.log:
            cbar_kwargs |= {
                "norm": "log",
                "format": LogFormatter(10),
                "ticks": LogLocator(base=10),
            }

        # Update colorbar range
        self.norm = cbar_range

        # Check colormap
        if self.cmap is None:
            raise ValueError("Cannot add colorbar without a colormap.")

        # Add colorbar
        cbar_ax = self.fig.add_axes(coords)

        self.fig.colorbar(
            plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
            cax=cbar_ax,
            **cbar_kwargs,
        )

        # Set colorbar title
        cbar_ax.set_title(**title_kwargs)

    def add_elem_type_legend(
        self,
        kwargs: dict[str, Any] | None,
    ) -> None:
        """Add a legend to show the colors based on element types.

        Args:
            kwargs (dict | None): Keyword arguments passed to
                plt.legend() for customizing legend appearance.
        """
        kwargs = kwargs or {}

        font_size = 10

        # Get present elements
        legend_elements = [
            plt.Line2D(
                *([0], [0]),
                marker="s",
                color="w",
                label=elem_class,
                markerfacecolor=color,
                markersize=1.2 * font_size,
            )
            for elem_class, color in self.elem_type_colors.items()
            if elem_class in self.elem_types
        ]

        default_legend_kwargs = dict(
            loc="center left",
            bbox_to_anchor=(0, -42),
            ncol=6,
            frameon=False,
            fontsize=font_size,
            handlelength=1,  # more compact legend
        )
        kwargs = default_legend_kwargs | kwargs

        plt.legend(handles=legend_elements, **kwargs)

    def set_elem_background_color(self, alpha: float = 0.1) -> None:
        """Set element tile background color by element type.

        Args:
            alpha (float): Transparency.
        """
        for element in Element:
            # Hide f-block
            if self.hide_f_block and (element.is_lanthanoid or element.is_actinoid):
                continue

            # Get element axis
            symbol = element.symbol
            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax: plt.Axes = self.axes[row - 1][column - 1]

            # Set background color by element type
            rgb = mpl.colors.to_rgb(self.get_elem_type_color(symbol))
            ax.set_facecolor((*rgb, alpha))


class ChildPlotters:
    """Child plotters for PTableProjector to make different types
    (heatmap/line/scatter/histogram) for individual element tiles.
    """

    @staticmethod
    def rectangle(
        ax: plt.axes,
        data: SupportedValueType,
        norm: Normalize,
        cmap: Colormap,
        start_angle: float,
        tick_kwargs: dict[str, Any],  # noqa: ARG004
    ) -> None:
        """Rectangle heatmap plotter. Could be evenly split
        depending on the length of the data (could mix and match).

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for the child plotter.
            norm (Normalize): Normalizer for data-color mapping.
            cmap (Colormap): Colormap used for value mapping.
            start_angle (float): The starting angle for the splits in degrees,
                and the split proceeds counter-clockwise (0 refers to the x-axis).
            tick_kwargs: For compatibility with other plotters.
        """
        # Map values to colors
        if isinstance(data, (Sequence, np.ndarray)):
            colors = [cmap(norm(value)) for value in data]
        else:
            raise TypeError("Unsupported data type.")

        # Add the pie chart
        ax.pie(
            np.ones(len(colors)),
            colors=colors,
            startangle=start_angle,
            wedgeprops={"clip_on": True},
        )

        # Crop the central rectangle from the pie chart
        rect = Rectangle(xy=(-0.5, -0.5), width=1, height=1, fc="none", ec="none")
        ax.set_clip_path(rect)

        ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))

    @staticmethod
    def scatter(
        ax: plt.axes,
        data: SupportedValueType,
        norm: Normalize,
        cmap: Colormap,
        tick_kwargs: dict[str, Any],
        **child_kwargs: Any,
    ) -> None:
        """Scatter plotter.

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for the child plotter.
            norm (Normalize): Normalizer for data-color mapping.
            cmap (Colormap): Colormap.
            child_kwargs (dict): kwargs to pass to the child plotter call.
            tick_kwargs (dict): kwargs to pass to ax.tick_params().
        """
        # Add scatter
        if len(data) == 2:
            ax.scatter(x=data[0], y=data[1], **child_kwargs)
        elif len(data) == 3:
            ax.scatter(x=data[0], y=data[1], c=cmap(norm(data[2])), **child_kwargs)

        # Set tick labels
        ax.tick_params(**tick_kwargs)

        # Hide the right and top spines
        ax.axis("on")  # turned off by default
        ax.spines[["right", "top"]].set_visible(False)

    @staticmethod
    def line(
        ax: plt.axes,
        data: SupportedValueType,
        tick_kwargs: dict[str, Any],
        **child_kwargs: Any,
    ) -> None:
        """Line plotter.

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for the child plotter.
            child_kwargs (dict): kwargs to pass to the child plotter call.
            tick_kwargs (dict): kwargs to pass to ax.tick_params().
        """
        # Add line
        ax.plot(data[0], data[1], **child_kwargs)

        # Set tick labels
        ax.tick_params(**tick_kwargs)

        # Hide the right and top spines
        ax.axis("on")  # turned off by default
        ax.spines[["right", "top"]].set_visible(False)

    @staticmethod
    def histogram(
        ax: plt.axes,
        data: SupportedValueType,
        cmap: Colormap,
        cbar_axis: Literal["x", "y"],
        tick_kwargs: dict[str, Any],
        **child_kwargs: Any,
    ) -> None:
        """Histogram plotter.

        Taken from https://stackoverflow.com/questions/23061657/
        plot-histogram-with-colors-taken-from-colormap

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for the child plotter.
            cmap (Colormap): Colormap.
            cbar_axis ("x" | "y"): The axis colormap would be based on.
            child_kwargs: kwargs to pass to the child plotter call.
            tick_kwargs (dict): kwargs to pass to ax.tick_params().
        """
        # Preprocess x_range if only one boundary is given
        x_range = child_kwargs.pop("range", None)

        if x_range is None or x_range == [None, None]:
            x_range = [np.min(data), np.max(data)]

        elif x_range[0] is None:
            x_range = [np.min(data), x_range[1]]

        else:
            x_range = [x_range[0], np.max(data)]

        # Add histogram
        n, bins, patches = ax.hist(data, range=x_range, **child_kwargs)

        # Scale values according to axis
        if cbar_axis == "x":
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            cols = bin_centers - min(bin_centers)
            cols /= max(cols)

        else:
            cols = (n - n.min()) / (n.max() - n.min())

        # Apply colors
        for col, patch in zip(cols, patches):
            plt.setp(patch, "facecolor", cmap(col))

        # Set tick labels
        ax.tick_params(**tick_kwargs)
        ax.set(yticklabels=(), yticks=())
        # Set x-ticks to min/max only
        ax.set(xticks=[math.floor(x_range[0]), math.ceil(x_range[1])])

        # Hide the right, left and top spines
        ax.axis("on")
        ax.spines[["right", "top", "left"]].set_visible(False)


class HMapPTableProjector(PTableProjector):
    """With more heatmap-specific functionalities."""

    def __init__(
        self,
        *,
        values_show_mode: Literal["value", "fraction", "percent", "off"] = "value",
        sci_notation: bool = False,
        tile_colors: dict[str, ColorType] | Literal["AUTO"] = "AUTO",
        text_colors: dict[str, ColorType] | ColorType | Literal["AUTO"] = "AUTO",
        **kwargs: dict[str, Any],
    ) -> None:
        """Init Heatmap plotter.

        Args:
            values_show_mode ("value" | "fraction" | "percent" | "off"):
                Values display mode.
            sci_notation (bool): Whether to use scientific notation for
                values and colorbar tick labels.
            tile_colors (dict[str, ColorType] | "AUTO"): Tile colors.
                Defaults to "AUTO" for auto generation.
            text_colors: Colors for element symbols and values.
                - "AUTO": Auto pick "black" or "white" based on the contrast
                    of tile color for each element.
                - ColorType: Use the same ColorType for each element.
                - dict[str, ColorType]: Element to color mapping.
            kwargs (dict): Kwargs to pass to super class.
        """
        super().__init__(**kwargs)  # type: ignore[arg-type]

        self.sci_notation = sci_notation
        self.values_show_mode = values_show_mode

        # Normalize data for "fraction/percent" modes
        if values_show_mode in {"fraction", "percent"}:
            normalized_data = self.ptable_data
            normalized_data.normalize()
            self.data = normalized_data.data  # call setter to update metadata

        # Generate tile colors
        self.tile_colors = tile_colors  # type: ignore[assignment]
        # Auto generate tile values
        self.tile_values = None  # type: ignore[assignment]

        # Generate element symbols colors
        self.text_colors = text_colors  # type: ignore[assignment]

    @property
    def tile_colors(self) -> dict[str, ColorType]:
        """The final element symbol to color mapping."""
        return self._tile_colors

    @tile_colors.setter
    def tile_colors(
        self,
        tile_colors: dict[str, ColorType] | Literal["AUTO"] = "AUTO",
    ) -> None:
        """An element symbol to color mapping, and apply overwrite colors.

        Args:
            tile_colors (dict[str, ColorType] | "AUTO"): Tile colors.
                Defaults to "AUTO" for auto generation.
        """
        # Generate tile colors from values if not given
        self._tile_colors = {} if tile_colors == "AUTO" else tile_colors

        if tile_colors == "AUTO":
            if self.cmap is None:
                raise ValueError("Cannot generate tile colors without colormap.")

            for symbol in self.data.index:
                # Get value and map to color
                value = self.data.loc[symbol, Key.heat_val][0]
                self._tile_colors[symbol] = self.cmap(self.norm(value))

    @property
    def tile_values(self) -> dict[str, str]:
        """Displayed values for each tile."""
        return self._tile_values

    @tile_values.setter
    def tile_values(self, tile_values: dict[str, str] | None) -> None:
        # Generate tile values from PTableData if not provided
        if tile_values is None:
            tile_values = {
                elem: self.data.loc[elem, Key.heat_val][0] for elem in self.data.index
            }

        self._tile_values = tile_values

    @property
    def text_colors(self) -> dict[str, ColorType]:
        """Element to text (symbol and value) color mapping."""
        return self._text_colors

    @text_colors.setter
    def text_colors(
        self,
        text_colors: dict[str, ColorType]
        | ColorType
        | Literal["AUTO", ElemColorMode.element_types],
    ) -> None:
        """Generate and set symbol to text colors mapping.

        Args:
            text_colors: Colors for element symbols and values.
                - "AUTO": Auto pick "black" or "white" based on the contrast
                    of tile color for each element.
                - ColorType: Use the same ColorType for each element.
                - dict[str, ColorType]: Element to color mapping.
        """
        if text_colors == "AUTO":
            text_colors = {
                symbol: pick_bw_for_contrast(self.tile_colors[symbol])
                for symbol in self.data.index
            }

        elif text_colors == ElemColorMode.element_types:
            text_colors = {
                symbol: self.get_elem_type_color(symbol, default="black")
                for symbol in self.data.index
            }

        elif not isinstance(text_colors, dict):
            text_colors = {symbol: text_colors for symbol in self.data.index}

        self._text_colors = cast(dict[str, ColorType], text_colors)

    def add_child_plots(  # type: ignore[override]
        self,
        *,
        f_block_voffset: float = 0,  # noqa: ARG002 TODO: fix this
        tick_kwargs: dict[str, Any] | None = None,
        ax_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Add custom child plots to the periodic table grid.

        Args:
            f_block_voffset (float): The vertical offset of f-block elements.
            tick_kwargs (dict): Keyword arguments to pass to ax.tick_params().
            ax_kwargs (dict): Keyword arguments to pass to ax.set().
            on_empty ("hide" | "show"): Whether to show or hide tiles for
                elements without data.
        """
        # Update kwargs
        ax_kwargs = ax_kwargs or {}
        tick_kwargs = {"axis": "both", "which": "major", "labelsize": 8} | (
            tick_kwargs or {}
        )

        for element in Element:
            # Hide f-block
            if self.hide_f_block and (element.is_lanthanoid or element.is_actinoid):
                continue

            # Get axis index by element symbol
            symbol: str = element.symbol
            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax: plt.Axes = self.axes[row - 1][column - 1]

            # Get and check tile data
            try:
                plot_data: np.ndarray | Sequence[float] = self.data.loc[
                    symbol, Key.heat_val
                ]

            except KeyError:
                plot_data = None

            # Skip element without data if "hide"
            if (plot_data is None or len(plot_data) == 0) and self.on_empty == "hide":
                continue

            # Add child heatmap plot
            ax.pie(
                np.ones(1),
                colors=[self.tile_colors.get(symbol, "lightgray")],
                wedgeprops={"clip_on": True},
            )

            # Crop the central rectangle from the pie chart
            rect = Rectangle(
                xy=(-0.5, -0.5), width=1, height=1, fc="none", ec="grey", lw=2
            )
            ax.add_patch(rect)

            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)

            # Pass axis kwargs
            if ax_kwargs:
                ax.set(**ax_kwargs)

    def add_elem_values(
        self,
        *,
        text_fmt: str,
        pos: tuple[float, float] = (0.5, 0.25),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Format and show element values.

        Args:
            text_fmt (str): f-string format for the value text.
            pos (tuple[float, float]): Position of the value in the tile.
            kwargs (dict): Additional keyword arguments to pass to the `ax.text`.
        """
        # Update symbol kwargs
        kwargs = kwargs or {}
        if self.sci_notation:
            kwargs.setdefault("fontsize", 10)
        else:
            kwargs.setdefault("fontsize", 12)

        # Add value for each element
        for element in Element:
            # Hide f-block
            if self.hide_f_block and (element.is_lanthanoid or element.is_actinoid):
                continue

            # Get axis index by element symbol
            symbol: str = element.symbol
            if symbol not in self.data.index and self.on_empty == "hide":
                continue

            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax: plt.Axes = self.axes[row - 1][column - 1]

            # Get and format value
            value = self.tile_values.get(symbol, "-")
            try:
                value = f"{value:{text_fmt}}"
            except ValueError:
                pass

            # Simplify scientific notation, e.g. 1e-01 to 1e-1
            if self.sci_notation and ("e-0" in value or "e+0" in value):
                value = value.replace("e-0", "e-").replace("e+0", "e+")

            ax.text(
                *pos,
                value,
                color=self.text_colors.get(symbol, "black"),
                ha="center",
                va="center",
                transform=ax.transAxes,
                **kwargs,
            )


def ptable_heatmap(
    data: pd.DataFrame | pd.Series | dict[str, list[list[float]]] | PTableData,
    *,
    # Heatmap specific
    colormap: str = "viridis",
    exclude_elements: Sequence[str] = (),
    inf_color: ColorType = "lightskyblue",
    nan_color: ColorType = "white",
    log: bool = False,
    sci_notation: bool = False,
    tile_size: tuple[float, float] = (0.75, 0.75),  # TODO: WIP, don't use
    # Figure-scope
    on_empty: Literal["hide", "show"] = "hide",
    hide_f_block: bool | Literal["AUTO"] = "AUTO",
    f_block_voffset: float = 0,  # TODO: WIP, don't use
    plot_kwargs: dict[str, Any] | None = None,
    # Axis-scope
    ax_kwargs: dict[str, Any] | None = None,
    text_colors: ColorType = "AUTO",
    # Symbol
    symbol_text: str | Callable[[Element], str] = lambda elem: elem.symbol,
    symbol_pos: tuple[float, float] | None = None,
    symbol_kwargs: dict[str, Any] | None = None,
    # Values
    values_show_mode: Literal["value", "fraction", "percent", "off"] = "value",
    values_pos: tuple[float, float] | None = None,
    values_fmt: str = "AUTO",
    values_kwargs: dict[str, Any] | None = None,
    # Colorbar
    show_cbar: bool = True,
    cbar_coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.05),
    cbar_range: tuple[float | None, float | None] = (None, None),
    cbar_label_fmt: str = "AUTO",
    cbar_title: str = "Element Count",
    cbar_title_kwargs: dict[str, Any] | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
) -> plt.Figure:
    """Plot a heatmap across the periodic table.

    Args:
        data (pd.DataFrame | pd.Series | dict[str, list[list[float]]]):
            Map from element symbols to plot data. E.g. if dict,
            {"Fe": [1, 2], "Co": [3, 4]}, where the 1st value would
            be plotted on the lower-left corner and the 2nd on the upper-right.
            If pd.Series, index is element symbols and values lists.
            If pd.DataFrame, column names are element symbols,
            plots are created from each column.

        # Heatmap specific
        colormap (str): The colormap to use.
        exclude_elements (Sequence[str]): Elements to exclude.
        inf_color (ColorType): The color to use for infinity.
        nan_color (ColorType): The color to use for missing value (NaN).
        log (bool): Whether to show colorbar in log scale.
        sci_notation (bool): Whether to use scientific notation for values and
            colorbar tick labels.
        tile_size (tuple[float, float]): The relative height and width of the tile.

        # Figure-scope
        on_empty ("hide" | "show"): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool | "AUTO"): Hide f-block (Lanthanum and Actinium series).
            Defaults to "AUTO", meaning hide if no data is provided.
        f_block_voffset (float): The vertical offset of f-block elements.
        plot_kwargs (dict): Additional keyword arguments to
            pass to the plt.subplots function call.

        # Axis-scope
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options.
        text_colors: Colors for element symbols and values.
                - "AUTO": Auto pick "black" or "white" based on the contrast
                    of tile color for each element.
                - ColorType: Use the same ColorType for each element.
                - dict[str, ColorType]: Element to color mapping.

        # Symbol
        symbol_text (str | Callable[[Element], str]): Text to display for
            each element symbol. Defaults to lambda elem: elem.symbol.
        symbol_pos (tuple[float, float]): Position of element symbols
            relative to the lower left corner of each tile.
            Defaults to (0.5, 0.5). (1, 1) is the upper right corner.
        symbol_kwargs (dict): Keyword arguments passed to plt.text() for
            element symbols. Defaults to None.

        # Values
        values_show_mode (str): The values display mode:
            - "off": Hide values.
            - "value": Display values as is.
            - "fraction": As a fraction of the total (0.10).
            - "percent": As a percentage of the total (10%).
            "fraction" and "percent" can be used to make the colors in
                different plots comparable.
        values_pos (tuple[float, float]): The position of values inside the tile.
        values_fmt (str | "AUTO"): f-string format for values. Defaults to ".1%"
            (1 decimal place) if values_show_mode is "percent", else ".3g".
        values_color (str | "AUTO"): The font color of values. Use "AUTO" for
            automatically switch between black/white depending on the background.
        values_kwargs (dict): Keyword arguments passed to plt.text() for
            values. Defaults to None.

        # Colorbar
        show_cbar (bool): Whether to show colorbar.
        cbar_coords (tuple[float, float, float, float]): Colorbar
            position and size: [x, y, width, height] anchored at lower left
            corner of the bar. Defaults to (0.18, 0.8, 0.42, 0.05).
        cbar_range (tuple): Colorbar values display range, use None for auto
            detection for the corresponding boundary.
        cbar_label_fmt (str): f-string format option for color tick labels.
        cbar_title (str): Colorbar title. Defaults to "Values".
        cbar_title_kwargs (dict): Keyword arguments passed to
            cbar.ax.set_title(). Defaults to dict(fontsize=12, pad=10).
        cbar_kwargs (dict): Keyword arguments passed to fig.colorbar().

    Returns:
        plt.Figure: matplotlib Figure with the heatmap.
    """
    # TODO: tile_size and f_block_voffset are work in progress,
    # as there're issues that haven't been resolved in #157
    if f_block_voffset != 0 or tile_size != (0.75, 0.75):
        warnings.warn(
            "f_block_voffset and tile_size is still being worked on.",
            stacklevel=2,
        )

    # Prevent log scale and percent/fraction display mode being use together
    if log and values_show_mode in {"percent", "fraction"}:
        raise ValueError(f"Combining log scale and {values_show_mode=} is unsupported")

    # Initialize periodic table plotter
    projector = HMapPTableProjector(
        data=data,  # type: ignore[arg-type]
        exclude_elements=exclude_elements,  # type: ignore[arg-type]
        sci_notation=sci_notation,
        values_show_mode=values_show_mode,
        tile_size=tile_size,  # type: ignore[arg-type]
        log=log,  # type: ignore[arg-type]
        colormap=colormap,  # type: ignore[arg-type]
        plot_kwargs=plot_kwargs,  # type: ignore[arg-type]
        text_colors=text_colors,
        on_empty=on_empty,  # type: ignore[arg-type]
        hide_f_block=hide_f_block,  # type: ignore[arg-type]
    )

    # Set NaN/infinity colors and values
    if projector.anomalies != "NA" and projector.anomalies:
        for elem, anomalies in projector.anomalies.items():
            if "nan" in anomalies:
                projector.tile_colors[elem] = nan_color
                projector.tile_values[elem] = "NaN"
            elif "inf" in anomalies:
                projector.tile_colors[elem] = inf_color
                projector.tile_values[elem] = "∞"

    # Exclude elements
    for elem in exclude_elements:
        projector.tile_colors[elem] = "white"
        projector.tile_values[elem] = "excl."
        projector.text_colors[elem] = "black"

    # Call child plotter: heatmap
    projector.add_child_plots(
        ax_kwargs=ax_kwargs,
        f_block_voffset=f_block_voffset,
    )

    # Set better default symbol position
    if symbol_pos is None:
        symbol_pos = (0.5, 0.65) if values_show_mode != "off" else (0.5, 0.5)

    # Add element symbols
    symbol_kwargs = symbol_kwargs or {"fontsize": 16, "fontweight": "bold"}

    projector.add_elem_symbols(
        text=symbol_text,
        pos=symbol_pos,
        text_color=projector.text_colors,
        kwargs=symbol_kwargs,
    )

    # Show values upon request
    if values_show_mode != "off":
        # Generate values format depending on the display mode
        if values_fmt == "AUTO":
            if values_show_mode == "percent":
                values_fmt = ".1%"
            elif projector.sci_notation:
                values_fmt = ".2e"
            else:
                values_fmt = ".3g"

        projector.add_elem_values(
            pos=values_pos or (0.5, 0.25),
            text_fmt=values_fmt,
            kwargs=values_kwargs,
        )

    # Show colorbar upon request
    if show_cbar:
        # Generate colorbar tick label format
        cbar_kwargs = cbar_kwargs or {}
        cbar_kwargs.setdefault(
            "format",
            get_cbar_label_formatter(
                cbar_label_fmt=cbar_label_fmt,
                values_fmt=values_fmt,
                values_show_mode=values_show_mode,
                sci_notation=sci_notation,
            ),
        )

        cbar_title_kwargs = cbar_title_kwargs or {"fontsize": 16, "fontweight": "bold"}

        projector.add_colorbar(
            title=cbar_title,
            coords=cbar_coords,
            cbar_range=cbar_range,
            cbar_kwargs=cbar_kwargs,
            title_kwargs=cbar_title_kwargs,
        )

    return projector.fig


def ptable_heatmap_splits(
    data: pd.DataFrame | pd.Series | dict[str, list[list[float]]],
    *,
    # Heatmap-split specific
    start_angle: float = 135,
    # Figure-scope
    colormap: str | Colormap = "viridis",
    on_empty: Literal["hide", "show"] = "hide",
    hide_f_block: bool | Literal["AUTO"] = "AUTO",
    plot_kwargs: dict[str, Any] | None = None,
    # Axis-scope
    ax_kwargs: dict[str, Any] | None = None,
    # Symbol
    symbol_text: str | Callable[[Element], str] = lambda elem: elem.symbol,
    symbol_pos: tuple[float, float] = (0.5, 0.5),
    symbol_kwargs: dict[str, Any] | None = None,
    # Colorbar
    cbar_title: str = "Values",
    cbar_title_kwargs: dict[str, Any] | None = None,
    cbar_coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.02),
    cbar_kwargs: dict[str, Any] | None = None,
) -> plt.Figure:
    """Plot evenly-split heatmaps, nested inside a periodic table.

    Args:
        data (pd.DataFrame | pd.Series | dict[str, list[list[float]]]):
            Map from element symbols to plot data. E.g. if dict,
            {"Fe": [1, 2], "Co": [3, 4]}, where the 1st value would
            be plotted on the lower-left corner and the 2nd on the upper-right.
            If pd.Series, index is element symbols and values lists.
            If pd.DataFrame, column names are element symbols,
            plots are created from each column.

        # Heatmap-split specific
        start_angle (float): The starting angle for the splits in degrees,
            and the split proceeds counter-clockwise (0 refers to the x-axis).
            Defaults to 135 degrees.

        # Figure-scope
        colormap (str): Matplotlib colormap name to use.
        on_empty ("hide" | "show"): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool | "AUTO"): Hide f-block (Lanthanum and Actinium series).
            Defaults to "AUTO", meaning hide if no data is provided.
        plot_kwargs (dict): Additional keyword arguments to
            pass to the plt.subplots function call.

        # Axis-scope
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options.

        # Symbol
        symbol_text (str | Callable[[Element], str]): Text to display for
            each element symbol. Defaults to lambda elem: elem.symbol.
        symbol_pos (tuple[float, float]): Position of element symbols
            relative to the lower left corner of each tile.
            Defaults to (0.5, 0.5). (1, 1) is the upper right corner.
        symbol_kwargs (dict): Keyword arguments passed to ax.text() for
            element symbols. Defaults to None.

        # Colorbar
        cbar_title (str): Colorbar title. Defaults to "Values".
        cbar_title_kwargs (dict): Keyword arguments passed to
            cbar.ax.set_title(). Defaults to dict(fontsize=12, pad=10).
        cbar_coords (tuple[float, float, float, float]): Colorbar
            position and size: [x, y, width, height] anchored at lower left
            corner of the bar. Defaults to (0.25, 0.77, 0.35, 0.02).
        cbar_kwargs (dict): Keyword arguments passed to fig.colorbar().

    Notes:
        Default figsize is set to (0.75 * n_groups, 0.75 * n_periods).

    Returns:
        plt.Figure: periodic table with a subplot in each element tile.
    """
    # Initialize periodic table plotter
    projector = PTableProjector(
        data=data,
        colormap=colormap,
        plot_kwargs=plot_kwargs,
        on_empty=on_empty,
        hide_f_block=hide_f_block,
    )

    # Call child plotter: evenly split rectangle
    child_kwargs = {
        "start_angle": start_angle,
        "cmap": projector.cmap,
        "norm": projector.norm,
    }

    projector.add_child_plots(
        ChildPlotters.rectangle,
        child_kwargs=child_kwargs,
        ax_kwargs=ax_kwargs,
    )

    # Add element symbols
    projector.add_elem_symbols(
        text=symbol_text,
        pos=symbol_pos,
        kwargs=symbol_kwargs,
    )

    # Add colorbar
    projector.add_colorbar(
        title=cbar_title,
        coords=cbar_coords,
        cbar_kwargs=cbar_kwargs,
        title_kwargs=cbar_title_kwargs,
    )

    return projector.fig


def ptable_heatmap_ratio(
    values_num: ElemValues,
    values_denom: ElemValues,
    *,
    count_mode: ElemCountMode = ElemCountMode.composition,
    normalize: bool = False,
    zero_color: ColorType = "#lightgrey",
    inf_color: ColorType = "lightskyblue",
    # zero_symbol: str = "-",  # TODO
    # inf_symbol: str = "∞",
    cbar_title: str = "Element Ratio",
    not_in_numerator: tuple[str, str] | None = ("#eff", "gray: not in 1st list"),
    not_in_denominator: tuple[str, str] | None = (
        "lightskyblue",
        "blue: not in 2nd list",
    ),
    not_in_either: tuple[str, str] | None = ("white", "white: not in either"),
    **kwargs: Any,
) -> plt.Figure:
    """Display the ratio of two maps from element symbols to heat values or of two sets
    of compositions.

    Args:
        values_num (dict[str, int | float] | pd.Series | list[str]): Map from
            element symbols to heatmap values or iterable of composition strings/objects
            in the numerator.
        values_denom (dict[str, int | float] | pd.Series | list[str]): Map from
            element symbols to heatmap values or iterable of composition strings/objects
            in the denominator.
        normalize (bool): Whether to normalize heatmap values so they sum to 1. Makes
            different ptable_heatmap_ratio plots comparable. Defaults to False.
        zero_color: ColorType = "#lightgrey",  # TODO:
        inf_color: ColorType = "lightskyblue",
        count_mode ("composition" | "fractional_composition" | "reduced_composition"):
            Reduce or normalize compositions before counting. See count_elements() for
            details. Only used when values is list of composition strings/objects.
        cbar_title (str): Title for the colorbar. Defaults to "Element Ratio".
        not_in_numerator (tuple[str, str]): Color and legend description used for
            elements missing from numerator. Defaults to
            ("#eff", "gray: not in 1st list").
        not_in_denominator (tuple[str, str]): See not_in_numerator. Defaults to
            ("lightskyblue", "blue: not in 2nd list").
        not_in_either (tuple[str, str]): See not_in_numerator. Defaults to
            ("white", "white: not in either").
        **kwargs: Additional keyword arguments passed to ptable_heatmap().

    Returns:
        plt.Figure: matplotlib Figures object.
    """
    values_num = count_elements(values_num, count_mode)
    values_denom = count_elements(values_denom, count_mode)

    values = values_num / values_denom

    if normalize:
        values /= values.sum()

    # TODO: need to assign zero color and displayed value
    fig = ptable_heatmap(
        values,
        cbar_title=cbar_title,
        inf_color=inf_color,
        on_empty="show",
        **kwargs,
    )

    # # Add legend handles
    # for tup in (
    #     (0.18, "zero", *(not_in_numerator or ())),
    #     (0.12, "infty", *(not_in_denominator or ())),
    #     (0.06, "na", *(not_in_either or ())),
    # ):
    #     if len(tup) < 3:
    #         continue
    #     y_pos, key, color, txt = tup
    #     kwargs[f"{key}_color"] = color
    #     bbox = dict(facecolor=color, edgecolor="gray")
    #     fig.text(0.005, y_pos, txt, fontsize=10, bbox=bbox, transform=fig.transAxes)

    return fig


def ptable_hists(
    data: pd.DataFrame | pd.Series | dict[str, list[float]],
    *,
    # Histogram-specific
    bins: int = 20,
    x_range: tuple[float | None, float | None] | None = None,
    log: bool = False,
    # Figure-scope
    colormap: str | Colormap | None = "viridis",
    on_empty: Literal["show", "hide"] = "hide",
    hide_f_block: bool | Literal["AUTO"] = "AUTO",
    plot_kwargs: dict[str, Any] | None = None,
    # Axis-scope
    ax_kwargs: dict[str, Any] | None = None,
    child_kwargs: dict[str, Any] | None = None,
    # Colorbar
    cbar_axis: Literal["x", "y"] = "x",
    cbar_title: str = "Values",
    cbar_title_kwargs: dict[str, Any] | None = None,
    cbar_coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.02),
    cbar_kwargs: dict[str, Any] | None = None,
    # Symbol
    symbol_pos: tuple[float, float] = (0.5, 0.8),
    symbol_text: str | Callable[[Element], str] = lambda elem: elem.symbol,
    symbol_kwargs: dict[str, Any] | None = None,
    # Element types based colors and legend
    color_elem_strategy: Literal["symbol", "background", "both", "off"] = "background",
    elem_type_colors: dict[str, str] | None = None,
    add_elem_type_legend: bool = False,
    elem_type_legend_kwargs: dict[str, Any] | None = None,
) -> plt.Figure:
    """Plot histograms for each element laid out in a periodic table.

    Args:
        data (pd.DataFrame | pd.Series | dict[str, list[float]]): Map from element
            symbols to histogram values. E.g. if dict, {"Fe": [1, 2, 3], "O": [4, 5]}.
            If pd.Series, index is element symbols and values lists. If pd.DataFrame,
            column names are element symbols histograms are plotted from each column.

        # Histogram-specific
        bins (int): Number of bins for the histograms. Defaults to 20.
        x_range (tuple[float | None, float | None]): x-axis range for all histograms.
            Defaults to None.
        log (bool): Whether to log scale y-axis of each histogram. Defaults to False.

        # Figure-scope
        colormap (str): Matplotlib colormap name to use. Defaults to "viridis". See
            options at https://matplotlib.org/stable/users/explain/colors/colormaps.
        on_empty ("hide" | "show"): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool | "AUTO"): Hide f-block (Lanthanum and Actinium series).
            Defaults to "AUTO", meaning hide if no data is provided.
        plot_kwargs (dict): Additional keyword arguments to
            pass to the plt.subplots function call.

        # Axis-scope
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html#matplotlib-axes-axes-set
        child_kwargs (dict): Keywords passed to ax.hist() for each histogram.
            Defaults to None.

        # Colorbar
        cbar_axis ("x" | "y"): The axis colormap would be based on.
        cbar_title (str): Color bar title. Defaults to "Histogram Value".
        cbar_title_kwargs (dict): Keyword arguments passed to cbar.ax.set_title().
            Defaults to dict(fontsize=12, pad=10).
        cbar_coords (tuple[float, float, float, float]): Color bar position and size:
            [x, y, width, height] anchored at lower left corner of the bar. Defaults to
            (0.25, 0.77, 0.35, 0.02).
        cbar_kwargs (dict): Keyword arguments passed to fig.colorbar().

        # Symbol
        symbol_pos (tuple[float, float]): Position of element symbols relative to the
            lower left corner of each tile. Defaults to (0.5, 0.8). (1, 1) is the upper
            right corner.
        symbol_text (str | Callable[[Element], str]): Text to display for each element
            symbol. Defaults to lambda elem: elem.symbol.
        symbol_kwargs (dict): Keyword arguments passed to ax.text() for element
            symbols. Defaults to None.

        # Element types based colors and legend
        color_elem_strategy ("symbol" | "background" | "both" | "off"): Whether to
            color element symbols, tile backgrounds, or both based on element type.
            Defaults to "background".
        elem_type_colors (dict | None): dict to map element types to colors.
            None to use default element type colors.
        add_elem_type_legend (bool): Whether to show a legend for element
            types. Defaults to True.
        elem_type_legend_kwargs (dict): kwargs to plt.legend(), e.g. to
            set the legend title, use {"title": "Element Types"}.

    Returns:
        plt.Figure: periodic table with a histogram in each element tile.
    """
    # Initialize periodic table plotter
    projector = PTableProjector(
        data=data,
        colormap=colormap,
        plot_kwargs=plot_kwargs,
        on_empty=on_empty,
        hide_f_block=hide_f_block,
        elem_type_colors=elem_type_colors,
    )

    # Call child plotter: histogram
    child_kwargs = child_kwargs or {}
    child_kwargs |= {
        "bins": bins,
        "range": x_range,
        "log": log,
        "cbar_axis": cbar_axis,
        "cmap": projector.cmap,
    }

    projector.add_child_plots(
        ChildPlotters.histogram,
        child_kwargs=child_kwargs,
        ax_kwargs=ax_kwargs,
    )

    # Add element symbols
    projector.add_elem_symbols(
        text=symbol_text,
        pos=symbol_pos,
        text_color=ElemColorMode.element_types
        if color_elem_strategy in {"both", "symbol"}
        else "black",
        kwargs=symbol_kwargs,
    )

    # Color element tile background
    if color_elem_strategy in {"both", "background"}:
        projector.set_elem_background_color()

    # Add colorbar
    if colormap is not None:
        projector.add_colorbar(
            title=cbar_title,
            coords=cbar_coords,
            cbar_kwargs=cbar_kwargs,
            title_kwargs=cbar_title_kwargs,
        )

    # Add element type legend
    if add_elem_type_legend:
        projector.add_elem_type_legend(
            kwargs=elem_type_legend_kwargs,
        )

    return projector.fig


def ptable_scatters(
    data: pd.DataFrame | pd.Series | dict[str, list[list[float]]],
    *,
    # Figure-scope
    colormap: str | Colormap | None = None,
    on_empty: Literal["hide", "show"] = "hide",
    hide_f_block: bool | Literal["AUTO"] = "AUTO",
    plot_kwargs: dict[str, Any] | None = None,
    # Axis-scope
    ax_kwargs: dict[str, Any] | None = None,
    child_kwargs: dict[str, Any] | None = None,
    # Colorbar
    cbar_title: str = "Values",
    cbar_title_kwargs: dict[str, Any] | None = None,
    cbar_coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.02),
    cbar_kwargs: dict[str, Any] | None = None,
    # Symbol
    symbol_text: str | Callable[[Element], str] = lambda elem: elem.symbol,
    symbol_pos: tuple[float, float] = (0.5, 0.8),
    symbol_kwargs: dict[str, Any] | None = None,
    # Element types based colors and legend
    color_elem_strategy: Literal["symbol", "background", "both", "off"] = "background",
    elem_type_colors: dict[str, str] | None = None,
    add_elem_type_legend: bool = False,
    elem_type_legend_kwargs: dict[str, Any] | None = None,
) -> plt.Figure:
    """Make scatter plots for each element, nested inside a periodic table.

    Args:
        data (pd.DataFrame | pd.Series | dict[str, list[list[float]]]):
            Map from element symbols to plot data. E.g. if dict,
            {"Fe": [1, 2], "Co": [3, 4]}, where the 1st value would
            be plotted on the lower-left corner and the 2nd on the upper-right.
            If pd.Series, index is element symbols and values lists.
            If pd.DataFrame, column names are element symbols,
            plots are created from each column.

        # Figure-scope
        colormap (str): Matplotlib colormap name to use. Defaults to None'. See
            options at https://matplotlib.org/stable/users/explain/colors/colormaps.
        on_empty ("hide" | "show"): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool | "AUTO"): Hide f-block (Lanthanum and Actinium series).
            Defaults to "AUTO", meaning hide if no data is provided.
        plot_kwargs (dict): Additional keyword arguments to
            pass to the plt.subplots function call.

        # Axis-scope
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html#matplotlib-axes-axes-set
        child_kwargs: Arguments to pass to the child plotter call.

        # Colorbar
        cbar_title (str): Color bar title. Defaults to "Histogram Value".
        cbar_title_kwargs (dict): Keyword arguments passed to cbar.ax.set_title().
            Defaults to dict(fontsize=12, pad=10).
        cbar_coords (tuple[float, float, float, float]): Color bar position and size:
            [x, y, width, height] anchored at lower left corner of the bar. Defaults to
            (0.25, 0.77, 0.35, 0.02).
        cbar_kwargs (dict): Keyword arguments passed to fig.colorbar().

        # Symbol
        symbol_text (str | Callable[[Element], str]): Text to display for
            each element symbol. Defaults to lambda elem: elem.symbol.
        symbol_pos (tuple[float, float]): Position of element symbols
            relative to the lower left corner of each tile.
            Defaults to (0.5, 0.5). (1, 1) is the upper right corner.
        symbol_kwargs (dict): Keyword arguments passed to ax.text() for
            element symbols. Defaults to None.

        # Element types based colors and legend
        color_elem_strategy ("symbol" | "background" | "both" | "off"): Whether to
            color element symbols, tile backgrounds, or both based on element type.
            Defaults to "background".
        elem_type_colors (dict | None): dict to map element types to colors.
            None to use default element type colors.
        add_elem_type_legend (bool): Whether to show a legend for element
            types. Defaults to True.
        elem_type_legend_kwargs (dict): kwargs to plt.legend(), e.g. to
            set the legend title, use {"title": "Element Types"}.
    """
    # Initialize periodic table plotter
    projector = PTableProjector(
        data=data,
        colormap=colormap,
        plot_kwargs=plot_kwargs,
        on_empty=on_empty,
        hide_f_block=hide_f_block,
        elem_type_colors=elem_type_colors,
    )

    # Call child plotter: Scatter
    child_kwargs = child_kwargs or {}
    child_kwargs |= {"cmap": projector.cmap, "norm": projector.norm}

    projector.add_child_plots(
        ChildPlotters.scatter,
        child_kwargs=child_kwargs,
        ax_kwargs=ax_kwargs,
    )

    # Add element symbols
    projector.add_elem_symbols(
        text=symbol_text,
        pos=symbol_pos,
        text_color=ElemColorMode.element_types
        if color_elem_strategy in {"both", "symbol"}
        else "black",
        kwargs=symbol_kwargs,
    )

    # Color element tile background
    if color_elem_strategy in {"both", "background"}:
        projector.set_elem_background_color()

    # Add colorbar if colormap is given and data length is 3
    if colormap is not None:
        projector.add_colorbar(
            title=cbar_title,
            coords=cbar_coords,
            cbar_kwargs=cbar_kwargs,
            title_kwargs=cbar_title_kwargs,
        )

    # Add element type legend
    if add_elem_type_legend:
        projector.add_elem_type_legend(
            kwargs=elem_type_legend_kwargs,
        )

    return projector.fig


def ptable_lines(
    data: pd.DataFrame | pd.Series | dict[str, list[list[float]]],
    *,
    # Figure-scope
    on_empty: Literal["hide", "show"] = "hide",
    hide_f_block: bool | Literal["AUTO"] = "AUTO",
    plot_kwargs: dict[str, Any] | None = None,
    # Axis-scope
    ax_kwargs: dict[str, Any] | None = None,
    child_kwargs: dict[str, Any] | None = None,
    # Symbol
    symbol_kwargs: dict[str, Any] | None = None,
    symbol_text: str | Callable[[Element], str] = lambda elem: elem.symbol,
    symbol_pos: tuple[float, float] = (0.5, 0.8),
    # Element types based colors and legend
    color_elem_strategy: Literal["symbol", "background", "both", "off"] = "background",
    elem_type_colors: dict[str, str] | None = None,
    add_elem_type_legend: bool = False,
    elem_type_legend_kwargs: dict[str, Any] | None = None,
) -> plt.Figure:
    """Line plots for each element, nested inside a periodic table.

    Args:
        data (pd.DataFrame | pd.Series | dict[str, list[list[float]]]):
            Map from element symbols to plot data. E.g. if dict,
            {"Fe": [1, 2], "Co": [3, 4]}, where the 1st value would
            be plotted on the lower-left corner and the 2nd on the upper-right.
            If pd.Series, index is element symbols and values lists.
            If pd.DataFrame, column names are element symbols,
            plots are created from each column.

        # Figure-scope
        on_empty ("hide" | "show"): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool | "AUTO"): Hide f-block (Lanthanum and Actinium series).
            Defaults to "AUTO", meaning hide if no data is provided.
        plot_kwargs (dict): Additional keyword arguments to
            pass to the plt.subplots function call.

        # Axis-scope
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html#matplotlib-axes-axes-set
        child_kwargs: Arguments to pass to the child plotter call.

        # Symbol
        symbol_text (str | Callable[[Element], str]): Text to display for
            each element symbol. Defaults to lambda elem: elem.symbol.
        symbol_pos (tuple[float, float]): Position of element symbols
            relative to the lower left corner of each tile.
            Defaults to (0.5, 0.5). (1, 1) is the upper right corner.
        symbol_kwargs (dict): Keyword arguments passed to ax.text() for
            element symbols. Defaults to None.

        # Element types based colors and legend
        color_elem_strategy ("symbol" | "background" | "both" | "off"): Whether to
            color element symbols, tile backgrounds, or both based on element type.
            Defaults to "background".
        elem_type_colors (dict | None): dict to map element types to colors.
            None to use default element type colors.
        add_elem_type_legend (bool): Whether to show a legend for element
            types. Defaults to True.
        elem_type_legend_kwargs (dict): kwargs to plt.legend(), e.g. to
            set the legend title, use {"title": "Element Types"}.
    """
    # Initialize periodic table plotter
    projector = PTableProjector(
        data=data,
        colormap=None,
        plot_kwargs=plot_kwargs,
        on_empty=on_empty,
        hide_f_block=hide_f_block,
        elem_type_colors=elem_type_colors,
    )

    # Call child plotter: line
    projector.add_child_plots(
        ChildPlotters.line,
        child_kwargs=child_kwargs,
        ax_kwargs=ax_kwargs,
    )

    # Add element symbols
    projector.add_elem_symbols(
        text=symbol_text,
        pos=symbol_pos,
        text_color=ElemColorMode.element_types
        if color_elem_strategy in {"both", "symbol"}
        else "black",
        kwargs=symbol_kwargs,
    )

    # Color element tile background
    if color_elem_strategy in {"both", "background"}:
        projector.set_elem_background_color()

    # Add element type legend
    if add_elem_type_legend:
        projector.add_elem_type_legend(
            kwargs=elem_type_legend_kwargs,
        )

    return projector.fig
