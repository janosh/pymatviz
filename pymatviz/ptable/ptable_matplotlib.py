"""Various periodic table heatmaps with matplotlib and plotly."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple, Union

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.typing import ColorType
from pymatgen.core import Element

from pymatviz.enums import ElemColorMode, ElemCountMode, Key
from pymatviz.process_data import count_elements
from pymatviz.ptable._projector import ChildPlotters, PTableProjector
from pymatviz.utils import df_ptable, get_cbar_label_formatter, pick_bw_for_contrast


if TYPE_CHECKING:
    from typing import Any, Callable

    import matplotlib.pyplot as plt
    from matplotlib.colors import Colormap

    from pymatviz.ptable._process_data import PTableData

# Custom types
ElemStr = str  # element as a str
ElemValues = Union[dict[Union[str, int], float], pd.Series, Sequence[str]]


class TileValueColor(NamedTuple):
    """Value and colors for heatmap tiles.

    value (str): The value to display.
    text_color (ColorType): Color for both symbol and value.
    tile_color (ColorType): Color for the tile.
    """

    value: str | float
    text_color: ColorType
    tile_color: ColorType


class OverwriteTileValueColor(TileValueColor):
    """Overwrite Value and colors for heatmap tiles,
    where None is used for not overwriting.
    """

    value: str | float | None
    text_color: ColorType | None
    tile_color: ColorType | None


class HMapPTableProjector(PTableProjector):
    """With more heatmap-specific functionalities."""

    def __init__(  # TODO: remove __init__ altogether
        self,
        *,
        values_show_mode: Literal["value", "fraction", "percent", "off"] = "value",
        sci_notation: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """Init Heatmap plotter.

        Args:
            values_show_mode ("value" | "fraction" | "percent" | "off"):
                Values display mode.
            sci_notation (bool): Whether to use scientific notation for
                values and colorbar tick labels.
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

    def generate_tile_value_colors(  # TODO: need unit test
        self,
        *,
        text_colors: Literal["AUTO"] | ColorType | dict[ElemStr, ColorType] = "AUTO",
        overwrite_tiles: dict[ElemStr, TileValueColor],
        nan_color: ColorType,
        inf_color: ColorType,
        excluded_tile_color: ColorType = "white",
    ) -> dict[ElemStr, TileValueColor]:
        """Generate value and colors for element tiles.

        Args:
            overwrite_tiles (dict[ElemStr, TileValueColor]): Final
                entried to overwrite tile value and colors. Note this
                would overwrite everything, include exclusion and anomalies.

        Returns:
            dict[ElemStr, TileValueColor]: Element to value-color mapping.
        """
        tile_entries: dict[ElemStr, TileValueColor] = {}

        for elem in Element:
            # Handle excluded elements
            if elem in self.exclude_elements:
                tile_entries[elem] = TileValueColor(
                    "excl.",
                    pick_bw_for_contrast(excluded_tile_color),
                    excluded_tile_color,
                )
                continue

            # Handle NaN/infinity colors and values
            if self.anomalies != "NA" and elem in self.anomalies:
                if "nan" in self.anomalies[elem]:
                    tile_entries[elem] = TileValueColor(
                        "NaN", pick_bw_for_contrast(nan_color), nan_color
                    )
                elif "inf" in self.anomalies[elem]:
                    tile_entries[elem] = TileValueColor(
                        "∞", pick_bw_for_contrast(inf_color), inf_color
                    )

                continue

            # Try to get data from DataFrame
            try:
                value: np.ndarray | Sequence[float] = self.data.loc[elem, Key.heat_val]

                # Generate tile color and text color
                tile_color = self.cmap(self.norm(value))

                if text_colors == "AUTO":
                    text_color = pick_bw_for_contrast(tile_color)
                elif isinstance(text_colors, dict):
                    text_color = text_colors.get(elem, "black")
                else:
                    text_color = text_colors

                tile_entries[elem] = TileValueColor(value, text_color, tile_color)

            except KeyError:
                tile_entries[elem] = TileValueColor("-", "black", "lightgrey")

        # Apply overwrite colors
        for elem, ow_tile_entry in overwrite_tiles.items():
            tile_entries[elem] = TileValueColor(
                ow_tile_entry.value or tile_entries[elem].value,
                ow_tile_entry.text_color or tile_entries[elem].text_color,
                ow_tile_entry.tile_color or tile_entries[elem].tile_color,
            )

        return tile_entries

    def add_child_plots(  # type: ignore[override]
        self,
        values: dict[str, TileValueColor],
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

            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)

            # Pass axis kwargs
            if ax_kwargs:
                ax.set(**ax_kwargs)


def ptable_heatmap(
    data: pd.DataFrame | pd.Series | dict[str, list[list[float]]] | PTableData,
    *,
    # Heatmap specific
    colormap: str = "viridis",
    exclude_elements: Sequence[str] = (),
    overwrite_tiles: dict[ElemStr, OverwriteTileValueColor] | None = None,
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
    text_colors: Literal["AUTO"] | ColorType | dict[ElemStr, ColorType] = "AUTO",
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
            - dict[ElemStr, ColorType]: Element to color mapping.

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
        on_empty=on_empty,  # type: ignore[arg-type]
        hide_f_block=hide_f_block,  # type: ignore[arg-type]
    )

    # Generate value and colors (TileValueColor) for each tile
    tile_value_colors = projector.generate_tile_value_colors(
        text_colors=text_colors,
        overwrite_tiles=overwrite_tiles or {},
        inf_color=inf_color,
        nan_color=nan_color,
    )

    # # Set better default symbol position  # TODO: one-shot in add_child_plots
    # if symbol_pos is None:
    #     symbol_pos = (0.5, 0.65) if values_show_mode != "off" else (0.5, 0.5)

    # # Add element symbols
    # symbol_kwargs = symbol_kwargs or {"fontsize": 16, "fontweight": "bold"}

    #  # Show values upon request  # TODO:
    # if values_show_mode != "off":
    #     # Generate values format depending on the display mode
    #     if values_fmt == "AUTO":
    #         if values_show_mode == "percent":
    #             values_fmt = ".1%"
    #         elif projector.sci_notation:
    #             values_fmt = ".2e"
    #         else:
    #             values_fmt = ".3g"

    #     projector.add_elem_values(
    #         pos=values_pos or (0.5, 0.25),
    #         text_fmt=values_fmt,
    #     )

    # Add element symbol, value and color for each tile
    projector.add_child_plots(
        values=tile_value_colors,
        ax_kwargs=ax_kwargs,
        f_block_voffset=f_block_voffset,
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
    data: pd.DataFrame | pd.Series | dict[ElemStr, list[list[float]]],
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
        data (pd.DataFrame | pd.Series | dict[ElemStr, list[list[float]]]):
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
    zero_color: ColorType = "lightgrey",
    inf_color: ColorType = "lightskyblue",
    zero_symbol: str = "-",  # TODO: need docstring
    cbar_title: str = "Element Ratio",
    not_in_numerator: tuple[str, str] | None = ("lightgray", "gray: not in 1st list"),
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
        values_num (dict[ElemStr, int | float] | pd.Series | list[ElemStr]): Map from
            element symbols to heatmap values or iterable of composition strings/objects
            in the numerator.
        values_denom (dict[ElemStr, int | float] | pd.Series | list[ElemStr]): Map from
            element symbols to heatmap values or iterable of composition strings/objects
            in the denominator.
        normalize (bool): Whether to normalize heatmap values so they sum to 1. Makes
            different ptable_heatmap_ratio plots comparable. Defaults to False.
        zero_color: ColorType = "lightgrey",  # TODO: finish docstring
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
    # Generate ratio data
    values_num = count_elements(values_num, count_mode)
    values_denom = count_elements(values_denom, count_mode)

    values = values_num / values_denom

    if normalize:
        values /= values.sum()

    # Drop entries that is not is either (as NaN)
    values = values.dropna(inplace=False)

    # TODO: need to assign zero color?
    fig = ptable_heatmap(
        values,
        cbar_title=cbar_title,
        inf_color=inf_color,
        on_empty="show",
        **kwargs,
    )

    # Add legend handles
    for tup in (
        (0.18, "zero", *(not_in_numerator or ())),
        (0.12, "infty", *(not_in_denominator or ())),
        (0.06, "na", *(not_in_either or ())),
    ):
        if len(tup) < 3:
            continue
        y_pos, key, color, txt = tup
        kwargs[f"{key}_color"] = color
        bbox = dict(facecolor=color, edgecolor="gray")
        fig.text(0, y_pos, txt, fontsize=10, bbox=bbox, transform=fig.transFigure)

    return fig


def ptable_hists(
    data: pd.DataFrame | pd.Series | dict[ElemStr, list[float]],
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
        data (pd.DataFrame | pd.Series | dict[ElemStr, list[float]]): Map from element
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
    data: pd.DataFrame | pd.Series | dict[ElemStr, list[list[float]]],
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
        data (pd.DataFrame | pd.Series | dict[ElemStr, list[list[float]]]):
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
    data: pd.DataFrame | pd.Series | dict[ElemStr, list[list[float]]],
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
        data (pd.DataFrame | pd.Series | dict[ElemStr, list[list[float]]]):
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
