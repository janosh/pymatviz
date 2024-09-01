"""Various periodic table heatmaps with matplotlib and plotly."""

from __future__ import annotations

import warnings
from math import isclose
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt

from pymatviz.enums import ElemColorMode, ElemCountMode
from pymatviz.process_data import count_elements
from pymatviz.ptable._projector import (
    ChildPlotters,
    HeatMapPTableProjector,
    OverwriteTileValueColor,
    PTableProjector,
)
from pymatviz.utils import ElemValues, get_cbar_label_formatter, pick_bw_for_contrast


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    import pandas as pd
    from matplotlib.colors import Colormap
    from matplotlib.typing import ColorType
    from pymatgen.core import Element

    from pymatviz.ptable._process_data import PTableData

# Custom types
ElemStr = str  # element as a str


def ptable_heatmap(
    data: pd.DataFrame | pd.Series | dict[str, list[list[float]]] | PTableData,
    *,
    # Heatmap specific
    colormap: str = "viridis",
    exclude_elements: Sequence[str] = (),
    overwrite_tiles: dict[ElemStr, OverwriteTileValueColor] | None = None,
    infty_color: ColorType = "lightskyblue",
    nan_color: ColorType = "lightgrey",
    log: bool = False,
    sci_notation: bool = False,
    tile_size: tuple[float, float] = (0.75, 0.75),  # TODO: WIP, don't use
    # Figure
    on_empty: Literal["hide", "show"] = "show",
    hide_f_block: bool | Literal["auto"] = "auto",
    f_block_voffset: float = 0,  # TODO: WIP, don't use
    plot_kwargs: dict[str, Any] | None = None,
    # Axis
    ax_kwargs: dict[str, Any] | None = None,
    text_colors: Literal["auto"] | ColorType | dict[ElemStr, ColorType] = "auto",
    # Symbol
    symbol_pos: tuple[float, float] | None = None,
    symbol_kwargs: dict[str, Any] | None = None,
    # Value
    value_show_mode: Literal["value", "fraction", "percent", "off"] = "value",
    value_pos: tuple[float, float] | None = None,
    value_fmt: str = "auto",
    value_kwargs: dict[str, Any] | None = None,
    # Colorbar
    show_cbar: bool = True,
    cbar_coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.05),
    cbar_range: tuple[float | None, float | None] = (None, None),
    cbar_label_fmt: str = "auto",
    cbar_title: str = "Element Count",
    cbar_title_kwargs: dict[str, Any] | None = None,
    cbar_kwargs: dict[str, Any] | None = None,
    # Migration
    return_type: Literal["figure", "axes"] = "axes",
    # Deprecated args, don't use
    colorscale: str | None = None,
    heat_mode: Literal["value", "fraction", "percent"] | None = None,
    show_values: bool | None = None,
    fmt: str | None = None,
    cbar_fmt: str | None = None,
    show_scale: bool | None = None,
) -> plt.Axes:
    """Plot a heatmap across the periodic table.

    Args:
        data (pd.DataFrame | pd.Series | dict[str, list[list[float]]]):
            Map from element symbols to plot data. E.g. if dict,
            {"Fe": [1, 2], "Co": [3, 4]}, where the 1st value would
            be plotted on the lower-left corner and the 2nd on the upper-right.
            If pd.Series, index is element symbols and values lists.
            If pd.DataFrame, column names are element symbols,
            plots are created from each column.

        --- Heatmap ---
        colormap (str): The colormap to use.
        exclude_elements (Sequence[str]): Elements to exclude.
        overwrite_tiles (dict[ElemStr, OverwriteTileValueColor]): Force
            overwrite value or color for element tiles.
        infty_color (ColorType): The color to use for infinity.
        nan_color (ColorType): The color to use for missing value (NaN).
        log (bool): Whether to show colorbar in log scale.
        sci_notation (bool): Whether to use scientific notation for values and
            colorbar tick labels.
        tile_size (tuple[float, float]): The relative height and width of the tile.

        --- Figure ---
        on_empty ("hide" | "show"): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool | "auto"): Hide f-block (Lanthanum and Actinium series).
            Defaults to "auto", meaning hide if no data is provided.
        f_block_voffset (float): The vertical offset of f-block elements.
        plot_kwargs (dict): Additional keyword arguments to
            pass to the plt.subplots function call.

        --- Axis ---
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options.
        text_colors: Colors for element symbols and values.
            - "auto": Auto pick "black" or "white" based on the contrast
                of tile color for each element.
            - ColorType: Use the same ColorType for each element.
            - dict[ElemStr, ColorType]: Element to color mapping.

        --- Symbol ---
        symbol_pos (tuple[float, float]): Position of element symbols
            relative to the lower left corner of each tile.
            Defaults to (0.5, 0.5). (1, 1) is the upper right corner.
        symbol_kwargs (dict): Keyword arguments passed to plt.text() for
            element symbols. Defaults to None.

        --- Value ---
        value_show_mode (str): The values display mode:
            - "off": Hide values.
            - "value": Display values as is.
            - "fraction": As a fraction of the total (0.10).
            - "percent": As a percentage of the total (10%).
            "fraction" and "percent" can be used to make the colors in
                different plots comparable.
        value_pos (tuple[float, float]): The position of values inside the tile.
        value_fmt (str | "auto"): f-string format for values. Defaults to ".1%"
            (1 decimal place) if values_show_mode is "percent", else ".3g".
        value_color (str | "auto"): The font color of values. Use "auto" for
            automatically switch between black/white depending on the background.
        value_kwargs (dict): Keyword arguments passed to plt.text() for
            values. Defaults to None.

        --- Colorbar ---
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

        --- Migration  # TODO: remove after 2025-07-01 ---
        return_type ("figure" | "axes"): Whether to return plt.Figure or plt.axes.
            We encourage you to migrate to "figure".

        --- Deprecated args, don't use ---
        colorscale: Use "colormap" instead.
        heat_mode: Use "value_show_mode" instead.
        show_values: Use "value_show_mode" instead.
        fmt: Use "value_fmt" instead.
        cbar_fmt: Use "cbar_label_fmt" instead.
        show_scale: Use "show_cbar" instead.

    Returns:
        plt.Axes: matplotlib axes with the heatmap.
        or
        plt.Figure: matplotlib figure with the heatmap.
    """
    # TODO: tile_size and f_block_voffset are work in progress,
    # as there're issues that haven't been resolved in #157
    if f_block_voffset != 0 or tile_size != (0.75, 0.75):
        warnings.warn(
            "f_block_voffset and tile_size is still being worked on.",
            stacklevel=2,
        )

    # Handle deprecated args  # TODO: remove after 2025-01-01
    if colorscale is not None:
        warnings.warn("colorscale is deprecated in favor of colormap.", stacklevel=2)
        colormap = colorscale
    if heat_mode is not None:
        warnings.warn(
            "heat_mode is deprecated in favor of value_show_mode.", stacklevel=2
        )
        value_show_mode = heat_mode
    if show_values is not None:
        warnings.warn(
            "show_values is deprecated in favor of value_show_mode.", stacklevel=2
        )
        if not show_values:
            value_show_mode = "off"
    if fmt is not None:
        warnings.warn("fmt is deprecated in favor of value_fmt.", stacklevel=2)
        value_fmt = fmt
    if cbar_fmt is not None:
        warnings.warn(
            "cbar_fmt is deprecated in favor of cbar_label_fmt.", stacklevel=2
        )
        cbar_label_fmt = cbar_fmt
    if show_scale is not None:
        warnings.warn("show_scale is deprecated in favor of show_cbar.", stacklevel=2)
        show_cbar = show_scale

    # Prevent log scale and percent/fraction display mode being used together
    if log and value_show_mode in {"percent", "fraction"}:
        raise ValueError(f"Combining log scale and {value_show_mode=} is unsupported")

    # Initialize periodic table plotter
    projector = HeatMapPTableProjector(
        data=data,
        exclude_elements=exclude_elements,
        # tile_size=tile_size,
        log=log,
        colormap=colormap,
        plot_kwargs=plot_kwargs,
        on_empty=on_empty,
        hide_f_block=hide_f_block,
    )

    # Filter near zero values in log mode
    if log:
        projector.filter_near_zero()

    # Normalize data for "fraction/percent" modes
    if value_show_mode in {"fraction", "percent"}:
        normalized_data = projector.ptable_data
        normalized_data.normalize()
        projector.data = normalized_data.data  # use setter to update metadata

    # Generate value f-string formatter
    if value_fmt == "auto":
        if value_show_mode == "percent":
            value_fmt = ".1%"
        elif sci_notation:
            value_fmt = ".2e"
        elif log:
            value_fmt = ".0f"
        else:
            value_fmt = ".3g"

    # Generate value and colors (TileValueColor) for each tile
    tile_entries = projector.generate_tile_value_colors(
        text_colors=text_colors,
        overwrite_tiles=overwrite_tiles or {},
        infty_color=infty_color,
        nan_color=nan_color,
    )

    # Generate symbol and value positions
    value_pos = value_pos or (0.5, 0.25)
    if symbol_pos is None:
        symbol_pos = (0.5, 0.5) if value_show_mode == "off" else (0.5, 0.65)

    # Add element symbol, value and color for each tile
    projector.add_heatmap_tiles(
        tile_entries=tile_entries,
        f_block_voffset=f_block_voffset,
        symbol_pos=symbol_pos,
        symbol_kwargs=symbol_kwargs,
        sci_notation=sci_notation,
        value_show_mode=value_show_mode,
        value_fmt=value_fmt,
        value_pos=value_pos,
        value_kwargs=value_kwargs,
        ax_kwargs=ax_kwargs,
    )

    # Show colorbar upon request
    if show_cbar:
        # Generate colorbar tick label format
        cbar_kwargs = cbar_kwargs or {}
        cbar_kwargs.setdefault(
            "format",
            get_cbar_label_formatter(
                cbar_label_fmt=cbar_label_fmt,
                values_fmt=value_fmt,
                values_show_mode=value_show_mode,
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

    if return_type == "figure":
        return projector.fig
    warnings.warn(
        "We encourage you to return plt.figure for more consistent results.",
        stacklevel=2,
    )
    return plt.gca()


def ptable_heatmap_ratio(
    values_num: ElemValues,
    values_denom: ElemValues,
    *,
    count_mode: ElemCountMode = ElemCountMode.composition,
    normalize: bool = False,
    infty_color: ColorType = "lightskyblue",
    zero_color: ColorType = "lightgrey",
    zero_tol: float = 1e-6,
    zero_symbol: str = "ZERO",
    cbar_title: str = "Element Ratio",
    not_in_numerator: tuple[str, str] | None = ("lightgray", "gray: not in 1st list"),
    not_in_denominator: tuple[str, str] | None = (
        "lightskyblue",
        "blue: not in 2nd list",
    ),
    not_in_either: tuple[str, str] | None = ("white", "white: not in either"),
    **kwargs: Any,
) -> plt.figure:
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
        infty_color (ColorType): Color for infinity.
        zero_color (ColorType): Color for (near) zero element tiles.
        zero_tol (float): Absolute tolerance to consider a value zero.
        zero_symbol (str): Value to display for (near) zero element tiles.
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

    # Generate overwrite tile entries for near zero values
    overwrite_tiles = {}
    for elem, value in values.items():
        if isclose(a=value, b=0, abs_tol=zero_tol):
            overwrite_tiles[elem] = OverwriteTileValueColor(
                zero_symbol, pick_bw_for_contrast(zero_color), zero_color
            )

    # Generate heatmap
    fig = ptable_heatmap(
        values,
        cbar_title=cbar_title,
        infty_color=infty_color,
        on_empty="show",
        overwrite_tiles=overwrite_tiles,
        return_type="figure",
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


def ptable_heatmap_splits(
    data: pd.DataFrame | pd.Series | dict[ElemStr, list[list[float]]],
    *,
    # Heatmap-split specific
    start_angle: float = 135,
    # Figure
    colormap: str | Colormap = "viridis",
    on_empty: Literal["hide", "show"] = "hide",
    hide_f_block: bool | Literal["auto"] = "auto",
    plot_kwargs: dict[str, Any] | None = None,
    # Axis
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

        --- Heatmap-split ---
        start_angle (float): The starting angle for the splits in degrees,
            and the split proceeds counter-clockwise (0 refers to the x-axis).
            Defaults to 135 degrees.

        --- Figure ---
        colormap (str): Matplotlib colormap name to use.
        on_empty ("hide" | "show"): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool | "auto"): Hide f-block (Lanthanum and Actinium series).
            Defaults to "auto", meaning hide if no data is provided.
        plot_kwargs (dict): Additional keyword arguments to
            pass to the plt.subplots function call.

        --- Axis ---
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options.

        --- Symbol ---
        symbol_text (str | Callable[[Element], str]): Text to display for
            each element symbol. Defaults to lambda elem: elem.symbol.
        symbol_pos (tuple[float, float]): Position of element symbols
            relative to the lower left corner of each tile.
            Defaults to (0.5, 0.5). (1, 1) is the upper right corner.
        symbol_kwargs (dict): Keyword arguments passed to ax.text() for
            element symbols. Defaults to None.

        --- Colorbar ---
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


def ptable_hists(
    data: pd.DataFrame | pd.Series | dict[ElemStr, list[float]],
    *,
    # Histogram-specific
    bins: int = 20,
    x_range: tuple[float | None, float | None] | None = None,
    log: bool = False,
    # Figure
    colormap: str | Colormap | None = "viridis",
    on_empty: Literal["show", "hide"] = "hide",
    hide_f_block: bool | Literal["auto"] = "auto",
    plot_kwargs: dict[str, Any] | None = None,
    # Axis
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

        --- Histogram-specific ---
        bins (int): Number of bins for the histograms. Defaults to 20.
        x_range (tuple[float | None, float | None]): x-axis range for all histograms.
            Defaults to None.
        log (bool): Whether to log scale y-axis of each histogram. Defaults to False.

        --- Figure ---
        colormap (str): Matplotlib colormap name to use. Defaults to "viridis". See
            options at https://matplotlib.org/stable/users/explain/colors/colormaps.
        on_empty ("hide" | "show"): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool | "auto"): Hide f-block (Lanthanum and Actinium series).
            Defaults to "auto", meaning hide if no data is provided.
        plot_kwargs (dict): Additional keyword arguments to
            pass to the plt.subplots function call.

        --- Axis ---
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html#matplotlib-axes-axes-set
        child_kwargs (dict): Keywords passed to ax.hist() for each histogram.
            Defaults to None.

        --- Colorbar ---
        cbar_axis ("x" | "y"): The axis colormap would be based on.
        cbar_title (str): Color bar title. Defaults to "Histogram Value".
        cbar_title_kwargs (dict): Keyword arguments passed to cbar.ax.set_title().
            Defaults to dict(fontsize=12, pad=10).
        cbar_coords (tuple[float, float, float, float]): Color bar position and size:
            [x, y, width, height] anchored at lower left corner of the bar. Defaults to
            (0.25, 0.77, 0.35, 0.02).
        cbar_kwargs (dict): Keyword arguments passed to fig.colorbar().

        --- Symbol ---
        symbol_pos (tuple[float, float]): Position of element symbols relative to the
            lower left corner of each tile. Defaults to (0.5, 0.8). (1, 1) is the upper
            right corner.
        symbol_text (str | Callable[[Element], str]): Text to display for each element
            symbol. Defaults to lambda elem: elem.symbol.
        symbol_kwargs (dict): Keyword arguments passed to ax.text() for element
            symbols. Defaults to None.

        --- Element types based colors and legend ---
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
    # Figure
    colormap: str | Colormap | None = None,
    on_empty: Literal["hide", "show"] = "hide",
    hide_f_block: bool | Literal["auto"] = "auto",
    plot_kwargs: dict[str, Any] | None = None,
    # Axis
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

        --- Figure ---
        colormap (str): Matplotlib colormap name to use. Defaults to None'. See
            options at https://matplotlib.org/stable/users/explain/colors/colormaps.
        on_empty ("hide" | "show"): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool | "auto"): Hide f-block (Lanthanum and Actinium series).
            Defaults to "auto", meaning hide if no data is provided.
        plot_kwargs (dict): Additional keyword arguments to
            pass to the plt.subplots function call.

        --- Axis ---
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html#matplotlib-axes-axes-set
        child_kwargs: Arguments to pass to the child plotter call.

        --- Colorbar ---
        cbar_title (str): Color bar title. Defaults to "Histogram Value".
        cbar_title_kwargs (dict): Keyword arguments passed to cbar.ax.set_title().
            Defaults to dict(fontsize=12, pad=10).
        cbar_coords (tuple[float, float, float, float]): Color bar position and size:
            [x, y, width, height] anchored at lower left corner of the bar. Defaults to
            (0.25, 0.77, 0.35, 0.02).
        cbar_kwargs (dict): Keyword arguments passed to fig.colorbar().

        --- Symbol ---
        symbol_text (str | Callable[[Element], str]): Text to display for
            each element symbol. Defaults to lambda elem: elem.symbol.
        symbol_pos (tuple[float, float]): Position of element symbols
            relative to the lower left corner of each tile.
            Defaults to (0.5, 0.5). (1, 1) is the upper right corner.
        symbol_kwargs (dict): Keyword arguments passed to ax.text() for
            element symbols. Defaults to None.

        --- Element types based colors and legend ---
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
    # Figure
    on_empty: Literal["hide", "show"] = "hide",
    hide_f_block: bool | Literal["auto"] = "auto",
    plot_kwargs: dict[str, Any] | None = None,
    # Axis
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

        --- Figure ---
        on_empty ("hide" | "show"): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool | "auto"): Hide f-block (Lanthanum and Actinium series).
            Defaults to "auto", meaning hide if no data is provided.
        plot_kwargs (dict): Additional keyword arguments to
            pass to the plt.subplots function call.

        --- Axis ---
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html#matplotlib-axes-axes-set
        child_kwargs: Arguments to pass to the child plotter call.

        --- Symbol ---
        symbol_text (str | Callable[[Element], str]): Text to display for
            each element symbol. Defaults to lambda elem: elem.symbol.
        symbol_pos (tuple[float, float]): Position of element symbols
            relative to the lower left corner of each tile.
            Defaults to (0.5, 0.5). (1, 1) is the upper right corner.
        symbol_kwargs (dict): Keyword arguments passed to ax.text() for
            element symbols. Defaults to None.

        --- Element types based colors and legend ---
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
