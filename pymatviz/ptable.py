"""Various periodic table heatmaps with matplotlib and plotly."""

from __future__ import annotations

import inspect
import itertools
import math
import warnings
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Literal, Union, get_args

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from matplotlib.cm import get_cmap
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pymatgen.core import Composition, Element

from pymatviz.enums import Key
from pymatviz.utils import df_ptable, pick_bw_for_contrast, si_fmt, si_fmt_int


if TYPE_CHECKING:
    from typing import Any, Callable, Final

    import plotly.graph_objects as go


# Data types supported by ptable plotters
SupportedValueType = Union[Sequence[float], np.ndarray]

SupportedDataType = Union[
    dict[str, Union[float, Sequence[float], np.ndarray]], pd.DataFrame, pd.Series
]

CountMode = Literal[
    Key.composition, "fractional_composition", "reduced_composition", "occurrence"
]

ElemValues = Union[dict[Union[str, int], float], pd.Series, Sequence[str]]

ELEM_CLASS_COLORS: Final = {
    "Diatomic Nonmetal": "green",
    "Noble Gas": "purple",
    "Alkali Metal": "red",
    "Alkaline Earth Metal": "orange",
    "Metalloid": "darkgreen",
    "Polyatomic Nonmetal": "teal",
    "Transition Metal": "blue",
    "Post Transition Metal": "cyan",
    "Lanthanide": "brown",
    "Actinide": "gray",
    "Nonmetal": "green",
    "Halogen": "teal",
    "Metal": "lightblue",
    "Alkaline Metal": "magenta",
    "Transactinide": "olive",
}


def add_element_type_legend(
    data: pd.DataFrame | pd.Series | dict[str, list[float]],
    elem_class_colors: dict[str, str] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
) -> None:
    """Add a legend to a matplotlib figure showing the colors of element types.


    Args:
        data (pd.DataFrame | pd.Series | dict[str, list[float]]): Map from element
            to plot data. Used only to determine which element types are present.
        elem_class_colors (dict[str, str]): Map from element
            types to colors. E.g. {"Alkali Metal": "red", "Noble Gas": "blue"}.
        legend_kwargs (dict): Keyword arguments passed to plt.legend() for customizing
            legend appearance. Defaults to None.
    """
    elem_class_colors = ELEM_CLASS_COLORS | (elem_class_colors or {})
    # else case list(data) covers dict and DataFrame
    elems_with_data = data.index if isinstance(data, pd.Series) else list(data)
    visible_elem_types = df_ptable.loc[elems_with_data, "type"].unique()
    font_size = 10
    legend_elements = [
        plt.Line2D(
            *([0], [0]),
            marker="s",
            color="w",
            label=elem_class,
            markerfacecolor=color,
            markersize=1.2 * font_size,
        )
        for elem_class, color in elem_class_colors.items()
        if elem_class in visible_elem_types
    ]
    legend_kwargs = dict(
        loc="center left",
        bbox_to_anchor=(0, -42),
        ncol=6,
        frameon=False,
        fontsize=font_size,
        handlelength=1,  # more compact legend
    ) | (legend_kwargs or {})
    plt.legend(handles=legend_elements, **legend_kwargs)


def count_elements(
    values: ElemValues,
    count_mode: CountMode = Key.composition,
    exclude_elements: Sequence[str] = (),
    fill_value: float | None = 0,
) -> pd.Series:
    """Count element occurrence in list of formula strings or dict-like compositions.
    If passed values are already a map from element symbol to counts, ensure the
    data is a pd.Series filled with zero values for missing element symbols.

    Provided as standalone function for external use or to cache long computations.
    Caching long element counts is done by refactoring
        ptable_heatmap(long_list_of_formulas) # slow
    to
        elem_counts = count_elements(long_list_of_formulas) # slow
        ptable_heatmap(elem_counts) # fast, only rerun this line to update the plot

    Args:
        values (dict[str, int | float] | pd.Series | list[str]): Iterable of
            composition strings/objects or map from element symbols to heatmap values.
        count_mode ('(element|fractional|reduced)_composition'):
            Only used when values is a list of composition strings/objects.
            - composition (default): Count elements in each composition as is,
                i.e. without reduction or normalization.
            - fractional_composition: Convert to normalized compositions in which the
                amounts of each species sum to before counting.
                Example: Fe2 O3 -> Fe0.4 O0.6
            - reduced_composition: Convert to reduced compositions (i.e. amounts
                normalized by greatest common denominator) before counting.
                Example: Fe4 P4 O16 -> Fe P O4.
            - occurrence: Count the number of times each element occurs in a list of
                formulas irrespective of compositions. E.g. [Fe2 O3, Fe O, Fe4 P4 O16]
                counts to {Fe: 3, O: 3, P: 1}.
        exclude_elements (Sequence[str]): Elements to exclude from the count. Defaults
            to ().
        fill_value (float | None): Value to fill in for missing elements. Defaults to 0.

    Returns:
        pd.Series: Map element symbols to heatmap values.
    """
    if count_mode not in get_args(CountMode):
        raise ValueError(f"Invalid {count_mode=} must be one of {get_args(CountMode)}")
    # ensure values is Series if we got dict/list/tuple
    srs = pd.Series(values)

    if is_numeric_dtype(srs):
        pass
    elif is_string_dtype(srs):
        # assume all items in values are composition strings
        if count_mode == "occurrence":
            srs = pd.Series(
                itertools.chain.from_iterable(
                    map(str, Composition(comp, allow_negative=True)) for comp in srs
                )
            ).value_counts()
        else:
            attr = (
                "element_composition" if count_mode == Key.composition else count_mode
            )
            srs = pd.DataFrame(
                getattr(Composition(formula, allow_negative=True), attr).as_dict()
                for formula in srs
            ).sum()  # sum up element occurrences
    else:
        raise ValueError(
            "Expected values to be map from element symbols to heatmap values or "
            f"list of compositions (strings or Pymatgen objects), got {values}"
        )

    try:
        # if index consists entirely of strings representing integers, convert to ints
        srs.index = srs.index.astype(int)
    except (ValueError, TypeError):
        pass

    if pd.api.types.is_integer_dtype(srs.index):
        # if index is all integers, assume they represent atomic
        # numbers and map them to element symbols (H: 1, He: 2, ...)
        idx_min, idx_max = srs.index.min(), srs.index.max()
        if idx_max > 118 or idx_min < 1:
            raise ValueError(
                "element value keys were found to be integers and assumed to represent "
                f"atomic numbers, but values range from {idx_min} to {idx_max}, "
                "expected range [1, 118]."
            )
        map_atomic_num_to_elem_symbol = (
            df_ptable.reset_index().set_index("atomic_number").symbol
        )
        srs.index = srs.index.map(map_atomic_num_to_elem_symbol)

    # ensure all elements are present in returned Series (with value zero if they
    # weren't in values before)
    srs = srs.reindex(df_ptable.index, fill_value=fill_value).rename("count")

    if len(exclude_elements) > 0:
        if isinstance(exclude_elements, str):
            exclude_elements = [exclude_elements]
        if isinstance(exclude_elements, tuple):
            exclude_elements = list(exclude_elements)
        try:
            srs = srs.drop(exclude_elements)
        except KeyError as exc:
            bad_symbols = ", ".join(x for x in exclude_elements if x not in srs)
            raise ValueError(
                f"Unexpected symbol(s) {bad_symbols} in {exclude_elements=}"
            ) from exc

    return srs


def data_preprocessor(data: SupportedDataType) -> pd.DataFrame:
    """Preprocess input data for ptable plotters, including:
        - Convert all data types to pd.DataFrame.
        - Impute missing values.
        - Handle anomalies such as NaN, infinity.
        - Write vmin/vmax as metadata into the DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with element names
            as index and values as columns.

    Example:
        >>> data_dict: dict = {
            "H": 1.0,
            "He": [2.0, 4.0],
            "Li": [[6.0, 8.0], [10.0, 12.0]],
        }

        OR
        >>> data_df: pd.DataFrame = pd.DataFrame(
            data_dict.items(),
            columns=["element", "heat_val"]
            ).set_index("element")

        OR
        >>> data_series: pd.Series = pd.Series(data_dict)

        >>> preprocess_data(data_dict / df / series)

             Element   Value
        0    H         [1.0, ]
        1    He        [2.0, 4.0]
        2    Li        [[6.0, 8.0], [10.0, 12.0]]

        Metadata:
            vmin: 1.0
            vmax: 12.0
    """

    def set_vmin_vmax(df: pd.DataFrame) -> pd.DataFrame:
        """Write vmin and vmax to DataFrame metadata."""
        # flatten up to triple nested lists
        values = df[Key.heat_val].explode().explode().explode()
        numeric_values = pd.to_numeric(values, errors="coerce")

        df.attrs["vmin"] = numeric_values.min()  # ignores NaNs
        df.attrs["vmax"] = numeric_values.max()
        return df

    # Check and handle different supported data types
    if isinstance(data, pd.DataFrame):
        data_df = data

    elif isinstance(data, pd.Series):
        data_df = data.to_frame(name=Key.heat_val)
        data_df.index.name = Key.element

    elif isinstance(data, dict):
        data_df = pd.DataFrame(
            data.items(), columns=[Key.element, Key.heat_val]
        ).set_index(Key.element)

    else:
        raise TypeError(f"Unsupported data type, choose from: {SupportedDataType}.")

    # Convert all values to np.array
    data_df[Key.heat_val] = data_df[Key.heat_val].map(
        lambda x: np.array([x]) if isinstance(x, float) else np.array(x)
    )

    # Handle missing and anomalous values
    data_df = handle_missing_and_anomaly(data_df)

    # Write vmin/vmax into metadata
    return set_vmin_vmax(data_df)


def handle_missing_and_anomaly(
    df: pd.DataFrame,
    # missing_strategy: Literal["zero", "mean"] = "mean",
) -> pd.DataFrame:
    """Handle missing value (NaN) and anomaly (infinity).

    Infinity would be replaced by vmax(∞) or vmin(-∞).
    Missing values would be handled by selected strategy:
        - zero: impute with zeros
        - mean: impute with mean value

    TODO: finish this function
    """
    return df


class PTableProjector:
    """Project (nest) a custom plot into a periodic table.

    Scopes mentioned in this plotter:
        plot: Refers to the global scope.
        ax: Refers to the axis where child plotter would plot.
        child: Refers to the child plotter itself, for example, ax.plot().
    """

    def __init__(
        self,
        *,
        data: SupportedDataType,
        colormap: str | Colormap | None,
        plot_kwargs: dict[str, Any] | None = None,
        hide_f_block: bool | None = None,
    ) -> None:
        """Initialize a ptable projector.

        Default figsize is set to (0.75 * n_groups, 0.75 * n_periods),
        and axes would be turned off by default.

        Args:
            data (SupportedDataType): The data to be visualized.
            colormap (str | Colormap | None): The colormap to use.
            plot_kwargs (dict): Additional keyword arguments to
                pass to the plt.subplots function call.
            hide_f_block: Hide f-block (Lanthanum and Actinium series). Defaults to
                None, meaning hide if no data is provided for f-block elements.
        """
        # Get colormap
        self.cmap: Colormap = colormap

        # Preprocess data
        self.data: pd.DataFrame = data

        if hide_f_block is None:
            hide_f_block = bool(
                {
                    atom_num
                    for atom_num in [*range(57, 72), *range(89, 104)]  # rare earths
                    # check if data is present for f-block elements
                    if (elem := Element.from_Z(atom_num).symbol) in self.data.index  # type: ignore[union-attr]
                    and self.data.loc[elem, Key.heat_val]  # type: ignore[union-attr]
                }
            )

        self.hide_f_block = hide_f_block

        # Initialize periodic table canvas
        n_periods = df_ptable.row.max()
        n_groups = df_ptable.column.max()

        if self.hide_f_block:
            n_periods -= 3

        # Set figure size
        plot_kwargs = plot_kwargs or {}
        plot_kwargs.setdefault("figsize", (0.75 * n_groups, 0.75 * n_periods))

        self.fig, self.axes = plt.subplots(n_periods, n_groups, **plot_kwargs)

        # Turn off all axes
        for ax in self.axes.flat:
            ax.axis("off")

    @property
    def cmap(self) -> Colormap | None:
        """The periodic table's matplotlib Colormap instance."""
        return self._cmap

    @cmap.setter
    def cmap(self, colormap: str | Colormap | None) -> None:
        """Set the periodic table's matplotlib Colormap instance."""
        self._cmap = None if colormap is None else plt.get_cmap(colormap)

    @property
    def data(self) -> pd.DataFrame:
        """The preprocessed data."""
        return self._data

    @data.setter
    def data(self, data: SupportedDataType) -> None:
        """Set and preprocess the data. Also set normalizer."""
        # Preprocess data
        self._data: pd.DataFrame = data_preprocessor(data)

        # Normalize data for colorbar
        self._norm: Normalize = Normalize(
            vmin=self._data.attrs["vmin"], vmax=self._data.attrs["vmax"]
        )

    @property
    def norm(self) -> Normalize:
        """Data min-max normalizer."""
        return self._norm

    def add_child_plots(
        self,
        child_plotter: Callable[[plt.axes, Any], None],
        child_args: dict[str, Any],
        *,
        ax_kwargs: dict[str, Any],
        on_empty: Literal["hide", "show"] = "hide",
    ) -> None:
        """Add custom child plots to the periodic table grid.

        Args:
            child_plotter: A callable for the child plotter.
            child_args: Arguments to pass to the child plotter call.
            ax_kwargs: Keyword arguments to pass to ax.set().
            on_empty: Whether to "show" or "hide" tiles for elements without data.
        """
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

            if (plot_data is None or len(plot_data) == 0) and on_empty == "hide":
                continue

            # Call child plotter
            child_plotter(ax, plot_data, **child_args)

            # Pass axis kwargs
            if ax_kwargs:
                ax.set(**ax_kwargs)

    def add_ele_symbols(
        self,
        text: str | Callable[[Element], str] = lambda elem: elem.symbol,
        pos: tuple[float, float] = (0.5, 0.5),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Add element symbols for each tile.

        Args:
            text (str | Callable): The text to add to the tile.
                If a callable, it should accept a pymatgen Element object and return a
                string. If a string, it can contain a format
                specifier for an `elem` variable which will be replaced by the element.
            pos: The position of the text relative to the axes.
            kwargs: Additional keyword arguments to pass to the `ax.text`.
        """
        # Update symbol args
        kwargs = kwargs or {}
        kwargs.setdefault("fontsize", 18)

        # Add symbol for each element
        for element in Element:
            # Hide f-block
            if self.hide_f_block and (element.is_lanthanoid or element.is_actinoid):
                continue

            # Get axis index by element symbol
            symbol: str = element.symbol
            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax: plt.Axes = self.axes[row - 1][column - 1]

            anno = text(element) if callable(text) else text.format(elem=element)
            ax.text(
                *pos, anno, ha="center", va="center", transform=ax.transAxes, **kwargs
            )

    def add_colorbar(
        self,
        title: str,
        coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.02),
        *,
        cbar_kwargs: dict[str, Any] | None = None,
        title_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Add a global colorbar.

        Args:
            title: Title for the colorbar.
            coords: Coordinates of the colorbar (left, bottom, width, height).
                    Defaults to (0.18, 0.8, 0.42, 0.02).
            cbar_kwargs: Additional keyword arguments to pass to fig.colorbar().
            title_kwargs: Additional keyword arguments for the colorbar title.
        """
        # Update colorbar args
        cbar_kwargs = {"orientation": "horizontal"} | (cbar_kwargs or {})

        # Check colormap
        if self.cmap is None:
            raise ValueError("Cannot add colorbar without colormap.")

        # Add colorbar
        cbar_ax = self.fig.add_axes(coords)

        self.fig.colorbar(
            plt.cm.ScalarMappable(norm=self._norm, cmap=self.cmap),
            cax=cbar_ax,
            **cbar_kwargs,
        )

        # Set colorbar title
        title_kwargs = title_kwargs or {}
        title_kwargs.setdefault("fontsize", 12)
        title_kwargs.setdefault("pad", 10)
        title_kwargs["label"] = title

        cbar_ax.set_title(**title_kwargs)


class ChildPlotters:
    """Collect some pre-defined child plotters."""

    @staticmethod
    def rectangle(
        ax: plt.axes,
        data: SupportedValueType,
        norm: Normalize,
        cmap: Colormap,
        start_angle: float,
    ) -> None:
        """Rectangle heatmap plotter, could be evenly split.

        Could be evenly split, depending on the
        length of the data (could mix and match).

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for to
                the child plotter.
            norm (Normalize): Normalizer for data-color mapping.
            cmap (Colormap): Colormap used for value mapping.
            start_angle (float): The starting angle for the splits in degrees,
                and the split proceeds counter-clockwise (0 refers to the x-axis).
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
            wedgeprops=dict(clip_on=True),
        )

        # Crop the central rectangle from the pie chart
        rect = Rectangle((-0.5, -0.5), 1, 1, fc="none", ec="none")
        ax.set_clip_path(rect)

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)

    @staticmethod
    def scatter(
        ax: plt.axes,
        data: SupportedValueType,
        **child_args: Any,
    ) -> None:
        """Scatter plotter.

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for to
                the child plotter.
            child_args (dict): args to pass to the child plotter call
        """
        # Add scatter
        if len(data) == 2:
            ax.scatter(x=data[0], y=data[1], **child_args)
        elif len(data) == 3:
            ax.scatter(x=data[0], y=data[1], c=data[2], **child_args)

        # Adjust tick labels
        # TODO: how to achieve this from external?
        ax.tick_params(axis="both", which="major", labelsize=8)

        # Hide the right and top spines
        ax.axis("on")  # turned off by default
        ax.spines[["right", "top"]].set_visible(False)

    @staticmethod
    def line(
        ax: plt.axes,
        data: SupportedValueType,
        **child_args: Any,
    ) -> None:
        """Line plotter.

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for to
                the child plotter.
            child_args (dict): args to pass to the child plotter call
        """
        # Add line
        ax.plot(data[0], data[1], **child_args)

        # Adjust tick labels
        # TODO: how to achieve this from external?
        ax.tick_params(axis="both", which="major", labelsize=8)

        # Hide the right and top spines
        ax.axis("on")  # turned off by default
        ax.spines[["right", "top"]].set_visible(False)


def ptable_heatmap(
    values: ElemValues,
    log: bool | Normalize = False,
    ax: plt.Axes | None = None,
    count_mode: CountMode = Key.composition,
    cbar_title: str = "Element Count",
    cbar_range: tuple[float | None, float | None] | None = None,
    cbar_coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.05),
    cbar_kwargs: dict[str, Any] | None = None,
    colorscale: str = "viridis",
    show_scale: bool = True,
    show_values: bool = True,
    infty_color: str = "lightskyblue",
    na_color: str = "white",
    heat_mode: Literal["value", "fraction", "percent"] | None = "value",
    fmt: str | Callable[..., str] | None = None,
    cbar_fmt: str | Callable[..., str] | None = None,
    text_color: str | tuple[str, str] = "auto",
    exclude_elements: Sequence[str] = (),
    zero_color: str = "#eff",  # light gray
    zero_symbol: str | float = "-",
    text_style: dict[str, Any] | None = None,
    label_font_size: int = 16,
    value_font_size: int = 12,
    tile_size: float | tuple[float, float] = 0.9,
    f_block_voffset: float = 0.5,
    hide_f_block: bool | None = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot a heatmap across the periodic table of elements.

    Args:
        values (dict[str, int | float] | pd.Series | list[str]): Map from element
            symbols to heatmap values or iterable of composition strings/objects.
        log (bool | Normalize, optional): Whether colormap scale is log or linear. Can
            also take any matplotlib.colors.Normalize subclass such as SymLogNorm as
            custom color scale. Defaults to False.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        count_mode ('composition' | 'fractional_composition' | 'reduced_composition'):
            Reduce or normalize compositions before counting. See count_elements() for
            details. Only used when values is list of composition strings/objects.
        cbar_title (str, optional): Color bar title. Defaults to "Element Count".
        cbar_range (tuple[float | None, float | None], optional): Color bar range.
            Can be used e.g. to create multiple plots with identical color bars for
            visual comparison. Defaults to automatic based on data range.
        cbar_coords (tuple[float, float, float, float], optional): Color bar position
            and size: [x, y, width, height] anchored at lower left corner. Defaults to
            (0.18, 0.8, 0.42, 0.05).
        cbar_kwargs (dict[str, Any], optional): Additional keyword arguments passed to
            fig.colorbar(). Defaults to None.
        colorscale (str, optional): Matplotlib colormap name to use. Defaults to
            "viridis". See https://matplotlib.org/stable/users/explain/colors/colormaps
            for available options.
        show_scale (bool, optional): Whether to show the color bar. Defaults to True.
        show_values (bool, optional): Whether to show the heatmap values in each tile.
            Defaults to True.
        infty_color: Color to use for elements with value infinity. Defaults to
            "lightskyblue".
        na_color: Color to use for elements with value infinity. Defaults to "white".
        heat_mode ("value" | "fraction" | "percent" | None): Whether to display heat
            values as is, normalized as a fraction of the total, as percentages
            or not at all (None). Defaults to "value".
            "fraction" and "percent" can be used to make the colors in different
            ptable_heatmap() (and ptable_heatmap_ratio()) plots comparable.
        fmt (str): f-string format option for tile values. Defaults to ".1%"
            (1 decimal place) if heat_mode="percent" else ".3g". Use e.g. ",.0f" to
            format values with thousands separators and no decimal places.
        cbar_fmt (str): f-string format option to set a different color bar tick
            label format. Defaults to the above fmt.
        text_color (str | tuple[str, str]): What color to use for element symbols and
            heat labels. Must be a valid color name, or a 2-tuple of names, one to use
            for the upper half of the color scale, one for the lower half. The special
            value "auto" applies "black" on the lower and "white" on the upper half of
            the color scale. "auto_reverse" does the opposite. Defaults to "auto".
        exclude_elements (list[str]): Elements to exclude from the heatmap. E.g. if
            oxygen overpowers everything, you can try log=True or
            exclude_elements=["O"]. Defaults to ().
        zero_color (str): Hex color or recognized matplotlib color name to use for
            elements with value zero. Defaults to "#eff" (light gray).
        zero_symbol (str | float): Symbol to use for elements with value zero.
            Defaults to "-".
        text_style (dict[str, Any]): Additional keyword arguments passed to
            plt.text(). Defaults to dict(
                ha="center", fontsize=label_font_size, fontweight="semibold"
            )
        label_font_size (int): Font size for element symbols. Defaults to 16.
        value_font_size (int): Font size for heat values. Defaults to 12.
        tile_size (float | tuple[float, float]): Size of each tile in the periodic
            table as a fraction of available space before touching neighboring tiles.
            1 or (1, 1) means no gaps between tiles. Defaults to 0.9.
        cbar_coords (tuple[float, float, float, float]): Color bar position and size:
            [x, y, width, height] anchored at lower left corner of the bar. Defaults to
            (0.18, 0.8, 0.42, 0.05).
        f_block_voffset (float): Vertical offset for lanthanides and actinides
            (row 6 and 7) from the rest of the periodic table. Defaults to 0.5.
        hide_f_block (bool): Hide f-block (Lanthanum and Actinium series). Defaults to
            None, meaning hide if no data is provided for f-block elements.
        **kwargs: Additional keyword arguments passed to plt.figure().

    Returns:
        plt.Axes: matplotlib Axes with the heatmap.
    """

    def tick_fmt(val: float, _pos: int) -> str:
        # val: value at color axis tick (e.g. 10.0, 20.0, ...)
        # pos: zero-based tick counter (e.g. 0, 1, 2, ...)
        default_fmt = (
            ".0%" if heat_mode == "percent" else (".0f" if val < 1e4 else ".2g")
        )
        return f"{val:{cbar_fmt or fmt or default_fmt}}"

    if fmt is None:
        fmt = partial(si_fmt, fmt=".1%" if heat_mode == "percent" else ".0f")
    if cbar_fmt is None:
        cbar_fmt = fmt

    valid_logs = (bool, Normalize)
    if not isinstance(log, valid_logs):
        raise TypeError(f"Invalid {log=}, must be instance of {valid_logs}")

    if log and heat_mode in ("fraction", "percent"):
        raise ValueError(
            "Combining log color scale and heat_mode='fraction'/'percent' unsupported"
        )
    if "cmap" in kwargs:
        colorscale = kwargs.pop("cmap")
        warnings.warn(
            "cmap argument is deprecated, use colorscale instead",
            category=DeprecationWarning,
        )

    values = count_elements(values, count_mode, exclude_elements)

    # replace positive and negative infinities with NaN values, then drop all NaNs
    clean_vals = values.replace([np.inf, -np.inf], np.nan).dropna()

    if heat_mode in ("fraction", "percent"):
        # ignore inf values in sum() else all would be set to 0 by normalizing
        values /= clean_vals.sum()
        clean_vals /= clean_vals.sum()  # normalize as well for norm.autoscale() below

    color_map = get_cmap(colorscale)

    n_rows = df_ptable.row.max()
    n_columns = df_ptable.column.max()

    # TODO can we pass as a kwarg and still ensure aspect ratio respected?
    fig = plt.figure(figsize=(0.75 * n_columns, 0.7 * n_rows), **kwargs)

    ax = ax or plt.gca()

    if isinstance(tile_size, (float, int)):
        tile_width = tile_height = tile_size
    else:
        tile_width, tile_height = tile_size

    norm_map = {True: LogNorm(), False: Normalize()}
    norm = norm_map.get(log, log)

    norm.autoscale(clean_vals.to_numpy())
    if cbar_range is not None and len(cbar_range) == 2:
        if cbar_range[0] is not None:
            norm.vmin = cbar_range[0]
        if cbar_range[1] is not None:
            norm.vmax = cbar_range[1]

    text_style = dict(
        horizontalalignment="center", fontsize=label_font_size, fontweight="semibold"
    ) | (text_style or {})

    for symbol, row, column, *_ in df_ptable.itertuples():
        if hide_f_block and (row in (6, 7)):
            continue

        period = n_rows - row  # invert row count to make periodic table right side up
        tile_value = values.get(symbol)

        # inf (float/0) or NaN (0/0) are expected when passing in values from
        # ptable_heatmap_ratio
        if symbol in exclude_elements:
            color = "white"
            label = "excl."
        elif tile_value == np.inf:
            color = infty_color  # not in denominator
            label = r"$\infty$"
        elif pd.isna(tile_value):
            color = na_color  # neither numerator nor denominator
            label = r"$0\,/\,0$"
        elif tile_value == 0:
            color = zero_color
            label = str(zero_symbol)
        else:
            color = color_map(norm(tile_value))

            if callable(fmt):
                if len(inspect.signature(fmt).parameters) == 2:
                    # 2nd arg=0 needed for matplotlib which always passes 2 positional
                    # args to fmt()
                    label = fmt(tile_value, 0)
                else:
                    label = fmt(tile_value)

            elif heat_mode == "percent":
                label = f"{tile_value:{fmt or '.1f'}}"
            else:
                fmt = fmt or (".0f" if tile_value > 100 else ".1f")
                label = f"{tile_value:{fmt}}"
            # replace shortens scientific notation 1e+01 to 1e1 so it fits inside cells
            label = label.replace("e+0", "e")
        if period < 3:  # vertical offset for lanthanides + actinides
            period += f_block_voffset
        rect = Rectangle(
            (column, period), tile_width, tile_height, edgecolor="gray", facecolor=color
        )

        if heat_mode is None or not show_values:
            # no value to display below in colored rectangle so center element symbol
            text_style.setdefault("verticalalignment", "center")

        if symbol in exclude_elements:
            text_clr = "black"
        elif text_color == "auto":
            if isinstance(color, (tuple, list)) and len(color) >= 3:
                # treat color as RGB tuple and choose black or white text for contrast
                text_clr = pick_bw_for_contrast(color)
            else:
                text_clr = "black"
        elif isinstance(text_color, (tuple, list)):
            text_clr = text_color[0] if norm(tile_value) > 0.5 else text_color[1]
        else:
            text_clr = text_color

        text_style.setdefault("color", text_clr)

        plt.text(
            column + 0.5 * tile_width,
            # 0.45 needed to achieve vertical centering, not sure why 0.5 is off
            period + (0.5 if show_values else 0.45) * tile_height,
            symbol,
            **text_style,
        )

        if heat_mode is not None and show_values:
            plt.text(
                column + 0.5 * tile_width,
                period + 0.1 * tile_height,
                label,
                fontsize=value_font_size,
                horizontalalignment="center",
                color=text_clr,
            )

        ax.add_patch(rect)

    if heat_mode is not None and show_scale:
        # colorbar position and size: [x, y, width, height]
        # anchored at lower left corner
        cbar_ax = ax.inset_axes(cbar_coords, transform=ax.transAxes)
        # format major and minor ticks
        # TODO maybe give user direct control over labelsize, instead of hard-coding
        # 8pt smaller than default
        cbar_ax.tick_params(which="both", labelsize=text_style["fontsize"])

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=colorscale)

        if callable(cbar_fmt):
            tick_fmt = cbar_fmt

        cbar_kwargs = cbar_kwargs or {}
        cbar = fig.colorbar(
            mappable,
            cax=cbar_kwargs.pop("cax", cbar_ax),
            orientation=cbar_kwargs.pop("orientation", "horizontal"),
            format=cbar_kwargs.pop("format", tick_fmt),
            **cbar_kwargs,
        )

        cbar.outline.set_linewidth(1)
        cbar_ax.set_title(cbar_title, pad=10, **text_style)

    plt.ylim(0.3, n_rows + 0.1)
    plt.xlim(0.9, n_columns + 1)

    plt.axis("off")
    return ax


def ptable_heatmap_splits(
    data: pd.DataFrame | pd.Series | dict[str, list[list[float]]],
    colormap: str | None = None,
    start_angle: float = 135,
    symbol_text: str | Callable[[Element], str] = lambda elem: elem.symbol,
    symbol_pos: tuple[float, float] = (0.5, 0.5),
    cbar_coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.02),
    cbar_title: str = "Values",
    on_empty: Literal["hide", "show"] = "hide",
    hide_f_block: bool | None = None,
    ax_kwargs: dict[str, Any] | None = None,
    symbol_kwargs: dict[str, Any] | None = None,
    plot_kwargs: dict[str, Any]
    | Callable[[Sequence[float]], dict[str, Any]]
    | None = None,
    cbar_title_kwargs: dict[str, Any] | None = None,
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
        colormap (str): Matplotlib colormap name to use.
        start_angle (float): The starting angle for the splits in degrees,
                and the split proceeds counter-clockwise (0 refers to
                the x-axis). Defaults to 135 degrees.
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html#matplotlib-axes-axes-set
        symbol_text (str | Callable[[Element], str]): Text to display for
            each element symbol. Defaults to lambda elem: elem.symbol.
        symbol_kwargs (dict): Keyword arguments passed to plt.text() for
            element symbols. Defaults to None.
        symbol_pos (tuple[float, float]): Position of element symbols
            relative to the lower left corner of each tile.
            Defaults to (0.5, 0.5). (1, 1) is the upper right corner.
        cbar_coords (tuple[float, float, float, float]): Colorbar
            position and size: [x, y, width, height] anchored at lower left
            corner of the bar. Defaults to (0.25, 0.77, 0.35, 0.02).
        cbar_title (str): Colorbar title. Defaults to "Values".
        cbar_title_kwargs (dict): Keyword arguments passed to
            cbar.ax.set_title(). Defaults to dict(fontsize=12, pad=10).
        cbar_kwargs (dict): Keyword arguments passed to fig.colorbar().
        on_empty ('hide' | 'show'): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool): Hide f-block (Lanthanum and Actinium series). Defaults to
            None, meaning hide if no data is provided for f-block elements.
        plot_kwargs (dict): Additional keyword arguments to
                pass to the plt.subplots function call.

    Notes:
        Default figsize is set to (0.75 * n_groups, 0.75 * n_periods).

    Returns:
        plt.Figure: periodic table with a subplot in each element tile.
    """
    # Re-initialize kwargs as empty dict if None
    plot_kwargs = plot_kwargs or {}
    ax_kwargs = ax_kwargs or {}
    symbol_kwargs = symbol_kwargs or {}
    cbar_title_kwargs = cbar_title_kwargs or {}
    cbar_kwargs = cbar_kwargs or {}

    # Initialize periodic table plotter
    plotter = PTableProjector(
        data=data,
        colormap=colormap,
        plot_kwargs=plot_kwargs,  # type: ignore[arg-type]
        hide_f_block=hide_f_block,
    )

    # Call child plotter: evenly split rectangle
    child_args = {
        "start_angle": start_angle,
        "cmap": plotter.cmap,
        "norm": plotter.norm,
    }

    plotter.add_child_plots(
        ChildPlotters.rectangle,  # type: ignore[arg-type]
        child_args=child_args,
        ax_kwargs=ax_kwargs,
        on_empty=on_empty,
    )

    # Add element symbols
    plotter.add_ele_symbols(
        text=symbol_text,
        pos=symbol_pos,
        kwargs=symbol_kwargs,
    )

    # Add colorbar
    plotter.add_colorbar(
        title=cbar_title,
        coords=cbar_coords,
        cbar_kwargs=cbar_kwargs,
        title_kwargs=cbar_title_kwargs,
    )

    return plotter.fig


def ptable_heatmap_ratio(
    values_num: ElemValues,
    values_denom: ElemValues,
    count_mode: CountMode = Key.composition,
    normalize: bool = False,
    cbar_title: str = "Element Ratio",
    not_in_numerator: tuple[str, str] | None = ("#eff", "gray: not in 1st list"),
    not_in_denominator: tuple[str, str] | None = (
        "lightskyblue",
        "blue: not in 2nd list",
    ),
    not_in_either: tuple[str, str] | None = ("white", "white: not in either"),
    **kwargs: Any,
) -> plt.Axes:
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
        count_mode ('composition' | 'fractional_composition' | 'reduced_composition'):
            Reduce or normalize compositions before counting. See count_elements() for
            details. Only used when values is list of composition strings/objects.
        cbar_title (str): Title for the colorbar. Defaults to "Element Ratio".
        not_in_numerator (tuple[str, str]): Color and legend description used for
            elements missing from numerator. Defaults to
            ('#eff', 'gray: not in 1st list').
        not_in_denominator (tuple[str, str]): See not_in_numerator. Defaults to
            ('lightskyblue', 'blue: not in 2nd list').
        not_in_either (tuple[str, str]): See not_in_numerator. Defaults to
            ('white', 'white: not in either').
        **kwargs: Additional keyword arguments passed to ptable_heatmap().

    Returns:
        plt.Axes: matplotlib Axes object
    """
    values_num = count_elements(values_num, count_mode)

    values_denom = count_elements(values_denom, count_mode)

    values = values_num / values_denom

    if normalize:
        values /= values.sum()

    # add legend handles
    for tup in (
        (2.1, "zero", *(not_in_numerator or ())),
        (1.4, "infty", *(not_in_denominator or ())),
        (0.7, "na", *(not_in_either or ())),
    ):
        if len(tup) < 3:
            continue
        y_pos, key, color, txt = tup
        kwargs[f"{key}_color"] = color
        bbox = dict(facecolor=color, edgecolor="gray")
        plt.text(0.8, y_pos, txt, fontsize=10, bbox=bbox)

    return ptable_heatmap(values, cbar_title=cbar_title, **kwargs)


def ptable_heatmap_plotly(
    values: ElemValues,
    count_mode: CountMode = Key.composition,
    colorscale: str | Sequence[str] | Sequence[tuple[float, str]] = "viridis",
    show_scale: bool = True,
    show_values: bool = True,
    heat_mode: Literal["value", "fraction", "percent"] | None = "value",
    fmt: str | None = None,
    hover_props: Sequence[str] | dict[str, str] | None = None,
    hover_data: dict[str, str | int | float] | pd.Series | None = None,
    font_colors: Sequence[str] = (),
    gap: float = 5,
    font_size: int | None = None,
    bg_color: str | None = None,
    color_bar: dict[str, Any] | None = None,
    cscale_range: tuple[float | None, float | None] = (None, None),
    exclude_elements: Sequence[str] = (),
    log: bool = False,
    fill_value: float | None = None,
    label_map: dict[str, str] | Callable[[str], str] | Literal[False] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Create a Plotly figure with an interactive heatmap of the periodic table.
    Supports hover tooltips with custom data or atomic reference data like
    electronegativity, atomic_radius, etc. See kwargs hover_data and hover_props, resp.

    Args:
        values (dict[str, int | float] | pd.Series | list[str]): Map from element
            symbols to heatmap values e.g. dict(Fe=2, O=3) or iterable of composition
            strings or Pymatgen composition objects.
        count_mode ("composition" | "fractional_composition" | "reduced_composition"):
            Reduce or normalize compositions before counting. See count_elements() for
            details. Only used when values is list of composition strings/objects.
        colorscale (str | list[str] | list[tuple[float, str]]): Color scale for heatmap.
            Defaults to 'viridis'. See plotly.com/python/builtin-colorscales for names
            of other builtin color scales. Note "YlGn" and px.colors.sequential.YlGn are
            equivalent. Custom scales are specified as ["blue", "red"] or
            [[0, "rgb(0,0,255)"], [0.5, "rgb(0,255,0)"], [1, "rgb(255,0,0)"]].
        show_scale (bool): Whether to show a bar for the color scale. Defaults to True.
        show_values (bool): Whether to show numbers on heatmap tiles. Defaults to True.
        heat_mode ("value" | "fraction" | "percent" | None): Whether to display heat
            values as is (value), normalized as a fraction of the total, as percentages
            or not at all (None). Defaults to "value".
            "fraction" and "percent" can be used to make the colors in different
            periodic table heatmap plots comparable.
        fmt (str): f-string format option for heat labels. Defaults to ".1%"
            (1 decimal place) if heat_mode="percent" else ".3g".
        hover_props (list[str] | dict[str, str]): Elemental properties to display in the
            hover tooltip. Can be a list of property names to display only the values
            themselves or a dict mapping names to what they should display as. E.g.
            dict(atomic_mass="atomic weight") will display as `"atomic weight = {x}"`.
            Defaults to None.
            Available properties are: symbol, row, column, name,
            atomic_number, atomic_mass, n_neutrons, n_protons, n_electrons, period,
            group, phase, radioactive, natural, metal, nonmetal, metalloid, type,
            atomic_radius, electronegativity, first_ionization, density, melting_point,
            boiling_point, number_of_isotopes, discoverer, year, specific_heat,
            n_shells, n_valence.
        hover_data (dict[str, str | int | float] | pd.Series): Map from element symbols
            to additional data to display in the hover tooltip. dict(Fe="this appears in
            the hover tooltip on a new line below the element name"). Defaults to None.
        font_colors (list[str]): One color name or two for [min_color, max_color].
            min_color is applied to annotations with heatmap values less than
            (max_val - min_val) / 2. Defaults to None, meaning auto-set to maximize
            contrast with color scale: white text for dark background and vice versa.
            swapped depending on the colorscale.
        gap (float): Gap in pixels between tiles of the periodic table. Defaults to 5.
        font_size (int): Element symbol and heat label text size. Any valid CSS size
            allowed. Defaults to automatic font size based on plot size. Element symbols
            will be bold and 1.5x this size.
        bg_color (str): Plot background color. Defaults to "rgba(0, 0, 0, 0)".
        color_bar (dict[str, Any]): Plotly colorbar properties documented at
            https://plotly.com/python/reference#heatmap-colorbar. Defaults to
            dict(orientation="h"). Commonly used keys are:
            - title: colorbar title
            - titleside: "top" | "bottom" | "right" | "left"
            - tickmode: "array" | "auto" | "linear" | "log" | "date" | "category"
            - tickvals: list of tick values
            - ticktext: list of tick labels
            - tickformat: f-string format option for tick labels
            - len: fraction of plot height or width depending on orientation
            - thickness: fraction of plot height or width depending on orientation
        cscale_range (tuple[float | None, float | None]): Colorbar range. Defaults to
            (None, None) meaning the range is automatically determined from the data.
        exclude_elements (list[str]): Elements to exclude from the heatmap. E.g. if
            oxygen overpowers everything, you can do exclude_elements=['O'].
            Defaults to ().
        log (bool): Whether to use a logarithmic color scale. Defaults to False.
            Piece of advice: colorscale='viridis' and log=True go well together.
        fill_value (float | None): Value to fill in for missing elements. Defaults to 0.
        label_map (dict[str, str] | Callable[[str], str] | None): Map heat values (after
            string formatting) to target strings. Set to False to disable. Defaults to
            dict.fromkeys((np.nan, None, "nan"), " ") so as not to display 'nan' for
            missing values.
        **kwargs: Additional keyword arguments passed to
            plotly.figure_factory.create_annotated_heatmap().

    Returns:
        Figure: Plotly Figure object.
    """
    if log and heat_mode in ("fraction", "percent"):
        raise ValueError(
            "Combining log color scale and heat_mode='fraction'/'percent' unsupported"
        )
    if len(cscale_range) != 2:
        raise ValueError(f"{cscale_range=} should have length 2")

    if isinstance(colorscale, (str, type(None))):
        colorscale = px.colors.get_colorscale(colorscale or "viridis")
    elif not isinstance(colorscale, Sequence) or not isinstance(
        colorscale[0], (str, list, tuple)
    ):
        raise TypeError(
            f"{colorscale=} should be string, list of strings or list of "
            "tuples(float, str)"
        )

    color_bar = color_bar or {}
    color_bar.setdefault("orientation", "h")
    # if values is a series with a name, use it as the colorbar title
    if isinstance(values, pd.Series) and values.name:
        color_bar.setdefault("title", values.name)

    values = count_elements(values, count_mode, exclude_elements, fill_value)

    if heat_mode in ("fraction", "percent"):
        # normalize heat values
        clean_vals = values.replace([np.inf, -np.inf], np.nan).dropna()
        # ignore inf values in sum() else all would be set to 0 by normalizing
        heat_value_element_map = values / clean_vals.sum()
    else:
        heat_value_element_map = values

    n_rows, n_columns = 10, 18
    # initialize tile text and hover tooltips to empty strings
    tile_texts, hover_texts = np.full([2, n_rows, n_columns], "", dtype=object)
    heatmap_values = np.full([n_rows, n_columns], np.nan)

    if label_map is None:
        # default to space string for None, np.nan and "nan". space is needed
        # for <br> in tile_text to work so all element symbols are vertically aligned
        label_map = dict.fromkeys([np.nan, None, "nan"], " ")  # type: ignore[list-item]

    for symbol, period, group, name, *_ in df_ptable.itertuples():
        # build table from bottom up so that period 1 becomes top row
        row = n_rows - period
        col = group - 1

        label = ""  # label (if not None) is placed below the element symbol
        if show_values:
            if symbol in exclude_elements:
                label = "excl."
            elif heat_value := heat_value_element_map.get(symbol):
                if heat_mode == "percent":
                    label = f"{heat_value:{fmt or '.1%'}}"
                else:
                    default_prec = ".1f" if heat_value < 100 else ",.0f"
                    if heat_value > 1e5:
                        default_prec = ".2g"
                    label = f"{heat_value:{fmt or default_prec}}".replace("e+0", "e")

            if callable(label_map):
                label = label_map(label)
            elif isinstance(label_map, dict):
                label = label_map.get(label, label)
        style = f"font-weight: bold; font-size: {1.5 * (font_size or 12)};"
        tile_text = (
            f"<span {style=}>{symbol}</span>{f'<br>{label}' if show_values else ''}"
        )

        tile_texts[row][col] = tile_text

        hover_text = name

        if hover_data is not None and symbol in hover_data:
            hover_text += f"<br>{hover_data[symbol]}"

        if hover_props is not None:
            if unsupported_keys := set(hover_props) - set(df_ptable):
                raise ValueError(
                    f"Unsupported hover_props: {', '.join(unsupported_keys)}. Available"
                    f" keys are: {', '.join(df_ptable)}.\nNote that some keys have "
                    "missing values."
                )
            df_row = df_ptable.loc[symbol]
            if isinstance(hover_props, dict):
                for col_name, col_label in hover_props.items():
                    hover_text += f"<br>{col_label} = {df_row[col_name]}"
            elif isinstance(hover_props, (list, tuple)):
                hover_text += "<br>" + "<br>".join(
                    f"{col_name} = {df_row[col_name]}" for col_name in hover_props
                )
            else:
                raise ValueError(
                    f"hover_props must be dict or sequence of str, got {hover_props}"
                )

        hover_texts[row][col] = hover_text

        # TODO maybe there's a more elegant way to handle excluded elements?
        if symbol in exclude_elements:
            continue

        color_val = heat_value_element_map[symbol]
        if log and color_val > 0:
            color_val = np.log10(color_val)
        # until https://github.com/plotly/plotly.js/issues/975 is resolved, we need to
        # insert transparency (rgba0) at low end of colorscale (+1e-6) to not show any
        # colors on empty tiles of the periodic table
        heatmap_values[row][col] = color_val

    if isinstance(font_colors, str):
        font_colors = [font_colors]
    if cscale_range == (None, None):
        cscale_range = (values.min(), values.max())

    fig = ff.create_annotated_heatmap(
        heatmap_values,
        annotation_text=tile_texts,
        text=hover_texts,
        showscale=show_scale,
        colorscale=colorscale,
        font_colors=font_colors or None,
        hoverinfo="text",
        xgap=gap,
        ygap=gap,
        zmin=None if log else cscale_range[0],
        zmax=None if log else cscale_range[1],
        # zauto=False if cscale_range is set, needed for zmin, zmax to work
        # see https://github.com/plotly/plotly.py/issues/193
        zauto=cscale_range == (None, None),
        **kwargs,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10, pad=10),
        paper_bgcolor=bg_color,
        plot_bgcolor="rgba(0, 0, 0, 0)",
        xaxis=dict(zeroline=False, showgrid=False),
        yaxis=dict(zeroline=False, showgrid=False, scaleanchor="x"),
        font_size=font_size,
        width=1000,
        height=500,
    )

    if color_bar.get("orientation") == "h":
        dct = dict(x=0.4, y=0.75, titleside="top", len=0.4)
        color_bar = {**dct, **color_bar}
    else:  # make title vertical
        dct = dict(titleside="right", len=0.87)
        color_bar = {**dct, **color_bar}
        if title := color_bar.get("title"):
            # <br><br> to increase title offset
            color_bar["title"] = f"<br><br>{title}"

    fig.update_traces(colorbar=dict(lenmode="fraction", thickness=15, **color_bar))
    return fig


def ptable_hists(
    data: pd.DataFrame | pd.Series | dict[str, list[float]],
    bins: int = 20,
    colormap: str | None = None,
    hist_kwds: dict[str, Any]
    | Callable[[Sequence[float]], dict[str, Any]]
    | None = None,
    cbar_coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.02),
    x_range: tuple[float | None, float | None] | None = None,
    symbol_kwargs: Any = None,
    symbol_text: str | Callable[[Element], str] = lambda elem: elem.symbol,
    cbar_title: str = "Values",
    cbar_title_kwds: dict[str, Any] | None = None,
    cbar_kwds: dict[str, Any] | None = None,
    symbol_pos: tuple[float, float] = (0.5, 0.8),
    log: bool = False,
    anno_kwds: dict[str, Any] | None = None,
    on_empty: Literal["show", "hide"] = "hide",
    color_elem_types: Literal["symbol", "background", "both", False]
    | dict[str, str] = "background",
    elem_type_legend: bool | dict[str, Any] = True,
    **kwargs: Any,
) -> plt.Figure:
    """Plot small histograms for each element laid out in a periodic table.

    Args:
        data (pd.DataFrame | pd.Series | dict[str, list[float]]): Map from element
            symbols to histogram values. E.g. if dict, {"Fe": [1, 2, 3], "O": [4, 5]}.
            If pd.Series, index is element symbols and values lists. If pd.DataFrame,
            column names are element symbols histograms are plotted from each column.
        bins (int): Number of bins for the histograms. Defaults to 20.
        colormap (str): Matplotlib colormap name to use. Defaults to None. See options
            at https://matplotlib.org/stable/users/explain/colors/colormaps.
        hist_kwds (dict | Callable): Keywords passed to ax.hist() for each histogram.
            If callable, it is called with the histogram values for each element and
            should return a dict of keyword arguments. Defaults to None.
        cbar_coords (tuple[float, float, float, float]): Color bar position and size:
            [x, y, width, height] anchored at lower left corner of the bar. Defaults to
            (0.25, 0.77, 0.35, 0.02).
        x_range (tuple[float | None, float | None]): x-axis range for all histograms.
            Defaults to None.
        symbol_text (str | Callable[[Element], str]): Text to display for each element
            symbol. Defaults to lambda elem: elem.symbol.
        symbol_kwargs (dict): Keyword arguments passed to plt.text() for element
            symbols. Defaults to None.
        cbar_title (str): Color bar title. Defaults to "Histogram Value".
        cbar_title_kwds (dict): Keyword arguments passed to cbar.ax.set_title().
            Defaults to dict(fontsize=12, pad=10).
        cbar_kwds (dict): Keyword arguments passed to fig.colorbar().
        symbol_pos (tuple[float, float]): Position of element symbols relative to the
            lower left corner of each tile. Defaults to (0.5, 0.8). (1, 1) is the upper
            right corner.
        log (bool): Whether to log scale y-axis of each histogram. Defaults to False.
        anno_kwds (dict): Keyword arguments passed to plt.annotate() for element
            annotations. Defaults to None. Useful for adding e.g. number of data points
            in each histogram. For that, use
            anno_kwds=lambda hist_vals: dict(text=len(hist_vals)).
            Recognized keys are text, xy, xycoords, fontsize, and any other
            plt.annotate() keywords.
        on_empty ('hide' | 'show'): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        color_elem_types ('symbol' | 'background' | 'both' | False | dict): Whether to
            color element symbols, tile backgrounds, or both based on element type.
            If dict, it should map element types to colors. Defaults to "background".
        elem_type_legend (bool | dict): Whether to show a legend for element
            types. Defaults to True. If dict, used as kwargs to plt.legend(), e.g. to
            set the legend title, use {"title": "Element Types"}.
        **kwargs: Additional keyword arguments passed to plt.subplots(). Defaults to
            dict(figsize=(0.75 * n_columns, 0.75 * n_rows)) with n_columns/n_rows the
            number of columns/rows in the periodic table.

    Returns:
        plt.Figure: periodic table with a histogram in each element tile.
    """
    n_rows = df_ptable.row.max()
    n_columns = df_ptable.column.max()

    kwargs.setdefault("figsize", (0.75 * n_columns, 0.75 * n_rows))
    fig, axes = plt.subplots(n_rows, n_columns, **kwargs)

    # Use series name as color bar title if available if no title was passed
    if isinstance(data, pd.Series) and cbar_title == "Values" and data.name:
        cbar_title = data.name
        data = data.to_dict()

    elif isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="list")

    if x_range is not None:
        vmin, vmax = x_range
    else:
        flat_list = [
            val
            for sublist in (data.values() if isinstance(data, dict) else data)
            for val in sublist
        ]
        vmin, vmax = min(flat_list), max(flat_list)
    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = None
    if colormap:
        cmap = plt.get_cmap(colormap) if isinstance(colormap, str) else colormap

    # Turn off axis of subplots on the grid that don't correspond to elements
    ax: plt.Axes
    for ax in axes.flat:
        ax.axis("off")

    elem_class_colors = ELEM_CLASS_COLORS | (
        color_elem_types if isinstance(color_elem_types, dict) else {}
    )

    symbol_kwargs = symbol_kwargs or {}
    for element in Element:
        symbol = element.symbol
        row, group = df_ptable.loc[symbol, ["row", "column"]]

        ax = axes[row - 1][group - 1]
        symbol_kwargs.setdefault("fontsize", 10)
        hist_data = data.get(symbol, [])

        if len(hist_data) == 0 and on_empty == "hide":
            continue

        if color_elem_types:
            elem_class = df_ptable.loc[symbol, "type"]
            if color_elem_types in ("symbol", "both"):
                symbol_kwargs["color"] = elem_class_colors.get(elem_class, "black")
            if color_elem_types in ("background", "both"):
                bg_color = elem_class_colors.get(elem_class, "white")
                ax.set_facecolor((*mpl.colors.to_rgb(bg_color), 0.07))

        ax.text(
            *symbol_pos,
            symbol_text(element)
            if callable(symbol_text)
            else symbol_text.format(elem=element),
            ha="center",
            va="center",
            transform=ax.transAxes,
            **symbol_kwargs,
        )
        ax.axis("on")  # re-enable axes of elements that exist

        if anno_kwds:
            defaults = dict(
                text=lambda hist_vals: si_fmt_int(len(hist_vals)),
                xy=(0.8, 0.8),
                xycoords="axes fraction",
                fontsize=8,
                horizontalalignment="center",
                verticalalignment="center",
            )
            if callable(anno_kwds):
                annotation = anno_kwds(hist_data)
            else:
                annotation = anno_kwds
                anno_text = anno_kwds.get("text")
                if isinstance(anno_text, dict):
                    anno_text = anno_text.get(symbol)
                elif callable(anno_text):
                    anno_text = anno_text(hist_data)
                annotation["text"] = anno_text
            ax.annotate(**(defaults | annotation))

        if hist_data is not None:
            hist_kwargs = hist_kwds(hist_data) if callable(hist_kwds) else hist_kwds
            _n, bins_array, patches = ax.hist(
                hist_data, bins=bins, log=log, range=x_range, **(hist_kwargs or {})
            )
            if x_range:
                ax.set_xlim(x_range)
            x_min, x_max = math.floor(min(bins_array)), math.ceil(max(bins_array))
            x_ticks = list(x_range or [x_min, x_max])
            if x_ticks[0] is None:
                x_ticks[0] = x_min
            if x_ticks[1] is None:
                x_ticks[1] = x_max
            if x_min < 0 < x_max:
                # make sure we always show a mark at 0
                x_ticks.insert(1, 0)

            if cmap:
                for patch, x_val in zip(patches, bins_array[:-1]):
                    plt.setp(patch, "facecolor", cmap(norm(x_val)))
            ax.set_xticks(x_ticks)
            ax.tick_params(labelsize=8, direction="in")
        else:  # disable ticks for elements without data
            ax.set_xticks([])
        ax.set_yticks([])  # disable y ticks for all elements

        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(b=False)
        # Hide y ticks
        ax.tick_params(axis="y", which="both", length=0)

    # Add color bar
    if isinstance(cmap, Colormap):
        cbar_ax = fig.add_axes(cbar_coords)
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            **{"orientation": "horizontal"} | (cbar_kwds or {}),
        )
        # Set color bar title
        cbar_title_kwds = cbar_title_kwds or {}
        cbar_title_kwds.setdefault("fontsize", 12)
        cbar_title_kwds.setdefault("pad", 10)
        cbar_title_kwds["label"] = cbar_title
        cbar_ax.set_title(**cbar_title_kwds)

    if elem_type_legend and color_elem_types:
        legend_kwargs = elem_type_legend if isinstance(elem_type_legend, dict) else {}
        add_element_type_legend(
            data=data, elem_class_colors=elem_class_colors, legend_kwargs=legend_kwargs
        )

    return fig


def ptable_scatters(
    data: pd.DataFrame | pd.Series | dict[str, list[list[float]]],
    symbol_text: str | Callable[[Element], str] = lambda elem: elem.symbol,
    symbol_pos: tuple[float, float] = (0.5, 0.8),
    on_empty: Literal["hide", "show"] = "hide",
    hide_f_block: bool | None = None,
    plot_kwargs: dict[str, Any]
    | Callable[[Sequence[float]], dict[str, Any]]
    | None = None,
    child_args: dict[str, Any] | None = None,
    ax_kwargs: dict[str, Any] | None = None,
    symbol_kwargs: dict[str, Any] | None = None,
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
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html#matplotlib-axes-axes-set
        symbol_text (str | Callable[[Element], str]): Text to display for
            each element symbol. Defaults to lambda elem: elem.symbol.
        symbol_kwargs (dict): Keyword arguments passed to plt.text() for
            element symbols. Defaults to None.
        symbol_pos (tuple[float, float]): Position of element symbols
            relative to the lower left corner of each tile.
            Defaults to (0.5, 0.5). (1, 1) is the upper right corner.
        on_empty ('hide' | 'show'): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool): Hide f-block (Lanthanum and Actinium series). Defaults to
            None, meaning hide if no data is provided for f-block elements.
        child_args: Arguments to pass to the child plotter call.
        plot_kwargs (dict): Additional keyword arguments to
                pass to the plt.subplots function call.

    TODO: allow colormap with 3rd data dimension
    """
    # Re-initialize kwargs as empty dict if None
    plot_kwargs = plot_kwargs or {}
    ax_kwargs = ax_kwargs or {}

    child_args = child_args or {}

    symbol_kwargs = symbol_kwargs or {}
    symbol_kwargs.setdefault("fontsize", 12)

    # Initialize periodic table plotter
    plotter = PTableProjector(
        data=data,
        colormap=None,
        plot_kwargs=plot_kwargs,  # type: ignore[arg-type]
        hide_f_block=hide_f_block,
    )

    # Call child plotter: Scatter
    plotter.add_child_plots(
        ChildPlotters.scatter,
        child_args=child_args,
        ax_kwargs=ax_kwargs,
        on_empty=on_empty,
    )

    # Add element symbols
    plotter.add_ele_symbols(
        text=symbol_text,
        pos=symbol_pos,
        kwargs=symbol_kwargs,
    )

    return plotter.fig


def ptable_lines(
    data: pd.DataFrame | pd.Series | dict[str, list[list[float]]],
    symbol_text: str | Callable[[Element], str] = lambda elem: elem.symbol,
    symbol_pos: tuple[float, float] = (0.5, 0.8),
    on_empty: Literal["hide", "show"] = "hide",
    hide_f_block: bool | None = None,
    plot_kwargs: dict[str, Any]
    | Callable[[Sequence[float]], dict[str, Any]]
    | None = None,
    child_args: dict[str, Any] | None = None,
    ax_kwargs: dict[str, Any] | None = None,
    symbol_kwargs: dict[str, Any] | None = None,
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
        ax_kwargs (dict): Keyword arguments passed to ax.set() for each plot.
            Use to set x/y labels, limits, etc. Defaults to None. Example:
            dict(title="Periodic Table", xlabel="x-axis", ylabel="y-axis", xlim=(0, 10),
            ylim=(0, 10), xscale="linear", yscale="log"). See ax.set() docs for options:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set.html#matplotlib-axes-axes-set
        symbol_text (str | Callable[[Element], str]): Text to display for
            each element symbol. Defaults to lambda elem: elem.symbol.
        symbol_kwargs (dict): Keyword arguments passed to plt.text() for
            element symbols. Defaults to None.
        symbol_pos (tuple[float, float]): Position of element symbols
            relative to the lower left corner of each tile.
            Defaults to (0.5, 0.5). (1, 1) is the upper right corner.
        on_empty ('hide' | 'show'): Whether to show or hide tiles for elements without
            data. Defaults to "hide".
        hide_f_block (bool): Hide f-block (Lanthanum and Actinium series). Defaults to
            None, meaning hide if no data is provided for f-block elements.
        child_args: Arguments to pass to the child plotter call.
        plot_kwargs (dict): Additional keyword arguments to
                pass to the plt.subplots function call.
    """
    # Re-initialize kwargs as empty dict if None
    plot_kwargs = plot_kwargs or {}
    ax_kwargs = ax_kwargs or {}

    child_args = child_args or {}

    symbol_kwargs = symbol_kwargs or {}
    symbol_kwargs.setdefault("fontsize", 12)

    # Initialize periodic table plotter
    plotter = PTableProjector(
        data=data,
        colormap=None,
        plot_kwargs=plot_kwargs,  # type: ignore[arg-type]
        hide_f_block=hide_f_block,
    )

    # Call child plotter: line
    plotter.add_child_plots(
        ChildPlotters.line,
        child_args=child_args,
        ax_kwargs=ax_kwargs,
        on_empty=on_empty,
    )

    # Add element symbols
    plotter.add_ele_symbols(
        text=symbol_text,
        pos=symbol_pos,
        kwargs=symbol_kwargs,
    )

    return plotter.fig
