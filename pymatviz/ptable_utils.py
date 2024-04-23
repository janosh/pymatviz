"""Utils for periodic table plotters."""

import warnings
import itertools
from collections.abc import Sequence
from typing import Literal, Union, Sequence, get_args, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pymatgen.core import Composition, Element
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import Rectangle

from pymatviz.utils import df_ptable

if TYPE_CHECKING:
    from typing import Any, TypeAlias


# Data types supported by ptable plotters
SupportedValueType: TypeAlias = Union[Sequence[float], np.ndarray]

SupportedDataType: TypeAlias = Union[
    dict[str, Union[float, Sequence[float], np.ndarray]], pd.DataFrame, pd.Series
]

CountMode: TypeAlias = Literal[
    "composition", "fractional_composition", "reduced_composition", "occurrence"
]

ElemValues: TypeAlias = dict[str | int, float] | pd.Series | Sequence[str]


def count_elements(
    values: ElemValues,
    count_mode: CountMode = "composition",
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
            attr = "element_composition" if count_mode == "composition" else count_mode
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

    TODO: add and test imputation and anomaly handling

    Returns:
        pd.DataFrame: The preprocessed DataFrame with element names
            as index and values as columns.

    Example:
        >>> data: dict = {"H": 1.0, "He": [2.0, 4.0]}

        OR
        >>> data: pd.DataFrame = pd.DataFrame(
            {"H": 1.0, "He": [2.0, 4.0]}.items(),
            columns=["Element", "Value"]
            ).set_index("Element")

        OR
        >>> data: pd.Series = pd.Series({"H": 1.0, "He": [2.0, 4.0]})

        >>> preprocess_data(data)

             Element   Value
        0    H         [1.0, ]
        1    He        [2.0, 4.0]

        Metadata:
            vmin: 1.0
            vmax: 4.0
    """
    if isinstance(data, pd.DataFrame):
        data_df = data

    elif isinstance(data, pd.Series):
        data_df = data.to_frame(name="Value")
        data_df.index.name = "Element"

    elif isinstance(data, dict):
        data_df = pd.DataFrame(data.items(), columns=["Element", "Value"]).set_index(
            "Element"
        )

    else:
        raise TypeError(f"Unsupported data type, choose from: {SupportedDataType}.")

    # Convert all values to np.array
    data_df["Value"] = data_df["Value"].map(
        lambda x: np.array([x]) if isinstance(x, float) else np.array(x)
    )

    # Get and write vmin/vmax into metadata
    flattened_values: list[float] = [
        item for sublist in data_df["Value"] for item in sublist
    ]
    try:
        data_df.attrs["vmin"] = min(flattened_values)
        data_df.attrs["vmax"] = max(flattened_values)

    except ValueError:
        # Let it pass to test plotter
        data_df.attrs["vmin"] = 0
        data_df.attrs["vmax"] = 1
        # DEBUG: method failed for nested arrays (line plotter)
        warnings.warn("Normalization failed.")

    return data_df


class PTableProjector:
    """Project (nest) a custom plot into a periodic table.

    TODO: clarify scope of "plot/ax/child"
    """

    def __init__(
        self,
        data: SupportedDataType,
        colormap: str | Colormap | None,
        **kwargs: Any,
    ) -> None:
        """Initialize a ptable projector.

        Default figsize is set to (0.75 * n_groups, 0.75 * n_periods),
        and axes would be turned off by default.

        Args:
            data (SupportedDataType): The data to be visualized.
            colormap (str | Colormap | None): The colormap to use.
            **kwargs (Any): Additional keyword arguments to
                pass to the plt.subplots function call.
        """
        # Get colormap
        self.cmap: Colormap = colormap

        # Preprocess data
        self.data: pd.DataFrame = data

        # Initialize periodic table canvas
        n_periods = df_ptable.row.max()
        n_groups = df_ptable.column.max()

        # Set figure size
        kwargs.setdefault("figsize", (0.75 * n_groups, 0.75 * n_periods))

        self.fig, self.axes = plt.subplots(n_periods, n_groups, **kwargs)

        # Turn off all axes
        for ax in self.axes.flat:
            ax.axis("off")

    @property
    def cmap(self) -> Colormap | None:
        """The global Colormap.

        Returns:
            Colormap: The Colormap used.
        """
        return self._cmap

    @cmap.setter
    def cmap(self, colormap: str | Colormap | None) -> None:
        """The global colormap used.

        Args:
        colormap (str | Colormap | None): The colormap to use.
        """
        self._cmap = None if colormap is None else plt.get_cmap(colormap)

    @property
    def data(self) -> pd.DataFrame:
        """The preprocessed data.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        return self._data

    @property
    def norm(self) -> Normalize:
        """Data min-max normalizer."""
        return self._norm

    @data.setter
    def data(self, data: SupportedDataType) -> None:
        """Set and preprocess the data, also set normalizer.

        Args:
            data (SupportedDataType): The data to be used.
        """
        # Preprocess data
        self._data: pd.DataFrame = data_preprocessor(data)

        # Normalize data for colorbar
        self._norm: Normalize = Normalize(
            vmin=self._data.attrs["vmin"], vmax=self._data.attrs["vmax"]
        )

    def add_child_plots(
        self,
        child_plotter: Callable[[plt.axes, Any], None],
        child_args: dict[str, Any],
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
            # Get axis index by element symbol
            symbol: str = element.symbol
            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax: plt.Axes = self.axes[row - 1][column - 1]

            # Get and check tile data
            plot_data: np.ndarray | Sequence[float] = self.data.loc[symbol, "Value"]
            if len(plot_data) == 0 and on_empty == "hide":
                continue

            # Call child plotter
            if len(plot_data) > 0:
                child_plotter(ax, plot_data, child_args)

            # Pass axis kwargs
            if ax_kwargs:
                ax.set(**ax_kwargs)

    def add_ele_symbols(
        self,
        text: Callable[[Element], str] = lambda elem: elem.symbol,
        pos: tuple[float, float] = (0.5, 0.5),
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Add element symbols for each tile.

        Args:
            text: A callable or string specifying how to display
                the element symbol. If a callable is provided,
                it should accept an Element object and return a string.
                If a string is provided, it can contain a format
                specifier for the element symbol, e.g., "{elem.symbol}".
            pos: The position of the text relative to the axes.
            kwargs: Additional keyword arguments to pass to the `ax.text`.
        """
        # Update symbol args
        kwargs = kwargs or {}
        kwargs.setdefault("fontsize", 18)

        # Add symbol for each element
        for element in Element:
            # Get axis index by element symbol
            symbol: str = element.symbol
            row, column = df_ptable.loc[symbol, ["row", "column"]]
            ax: plt.Axes = self.axes[row - 1][column - 1]

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
    """Collect some pre-defined child plotters.

    TODO: add instruction for adding custom plotters.
    """

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
        child_args: dict[str, Any],
    ) -> None:
        """Scatter plotter.

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for to
                the child plotter.
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
        child_args: dict[str, Any],
    ) -> None:
        """Line plotter.

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for to
                the child plotter.
        """
        # Add line
        ax.plot(data[0], data[1], **child_args)

        # Adjust tick labels
        # TODO: how to achieve this from external?
        ax.tick_params(axis="both", which="major", labelsize=8)

        # Hide the right and top spines
        ax.axis("on")  # turned off by default
        ax.spines[["right", "top"]].set_visible(False)

    @staticmethod
    def hist(
        ax: plt.axes,
        data: SupportedValueType,
        child_args: dict[str, Any],
    ) -> None:
        """Histogram plotter.

        Args:
            ax (plt.axes): The axis to plot on.
            data (SupportedValueType): The values for to
                the child plotter.
        """
        ax.hist(data, **child_args)
