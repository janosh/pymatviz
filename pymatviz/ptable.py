from __future__ import annotations

import itertools
import math
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal, get_args

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

from pymatviz.utils import df_ptable, pick_bw_for_contrast, si_fmt, si_fmt_int


if TYPE_CHECKING:
    from typing import TypeAlias

    import plotly.graph_objects as go

    ElemValues: TypeAlias = dict[str | int, int | float] | pd.Series | Sequence[str]

CountMode = Literal[
    "composition", "fractional_composition", "reduced_composition", "occurrence"
]


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


def ptable_heatmap(
    values: ElemValues,
    log: bool | Normalize = False,
    ax: plt.Axes | None = None,
    count_mode: CountMode = "composition",
    cbar_title: str = "Element Count",
    cbar_range: tuple[float | None, float | None] | None = None,
    cbar_coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.05),
    cbar_kwargs: dict[str, Any] | None = None,
    colorscale: str = "viridis",
    infty_color: str = "lightskyblue",
    na_color: str = "white",
    heat_mode: Literal["value", "fraction", "percent"] | None = "value",
    fmt: str | Callable[..., str] | None = None,
    cbar_fmt: str | Callable[..., str] | None = None,
    text_color: str | tuple[str, str] = "auto",
    exclude_elements: Sequence[str] = (),
    zero_color: str = "#eff",  # light gray
    zero_symbol: str | float = "-",
    label_font_size: int = 16,
    value_font_size: int = 12,
    tile_size: float | tuple[float, float] = 0.9,
    rare_earth_voffset: float = 0.5,
    **kwargs: Any,
) -> plt.Axes:
    """Plot a heatmap across the periodic table of elements.

    Args:
        values (dict[str, int | float] | pd.Series | list[str]): Map from element
            symbols to heatmap values or iterable of composition strings/objects.
        log (bool | Normalize, optional): Whether color map scale is log or linear. Can
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
        cbar_fmt (str): f-string format option to set a different colorbar tick
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
        label_font_size (int): Font size for element symbols. Defaults to 16.
        value_font_size (int): Font size for heat values. Defaults to 12.
        tile_size (float | tuple[float, float]): Size of each tile in the periodic
            table as a fraction of available space before touching neighboring tiles.
            1 or (1, 1) means no gaps between tiles. Defaults to 0.9.
        cbar_coords (tuple[float, float, float, float]): Color bar position and size:
            [x, y, width, height] anchored at lower left corner of the bar. Defaults to
            (0.18, 0.8, 0.42, 0.05).
        rare_earth_voffset (float): Vertical offset for lanthanides and actinides
            (row 6 and 7) from the rest of the periodic table. Defaults to 0.5.
        **kwargs: Additional keyword arguments passed to plt.figure().

    Returns:
        ax: matplotlib Axes with the heatmap.
    """
    if fmt is None:
        fmt = lambda x, _: si_fmt(x, ".1%" if heat_mode == "percent" else ".0f")
    if cbar_fmt is None:
        cbar_fmt = fmt

    valid_logs = (bool, Normalize)
    if not isinstance(log, valid_logs):
        raise ValueError(f"Invalid {log=}, must be instance of {valid_logs}")

    if log and heat_mode in ("fraction", "percent"):
        raise ValueError(
            "Combining log color scale and heat_mode='fraction'/'percent' unsupported"
        )
    if "cmap" in kwargs:
        colorscale = kwargs.pop("cmap")
        print("cmap argument is deprecated, use colorscale instead.", file=sys.stderr)

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
    )

    for symbol, row, column, *_ in df_ptable.itertuples():
        row = n_rows - row  # invert row count to make periodic table right side up
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
                # 2nd arg=0 just for consistency with matplotlib fmt signature
                label = fmt(tile_value, 0)
            elif heat_mode == "percent":
                label = f"{tile_value:{fmt or '.1f'}}"
            else:
                fmt = fmt or (".0f" if tile_value > 100 else ".1f")
                label = f"{tile_value:{fmt}}"
            # replace shortens scientific notation 1e+01 to 1e1 so it fits inside cells
            label = label.replace("e+0", "e")
        if row < 3:  # vertical offset for lanthanides + actinides
            row += rare_earth_voffset
        rect = Rectangle(
            (column, row), tile_width, tile_height, edgecolor="gray", facecolor=color
        )

        if heat_mode is None:
            # no value to display below in colored rectangle so center element symbol
            text_style["verticalalignment"] = "center"

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

        plt.text(
            column + 0.5 * tile_width,
            row + 0.5 * tile_height,
            symbol,
            color=text_clr,
            **text_style,
        )

        if heat_mode is not None:
            plt.text(
                column + 0.5 * tile_width,
                row + 0.1 * tile_height,
                label,
                fontsize=value_font_size,
                horizontalalignment="center",
                color=text_clr,
            )

        ax.add_patch(rect)

    if heat_mode is not None:
        # color bar position and size: [x, y, width, height]
        # anchored at lower left corner
        cbar_ax = ax.inset_axes(cbar_coords, transform=ax.transAxes)
        # format major and minor ticks
        cbar_ax.tick_params(which="both", labelsize=14, width=1)

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=colorscale)

        def tick_fmt(val: float, _pos: int) -> str:
            # val: value at color axis tick (e.g. 10.0, 20.0, ...)
            # pos: zero-based tick counter (e.g. 0, 1, 2, ...)
            default_fmt = (
                ".0%" if heat_mode == "percent" else (".0f" if val < 1e4 else ".2g")
            )
            return f"{val:{cbar_fmt or fmt or default_fmt}}"

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


def ptable_heatmap_ratio(
    values_num: ElemValues,
    values_denom: ElemValues,
    count_mode: CountMode = "composition",
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
        cbar_title (str): Title for the color bar. Defaults to "Element Ratio".
        not_in_numerator (tuple[str, str]): Color and legend description used for
            elements missing from numerator. Defaults to
            ('#eff', 'gray: not in 1st list').
        not_in_denominator (tuple[str, str]): See not_in_numerator. Defaults to
            ('lightskyblue', 'blue: not in 2nd list').
        not_in_either (tuple[str, str]): See not_in_numerator. Defaults to
            ('white', 'white: not in either').
        **kwargs: Additional keyword arguments passed to ptable_heatmap().

    Returns:
        ax: The plot's matplotlib Axes.
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
    count_mode: CountMode = "composition",
    colorscale: str | Sequence[str] | Sequence[tuple[float, str]] = "viridis",
    showscale: bool = True,
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
    label_map: dict[str, str] | Literal[False] | None = None,
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
        showscale (bool): Whether to show a bar for the color scale. Defaults to True.
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
        color_bar (dict[str, Any]): Plotly color bar properties documented at
            https://plotly.com/python/reference#heatmap-colorbar. Defaults to
            dict(orientation="h"). Commonly used keys are:
            - title: color bar title
            - titleside: "top" | "bottom" | "right" | "left"
            - tickmode: "array" | "auto" | "linear" | "log" | "date" | "category"
            - tickvals: list of tick values
            - ticktext: list of tick labels
            - tickformat: f-string format option for tick labels
            - len: fraction of plot height or width depending on orientation
            - thickness: fraction of plot height or width depending on orientation
        cscale_range (tuple[float | None, float | None]): Color bar range. Defaults to
            (None, None) meaning the range is automatically determined from the data.
        exclude_elements (list[str]): Elements to exclude from the heatmap. E.g. if
            oxygen overpowers everything, you can do exclude_elements=['O'].
            Defaults to ().
        log (bool): Whether to use a logarithmic color scale. Defaults to False.
            Piece of advice: colorscale='viridis' and log=True go well together.
        fill_value (float | None): Value to fill in for missing elements. Defaults to 0.
        label_map (dict[str, str] | None): Map heat values (after string formatting)
            to target strings. Defaults to dict.fromkeys((np.nan, None, "nan"), " ")
            so as not to display 'nan' for missing values. Set to False to disable.
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
    elif isinstance(colorscale, Sequence) and isinstance(
        colorscale[0], (str, list, tuple)
    ):
        pass
    else:
        raise ValueError(
            f"{colorscale = } should be string, list of strings or list of "
            "tuples(float, str)"
        )

    color_bar = color_bar or {}
    color_bar.setdefault("orientation", "h")
    # if values is a series with a name, use it as the color bar title
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

        label = None  # label (if not None) is placed below the element symbol
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

        style = f"font-weight: bold; font-size: {1.5 * (font_size or 12)};"
        tile_text = (
            f"<span {style=}>{symbol}</span><br>"
            f"{(label_map or {}).get(label, label)}"  # type: ignore[arg-type]
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
        showscale=showscale,
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
    return_axes: bool = False,
    **kwargs: Any,
) -> plt.Figure:
    """Plot histograms of values across the periodic table of elements.

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
        return_axes (bool): Whether to return the matplotlib Figure and Axes objects.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to plt.subplots().
            figsize is set to (0.75 * n_columns, 0.75 * n_rows) where n_columns and
            n_rows are the number of columns and rows in the periodic table.

    Returns:
        plt.Figure: periodic table with a histogram in each element tile.
    """
    n_rows = df_ptable.row.max()
    n_columns = df_ptable.column.max()

    kwargs.setdefault("figsize", (0.75 * n_columns, 0.75 * n_rows))
    fig, axes = plt.subplots(n_rows, n_columns, **kwargs)

    if isinstance(data, pd.Series):
        # use series name as color bar title if available and no title was passed
        if cbar_title == "Values" and data.name:
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

    # turn off axis of subplots on the grid that don't correspond to elements
    ax: plt.Axes
    for ax in axes.flat:
        ax.axis("off")

    symbol_kwargs = symbol_kwargs or {}
    for Z in range(1, 119):
        element = Element.from_Z(Z)
        symbol = element.symbol
        row, group = df_ptable.loc[symbol, ["row", "column"]]

        ax = axes[row - 1][group - 1]
        symbol_kwargs.setdefault("fontsize", 10)
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

        hist_data = data.get(symbol, [])
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
        if hist_data:
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
            ax.spines[side].set_visible(False)
        # also hide tick marks
        ax.tick_params(axis="y", which="both", length=0)

    # add colorbar
    if isinstance(cmap, Colormap):
        cbar_ax = fig.add_axes(cbar_coords)
        _cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            **{"orientation": "horizontal"} | (cbar_kwds or {}),
        )
        # set color bar title
        cbar_title_kwds = cbar_title_kwds or {}
        cbar_title_kwds.setdefault("fontsize", 12)
        cbar_title_kwds.setdefault("pad", 10)
        cbar_title_kwds["label"] = cbar_title
        cbar_ax.set_title(**cbar_title_kwds)

    if return_axes:
        return fig, axes
    return fig
