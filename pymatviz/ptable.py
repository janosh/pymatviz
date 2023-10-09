from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pymatgen.core import Composition

from pymatviz.utils import df_ptable


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
        raise ValueError(f"{count_mode=} must be one of {get_args(CountMode)=}")
    # ensure values is Series if we got dict/list/tuple
    srs = pd.Series(values)

    if is_numeric_dtype(srs):
        pass
    elif is_string_dtype(srs):
        # assume all items in values are composition strings
        if count_mode == "occurrence":
            srs = pd.Series(
                itertools.chain.from_iterable(
                    map(str, Composition(comp)) for comp in srs
                )
            ).value_counts()
        else:
            attr = "element_composition" if count_mode == "composition" else count_mode
            srs = pd.DataFrame(
                getattr(Composition(formula), attr).as_dict() for formula in srs
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
    log: bool = False,
    ax: plt.Axes | None = None,
    count_mode: CountMode = "composition",
    cbar_title: str = "Element Count",
    cbar_max: float | None = None,
    cmap: str = "summer_r",
    zero_color: str = "#DDD",  # light gray
    infty_color: str = "lightskyblue",
    na_color: str = "white",
    heat_mode: Literal["value", "fraction", "percent"] | None = "value",
    fmt: str | None = None,
    cbar_fmt: str | None = None,
    text_color: str | tuple[str, str] = "auto",
    exclude_elements: Sequence[str] = (),
    zero_symbol: str | float = "-",
) -> plt.Axes:
    """Plot a heatmap across the periodic table of elements.

    Args:
        values (dict[str, int | float] | pd.Series | list[str]): Map from element
            symbols to heatmap values or iterable of composition strings/objects.
        log (bool, optional): Whether color map scale is log or linear.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        count_mode ('composition' | 'fractional_composition' | 'reduced_composition'):
            Reduce or normalize compositions before counting. See count_elements() for
            details. Only used when values is list of composition strings/objects.
        cbar_title (str, optional): Title for colorbar. Defaults to "Element Count".
        cbar_max (float, optional): Maximum value of the colorbar range. Will be ignored
            if smaller than the largest plotted value. For creating multiple plots with
            identical color bars for visual comparison. Defaults to 0.
        cmap (str, optional): Matplotlib colormap name to use. Defaults to "YlGn".
        zero_color (str): Color to use for elements with value zero. Defaults to "#DDD"
            (light gray).
        infty_color: Color to use for elements with value infinity. Defaults to
            "lightskyblue".
        na_color: Color to use for elements with value infinity. Defaults to "white".
        heat_mode ("value" | "fraction" | "percent" | None): Whether to display heat
            values as is, normalized as a fraction of the total, as percentages
            or not at all (None). Defaults to "value".
            "fraction" and "percent" can be used to make the colors in different
            ptable_heatmap() (and ptable_heatmap_ratio()) plots comparable.
        fmt (str): f-string format option for tile values. Defaults to ".1%"
            (1 decimal place) if heat_mode="percent" else ".3g".
        cbar_fmt (str): f-string format option to set a different colorbar tick
            label format. Defaults to the above fmt.
        text_color (str | tuple[str, str]): What color to use for element symbols and
            heat labels. Must be a valid color name, or a 2-tuple of names, one to use
            for the upper half of the color scale, one for the lower half. The special
            value 'auto' applies 'black' on the lower and 'white' on the upper half of
            the color scale. Defaults to "auto".
        exclude_elements (list[str]): Elements to exclude from the heatmap. E.g. if
            oxygen overpowers everything, you can try log=True or
            exclude_elements=['O']. Defaults to ().
        zero_symbol (str | float): Symbol to use for elements with value zero.
            Defaults to "-".

    Returns:
        ax: matplotlib Axes with the heatmap.
    """
    if log and heat_mode in ("fraction", "percent"):
        raise ValueError(
            "Combining log color scale and heat_mode='fraction'/'percent' unsupported"
        )

    values = count_elements(values, count_mode, exclude_elements)

    # replace positive and negative infinities with NaN values, then drop all NaNs
    clean_vals = values.replace([np.inf, -np.inf], np.nan).dropna()

    if heat_mode in ("fraction", "percent"):
        # ignore inf values in sum() else all would be set to 0 by normalizing
        values /= clean_vals.sum()
        clean_vals /= clean_vals.sum()  # normalize as well for norm.autoscale() below

    color_map = get_cmap(cmap)

    n_rows = df_ptable.row.max()
    n_columns = df_ptable.column.max()

    # TODO can we pass as a kwarg and still ensure aspect ratio respected?
    fig = plt.figure(figsize=(0.75 * n_columns, 0.7 * n_rows))

    ax = ax or plt.gca()

    rw = rh = 0.9  # rectangle width/height

    norm = LogNorm() if log else Normalize()

    norm.autoscale(clean_vals.to_numpy())
    if cbar_max is not None:
        norm.vmax = cbar_max

    text_style = dict(horizontalalignment="center", fontsize=16, fontweight="semibold")

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

            if heat_mode == "percent":
                label = f"{tile_value:{fmt or '.1f'}}"
            else:
                fmt = fmt or (".0f" if tile_value > 100 else ".1f")
                label = f"{tile_value:{fmt}}"
            # replace shortens scientific notation 1e+01 to 1e1 so it fits inside cells
            label = label.replace("e+0", "e")
        if row < 3:  # vertical offset for lanthanide + actinide series
            row += 0.5
        rect = Rectangle((column, row), rw, rh, edgecolor="gray", facecolor=color)

        if heat_mode is None:
            # no value to display below in colored rectangle so center element symbol
            text_style["verticalalignment"] = "center"

        if symbol in exclude_elements:
            text_clr = "black"
        elif text_color == "auto":
            text_clr = "white" if norm(tile_value) > 0.5 else "black"
        elif isinstance(text_color, (tuple, list)):
            text_clr = text_color[0] if norm(tile_value) > 0.5 else text_color[1]
        else:
            text_clr = text_color

        plt.text(
            column + 0.5 * rw, row + 0.5 * rh, symbol, color=text_clr, **text_style
        )

        if heat_mode is not None:
            plt.text(
                column + 0.5 * rw,
                row + 0.1 * rh,
                label,
                fontsize=10,
                horizontalalignment="center",
                color=text_clr,
            )

        ax.add_patch(rect)

    if heat_mode is not None:
        # colorbar position and size: [x, y, width, height]
        # anchored at lower left corner
        cb_ax = ax.inset_axes([0.18, 0.8, 0.42, 0.05], transform=ax.transAxes)
        # format major and minor ticks
        cb_ax.tick_params(which="both", labelsize=14, width=1)

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        def tick_fmt(val: float, _pos: int) -> str:
            # val: value at color axis tick (e.g. 10.0, 20.0, ...)
            # pos: zero-based tick counter (e.g. 0, 1, 2, ...)
            default_prec = (
                ".0%" if heat_mode == "percent" else (".0f" if val < 1e4 else ".2g")
            )
            return f"{val:{cbar_fmt or fmt or default_prec}}"

        cbar = fig.colorbar(
            mappable, cax=cb_ax, orientation="horizontal", format=tick_fmt
        )

        cbar.outline.set_linewidth(1)
        cb_ax.set_title(cbar_title, pad=10, **text_style)

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
    not_in_numerator: tuple[str, str] = ("#DDD", "gray: not in 1st list"),
    not_in_denominator: tuple[str, str] = ("lightskyblue", "blue: not in 2nd list"),
    not_in_either: tuple[str, str] = ("white", "white: not in either"),
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
            ('#DDD', 'gray: not in 1st list').
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

    kwargs["zero_color"] = not_in_numerator[0]
    kwargs["infty_color"] = not_in_denominator[0]
    kwargs["na_color"] = not_in_either[0]

    ax = ptable_heatmap(values, cbar_title=cbar_title, **kwargs)

    # add legend handles
    for y_pos, color, txt in (
        (2.1, *not_in_numerator),
        (1.4, *not_in_denominator),
        (0.7, *not_in_either),
    ):
        bbox = dict(facecolor=color, edgecolor="gray")
        plt.text(0.8, y_pos, txt, fontsize=10, bbox=bbox)

    return ax


def ptable_heatmap_plotly(
    values: ElemValues,
    count_mode: CountMode = "composition",
    colorscale: str | Sequence[str] | Sequence[tuple[float, str]] = "viridis",
    showscale: bool = True,
    heat_mode: Literal["value", "fraction", "percent"] | None = "value",
    precision: str | None = None,
    hover_props: Sequence[str] | dict[str, str] | None = None,
    hover_data: dict[str, str | int | float] | pd.Series | None = None,
    font_colors: Sequence[str] = ("#eee", "black"),
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
        precision (str): f-string format option for heat labels. Defaults to ".1%"
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
            (max_val - min_val) / 2. Defaults to ("#eee", "black") meaning light text
            for low values and dark text for high values. May need to be manually
            swapped depending on the colorscale.
        gap (float): Gap in pixels between tiles of the periodic table. Defaults to 5.
        font_size (int): Element symbol and heat label text size. Any valid CSS size
            allowed. Defaults to automatic font size based on plot size. Element symbols
            will be bold and 1.5x this size.
        bg_color (str): Plot background color. Defaults to "rgba(0, 0, 0, 0)".
        color_bar (dict[str, Any]): Plotly color bar properties documented at
            https://plotly.com/python/reference#heatmap-colorbar. Defaults to
            dict(orientation="h").
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
        colorscale = px.colors.get_colorscale(colorscale or "Pinkyl")
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

    if log and values.dropna()[values != 0].min() <= 1:
        smaller_1 = values[values <= 1]
        raise ValueError(
            "Log color scale requires all heat map values to be > 1 since values <= 1 "
            f"map to negative log values which throws off the color scale. Got "
            f"{smaller_1.size} values <= 1: {list(smaller_1)}"
        )

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
        label_map = dict.fromkeys([np.nan, None, "nan"], " ")  # type: ignore

    for symbol, period, group, name, *_ in df_ptable.itertuples():
        # build table from bottom up so that period 1 becomes top row
        row = n_rows - period
        col = group - 1

        label = None  # label (if not None) is placed below the element symbol
        if symbol in exclude_elements:
            label = "excl."
        elif heat_value := heat_value_element_map.get(symbol):
            if heat_mode == "percent":
                label = f"{heat_value:{precision or '.1%'}}"
            else:
                default_prec = ".1f" if heat_value < 100 else ",.0f"
                if heat_value > 1e5:
                    default_prec = ".2g"
                label = f"{heat_value:{precision or default_prec}}".replace("e+0", "e")

        style = f"font-weight: bold; font-size: {1.5 * (font_size or 12)};"
        tile_text = (
            f"<span {style=}>{symbol}</span><br>"
            f"{(label_map or {}).get(label, label)}"  # type: ignore
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

    # TODO: see if this ugly code can be handed off to plotly, looks like not atm
    # https://github.com/janosh/pymatviz/issues/52
    # https://github.com/plotly/documentation/issues/1611
    log_cbar = dict(
        tickvals=np.arange(int(np.log10(values.max())) + 1),
        ticktext=10 ** np.arange(int(np.log10(values.max())) + 1),
    )
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
        font_colors=font_colors,
        hoverinfo="text",
        xgap=gap,
        ygap=gap,
        colorbar=log_cbar if log else None,
        zmin=cscale_range[0],
        zmax=cscale_range[1],
        # https://github.com/plotly/plotly.py/issues/193
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
