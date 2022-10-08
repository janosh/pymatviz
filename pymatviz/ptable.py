from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pymatgen.core import Composition

from pymatviz.utils import df_ptable


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


if TYPE_CHECKING:
    from typing import TypeAlias

    ElemValues: TypeAlias = dict[str | int, int | float] | pd.Series | Sequence[str]

    CountMode = Literal[
        "element_composition", "fractional_composition", "reduced_composition"
    ]


def count_elements(
    elem_values: ElemValues,
    count_mode: CountMode = "element_composition",
    exclude_elements: Sequence[str] = (),
) -> pd.Series:
    """Processes elemental heatmap data. If passed a list of strings, assume they are
    compositions and count the occurrences of each chemical element. Else ensure the
    data is a pd.Series filled with zero values for missing element symbols.

    Args:
        elem_values (dict[str, int | float] | pd.Series | list[str]): Iterable of
            composition strings/objects or map from element symbols to heatmap values.
        count_mode ('{element|fractional|reduced}_composition'):
            Only used when elem_values is a list of composition strings/objects.
            - element_composition (default): Count elements in each composition as is,
                i.e. without reduction or normalization.
            - fractional_composition: Convert to normalized compositions in which the
                amounts of each species sum to before counting.
                Example: Fe2 O3 -> Fe0.4 O0.6
            - reduced_composition: Convert to reduced compositions (i.e. amounts
                normalized by greatest common denominator) before counting.
                Example: Fe4 P4 O16 -> Fe P O4.
        exclude_elements (Sequence[str]): Elements to exclude from the count. Defaults
            to ().

    Returns:
        pd.Series: Map element symbols to heatmap values.
    """
    # ensure elem_values is Series if we got dict/list/tuple
    srs = pd.Series(elem_values)

    if is_numeric_dtype(srs):
        pass
    elif is_string_dtype(srs):
        # assume all items in elem_values are composition strings
        srs = pd.DataFrame(
            getattr(Composition(comp_str), count_mode).as_dict() for comp_str in srs
        ).sum()  # sum up element occurrences
    else:
        raise ValueError(
            "Expected elem_values to be map from element symbols to heatmap values or "
            f"list of compositions (strings or Pymatgen objects), got {elem_values}"
        )

    try:
        # if index consists entirely of strings representing integers, convert to ints
        srs.index = srs.index.astype(int)
    except (ValueError, TypeError):
        pass

    if pd.api.types.is_integer_dtype(srs.index):
        # if index is all integers, assume they represent atomic
        # numbers and map them to element symbols (H: 1, He: 2, ...)
        if srs.index.max() > 118 or srs.index.min() < 1:
            raise ValueError(
                "element value keys were found to be integers and assumed to represent "
                "atomic numbers, but values are outside expected range [1, 118]."
            )
        map_atomic_num_to_elem_symbol = (
            df_ptable.reset_index().set_index("atomic_number").symbol
        )
        srs.index = srs.index.map(map_atomic_num_to_elem_symbol)

    # ensure all elements are present in returned Series (with value zero if they
    # weren't in elem_values before)
    srs = srs.reindex(df_ptable.index, fill_value=0).rename("count")

    if len(exclude_elements) > 0:
        try:
            srs = srs.drop(exclude_elements)
        except KeyError as exc:
            bad_symbols = ", ".join(x for x in exclude_elements if x not in srs)
            raise ValueError(
                f"Unexpected symbol(s) {bad_symbols} in "
                f"exclude_elements={exclude_elements}"
            ) from exc

    return srs


def ptable_heatmap(
    elem_values: ElemValues,
    log: bool = False,
    ax: plt.Axes = None,
    count_mode: CountMode = "element_composition",
    cbar_title: str = "Element Count",
    cbar_max: float | int | None = None,
    cmap: str = "summer_r",
    zero_color: str = "#DDD",  # light gray
    infty_color: str = "lightskyblue",
    na_color: str = "white",
    heat_mode: Literal["value", "fraction", "percent", None] = "value",
    precision: str = None,
    text_color: str | tuple[str, str] = "auto",
    exclude_elements: Sequence[str] = (),
    zero_symbol: str | float = "-",
) -> plt.Axes:
    """Plot a heatmap across the periodic table of elements.

    Args:
        elem_values (dict[str, int | float] | pd.Series | list[str]): Map from element
            symbols to heatmap values or iterable of composition strings/objects.
        log (bool, optional): Whether color map scale is log or linear.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        count_mode ('composition' | 'fractional_composition' | 'reduced_composition'):
            Reduce or normalize compositions before counting. See count_elements() for
            details. Only used when elem_values is list of composition strings/objects.
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
        precision (str): f-string format option for heat labels. Defaults to None in
            which case we fall back on ".1%" (1 decimal place) if heat_mode="percent"
            else ".3g".
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

    elem_values = count_elements(elem_values, count_mode, exclude_elements)

    # replace positive and negative infinities with NaN values, then drop all NaNs
    clean_vals = elem_values.replace([np.inf, -np.inf], np.nan).dropna()

    if heat_mode in ("fraction", "percent"):
        # ignore inf values in sum() else all would be set to 0 by normalizing
        elem_values /= clean_vals.sum()
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

        row = n_rows - row  # makes periodic table right side up
        heat_val = elem_values.get(symbol)

        # inf (float/0) or NaN (0/0) are expected when passing in elem_values from
        # ptable_heatmap_ratio
        if symbol in exclude_elements:
            color = "white"
            label = "excl."
        elif heat_val == np.inf:
            color = infty_color  # not in denominator
            label = r"$\infty$"
        elif pd.isna(heat_val):
            color = na_color  # neither numerator nor denominator
            label = r"$0\,/\,0$"
        elif heat_val == 0:
            color = zero_color
            label = str(zero_symbol)
        else:
            color = color_map(norm(heat_val))

            if heat_mode == "percent":
                label = f"{heat_val:{precision or '.1%'}}"
            else:
                prec = precision or (".0f" if heat_val > 100 else ".1f")
                label = f"{heat_val:{prec}}"
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
            text_clr = "white" if norm(heat_val) > 0.5 else "black"
        elif isinstance(text_color, (tuple, list)):
            text_clr = text_color[0] if norm(heat_val) > 0.5 else text_color[1]
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
            if heat_mode == "percent":
                # display color bar values as percentages
                return f"{val:.0%}"
            if val < 1e4:
                return f"{val:.0f}"
            return f"{val:.2g}"

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
    elem_values_num: ElemValues,
    elem_values_denom: ElemValues,
    count_mode: CountMode = "element_composition",
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
        elem_values_num (dict[str, int | float] | pd.Series | list[str]): Map from
            element symbols to heatmap values or iterable of composition strings/objects
            in the numerator.
        elem_values_denom (dict[str, int | float] | pd.Series | list[str]): Map from
            element symbols to heatmap values or iterable of composition strings/objects
            in the denominator.
        normalize (bool): Whether to normalize heatmap values so they sum to 1. Makes
            different ptable_heatmap_ratio plots comparable. Defaults to False.
        count_mode ('composition' | 'fractional_composition' | 'reduced_composition'):
            Reduce or normalize compositions before counting. See count_elements() for
            details. Only used when elem_values is list of composition strings/objects.
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
    elem_values_num = count_elements(elem_values_num, count_mode)

    elem_values_denom = count_elements(elem_values_denom, count_mode)

    elem_values = elem_values_num / elem_values_denom

    if normalize:
        elem_values /= elem_values.sum()

    kwargs["zero_color"] = not_in_numerator[0]
    kwargs["infty_color"] = not_in_denominator[0]
    kwargs["na_color"] = not_in_either[0]

    ax = ptable_heatmap(elem_values, cbar_title=cbar_title, **kwargs)

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
    elem_values: ElemValues,
    count_mode: CountMode = "element_composition",
    colorscale: str | Sequence[str] | Sequence[tuple[float, str]] | None = None,
    showscale: bool = True,
    heat_mode: Literal["value", "fraction", "percent", None] = "value",
    precision: str = None,
    hover_props: Sequence[str] | dict[str, str] | None = None,
    hover_data: dict[str, str | int | float] | pd.Series | None = None,
    font_colors: Sequence[str] = ("black",),
    gap: float = 5,
    font_size: int = None,
    bg_color: str = None,
    color_bar: dict[str, Any] = None,
    exclude_elements: Sequence[str] = (),
    log: bool = False,
) -> go.Figure:
    """Creates a Plotly figure with an interactive heatmap of the periodic table.
    Supports hover tooltips with custom data or atomic reference data like
    electronegativity, atomic_radius, etc. See kwargs hover_data and hover_props, resp.

    Args:
        elem_values (dict[str, int | float] | pd.Series | list[str]): Map from element
            symbols to heatmap values e.g. {"Fe": 2, "O": 3} or iterable of composition
            strings or Pymatgen composition objects.
        count_mode ("composition" | "fractional_composition" | "reduced_composition"):
            Reduce or normalize compositions before counting. See count_elements() for
            details. Only used when elem_values is list of composition strings/objects.
        colorscale (str | list[str] | list[tuple[float, str]]): Color scale for heatmap.
            Defaults to plotly.express.colors.sequential.Pinkyl. See
            https://plotly.com/python/builtin-colorscales for names of other builtin
            color scales. Note e.g. colorscale="YlGn" and px.colors.sequential.YlGn are
            equivalent. Custom scales are specified as ["blue", "red"] or
            [[0, "rgb(0,0,255)"], [0.5, "rgb(0,255,0)"], [1, "rgb(255,0,0)"]].
        showscale (bool): Whether to show a bar for the color scale. Defaults to True.
        heat_mode ("value" | "fraction" | "percent" | None): Whether to display heat
            values as is (value), normalized as a fraction of the total, as percentages
            or not at all (None). Defaults to "value".
            "fraction" and "percent" can be used to make the colors in different
            periodic table heatmap plots comparable.
        precision (str): f-string format option for heat labels. Defaults to None in
            which case we fall back on ".1%" (1 decimal place) if heat_mode="percent"
            else ".3g".
        hover_props (list[str] | dict[str, str]): Elemental properties to display in the
            hover tooltip. Can be a list of property names to display only the values
            themselves or a dict mapping names to what they should display as. E.g.
            {"atomic_mass": "atomic weight"} will display as "atomic weight = {x}".
            Defaults to None. Available properties are: symbol, row, column, name,
            atomic_number, atomic_mass, n_neutrons, n_protons, n_electrons, period,
            group, phase, radioactive, natural, metal, nonmetal, metalloid, type,
            atomic_radius, electronegativity, first_ionization, density, melting_point,
            boiling_point, number_of_isotopes, discoverer, year, specific_heat,
            n_shells, n_valence.
        hover_data (dict[str, str | int | float] | pd.Series): Map from element symbols
            to additional data to display in the hover tooltip. {"Fe": "this shows up in
            the hover tooltip on a new line below the element name"}. Defaults to None.
        font_colors (list[str]): One or two color strings [min_color, max_color].
            min_color is applied to annotations for heatmap values
            < (max_val - min_val) / 2. Defaults to ["black"].
        gap (float): Gap in pixels between tiles of the periodic table. Defaults to 5.
        font_size (int): Element symbol and heat label text size. Defaults to None,
            meaning automatic font size based on plot size.
        bg_color (str): Plot background color. Defaults to "rgba(0, 0, 0, 0)".
        color_bar (dict[str, Any]): Plotly color bar properties documented at
            https://plotly.com/python/reference#heatmap-colorbar. Defaults to None.
        exclude_elements (list[str]): Elements to exclude from the heatmap. E.g. if
            oxygen overpowers everything, you can do exclude_elements=['O'].
            Defaults to ().
        log (bool): Whether to use a logarithmic color scale. Defaults to False.
            Piece of advice: colorscale='viridis' and log=True go well together.

    Returns:
        Figure: Plotly Figure object.
    """
    if log and heat_mode in ("fraction", "percent"):
        raise ValueError(
            "Combining log color scale and heat_mode='fraction'/'percent' unsupported"
        )

    elem_values = count_elements(elem_values, count_mode, exclude_elements)

    if log and elem_values[elem_values != 0].min() <= 1:
        raise ValueError(
            "Log color scale requires all heat map values to be > 1 since values <= 1 "
            "map to negative log values which throws off the color scale."
        )

    if heat_mode in ("fraction", "percent"):
        # normalize heat values
        clean_vals = elem_values.replace([np.inf, -np.inf], np.nan).dropna()
        # ignore inf values in sum() else all would be set to 0 by normalizing
        heat_value_element_map = elem_values / clean_vals.sum()
    else:
        heat_value_element_map = elem_values

    n_rows, n_columns = 10, 18
    # initialize tile text and hover tooltips to empty strings
    tile_texts, hover_texts = np.full([2, n_rows, n_columns], "", dtype=object)
    heatmap_values = np.zeros([n_rows, n_columns])

    for symbol, period, group, name, *_ in df_ptable.itertuples():
        # build table from bottom up so that period 1 becomes top row
        row = n_rows - period
        col = group - 1

        label = None  # label (if not None) is placed below the element symbol
        if symbol in exclude_elements:
            label = "excl."
        elif symbol in heat_value_element_map:
            heat_value = heat_value_element_map[symbol]
            if heat_mode == "percent":
                label = f"{heat_value:{precision or '.1%'}}"
            else:
                if precision is None:
                    prec = ".1f" if heat_value < 100 else ".0f"
                    if heat_value > 1e5:
                        prec = ".2g"
                label = f"{heat_value:{precision or prec}}".replace("e+0", "e")

        style = f"font-weight: bold; font-size: {1.5 * (font_size or 12)};"
        tile_text = f"<span style={style}>{symbol}</span>"
        if label is not None:
            tile_text += f"<br>{label}"

        tile_texts[row][col] = tile_text

        hover_text = name

        if hover_data is not None and symbol in hover_data:
            hover_text += f"<br>{hover_data[symbol]}"

        if hover_props is not None:
            unsupported_keys = set(hover_props) - set(df_ptable)
            if unsupported_keys:
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
        heatmap_values[row][col] = color_val + 1e-6

    rgba0 = "rgba(0, 0, 0, 0)"
    if colorscale is None:
        colorscale = [rgba0] + px.colors.sequential.Pinkyl
    elif isinstance(colorscale, str):
        colorscale = [(0, rgba0)] + px.colors.get_colorscale(colorscale)
        colorscale[1][0] = 1e-6  # type: ignore
    elif isinstance(colorscale, Sequence) and isinstance(colorscale[0], str):
        colorscale = [rgba0] + list(colorscale)  # type: ignore
    elif isinstance(colorscale, Sequence) and isinstance(colorscale[0], (list, tuple)):
        # list of tuples(float in [0, 1], color)
        # make sure we're dealing with mutable lists
        colorscale = [(0, rgba0)] + list(map(list, colorscale))  # type: ignore
        colorscale[1][0] = 1e-6  # type: ignore
    else:
        raise ValueError(
            f"colorscale={colorscale} should be string, list of strings or list of "
            "tuples(float, str)"
        )

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
        # TODO: see if this ugly code can be handed off to plotly, looks like not atm
        # https://github.com/janosh/pymatviz/issues/52
        # https://github.com/plotly/documentation/issues/1611
        colorbar=dict(
            tickvals=np.arange(int(np.log10(elem_values.max())) + 1),
            ticktext=10 ** np.arange(int(np.log10(elem_values.max())) + 1),
        )
        if log
        else None,
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
    fig.update_traces(
        colorbar=dict(lenmode="fraction", len=0.87, thickness=15, **(color_bar or {}))
    )
    return fig
