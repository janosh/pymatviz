import sys
from typing import Any, Dict, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pymatgen.core import Composition

from ml_matrics.utils import ROOT, annotate_bar_heights


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


df_ptable = pd.read_csv(f"{ROOT}/ml_matrics/elements.csv")

ElemValues = Union[Dict[str, Union[int, float]], pd.Series, Sequence[str]]


def count_elements(elem_values: ElemValues) -> pd.Series:
    """Processes elemental heatmap data. If passed a list of strings, assume they are
    compositions and count the occurrences of each chemical element. Else ensure the
    data is a pd.Series filled with zero values for missing element symbols.

    Args:
        elem_values (dict[str, int | float] | pd.Series | list[str]): Map from element
            symbols to heatmap values or iterable of composition strings/objects.

    Returns:
        pd.Series: Map element symbols to heatmap values.
    """
    # ensure elem_values is Series if we got dict/list/tuple
    srs = pd.Series(elem_values)

    if is_numeric_dtype(srs):
        pass
    elif is_string_dtype(srs):
        # assume all items in elem_values are composition strings
        formula2dict = lambda str: pd.Series(
            Composition(str).fractional_composition.as_dict()
        )
        # sum up element occurrences
        srs = pd.Series(elem_values).apply(formula2dict).sum()
    else:
        raise ValueError(
            "Expected map from element symbols to heatmap values or a list of "
            f"compositions (strings or Pymatgen objects), got {elem_values=}"
        )

    # ensure all elements are present in returned Series (with value zero if they
    # weren't in formulas)
    zeros = pd.Series(0, index=df_ptable.symbol)

    # fill_value=0 required as max(NaN, any int) = NaN
    srs = srs.combine(zeros, max, fill_value=0)
    return srs


def ptable_heatmap(
    elem_values: ElemValues,
    log: bool = False,
    ax: Axes = None,
    cbar_title: str = "Element Count",
    cbar_max: Union[float, int, None] = None,
    cmap: str = "summer_r",
    zero_color: str = "#DDD",  # light gray
    infty_color: str = "lightskyblue",
    na_color: str = "white",
    heat_labels: Literal["value", "fraction", "percent", None] = "value",
    precision: str = None,
) -> Axes:
    """Plot a heatmap across the periodic table of elements.

    Args:
        elem_values (dict[str, int | float] | pd.Series | list[str]): Map from element
            symbols to heatmap values or iterable of composition strings/objects.
        log (bool, optional): Whether color map scale is log or linear.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
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
        heat_labels ("value" | "fraction" | "percent" | None): Whether to display heat
            values as is (value), normalized as a fraction of the total, as percentages
            or not at all (None). Defaults to "value".
            "fraction" and "percent" can be used to make the colors in different heatmap
            (and ratio) plots comparable.
        precision (str): f-string format option for heat labels. Defaults to None in
            which case we fall back on ".1%" (1 decimal place) if heat_labels="percent"
            else ".3g".

    Returns:
        ax: matplotlib Axes with the heatmap.
    """
    if log and heat_labels in ("fraction", "percent"):
        raise ValueError(
            "Combining log color scale and heat_labels='fraction'/'percent' unsupported"
        )

    elem_values = count_elements(elem_values)

    # replace positive and negative infinities with NaN values, then drop all NaNs
    clean_vals = elem_values.replace([np.inf, -np.inf], np.nan).dropna()

    if heat_labels in ("fraction", "percent"):
        # ignore inf values in sum() else all would be set to 0 by normalizing
        elem_values /= clean_vals.sum()
        clean_vals /= clean_vals.sum()  # normalize as well for norm.autoscale() below

    color_map = get_cmap(cmap)

    n_rows = df_ptable.row.max()
    n_columns = df_ptable.column.max()

    # TODO can we pass as a kwarg and still ensure aspect ratio respected?
    fig = plt.figure(figsize=(0.75 * n_columns, 0.7 * n_rows))

    if ax is None:
        ax = plt.gca()

    rw = rh = 0.9  # rectangle width/height

    norm = LogNorm() if log else Normalize()

    norm.autoscale(clean_vals.to_numpy())
    if cbar_max is not None:
        norm.vmax = cbar_max

    text_style = dict(horizontalalignment="center", fontsize=16, fontweight="semibold")

    for symbol, row, column, *_ in df_ptable.values:

        row = n_rows - row  # makes periodic table right side up
        heat_val = elem_values[symbol]

        # inf (float/0) or NaN (0/0) are expected when passing in elem_values from
        # ptable_heatmap_ratio
        if heat_val == np.inf:
            color = infty_color  # not in denominator
            label = r"$\infty$"
        elif pd.isna(heat_val):
            color = na_color  # neither numerator nor denominator
            label = r"$0\,/\,0$"
        else:
            color = color_map(norm(heat_val)) if heat_val > 0 else zero_color

            if heat_labels == "percent":
                label = f"{heat_val:{precision or '.1%'}}"
            else:
                label = f"{heat_val:{precision or '.3g'}}"
            # replace shortens scientific notation 1e+01 to 1e1 so it fits inside cells
            label = label.replace("e+0", "e")
        if row < 3:  # vertical offset for lanthanide + actinide series
            row += 0.5
        rect = Rectangle((column, row), rw, rh, edgecolor="gray", facecolor=color)

        if heat_labels is None:
            # no value to display below in colored rectangle so center element symbol
            text_style["verticalalignment"] = "center"

        plt.text(column + 0.5 * rw, row + 0.5 * rh, symbol, **text_style)

        if heat_labels is not None:
            textcolor = "white" if norm(heat_val) > 0.8 else "black"

            plt.text(
                column + 0.5 * rw,
                row + 0.1 * rh,
                label,
                fontsize=12,
                horizontalalignment="center",
                color=textcolor,
            )

        ax.add_patch(rect)

    if heat_labels is not None:

        # colorbar position and size: [bar_xpos, bar_ypos, bar_width, bar_height]
        # anchored at lower left corner
        cb_ax = ax.inset_axes([0.18, 0.8, 0.42, 0.05], transform=ax.transAxes)
        # format major and minor ticks
        cb_ax.tick_params(which="both", labelsize=14, width=1)

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        def tick_fmt(val: float, pos: int) -> str:
            # val: value at color axis tick (e.g. 10.0, 20.0, ...)
            # pos: zero-based tick counter (e.g. 0, 1, 2, ...)
            if heat_labels == "percent":
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
    cbar_title: str = "Element Ratio",
    not_in_numerator: Tuple[str, str] = ("#DDD", "gray: not in 1st list"),
    not_in_denominator: Tuple[str, str] = ("lightskyblue", "blue: not in 2nd list"),
    not_in_either: Tuple[str, str] = ("white", "white: not in either"),
    **kwargs: Any,
) -> Axes:
    """Display the ratio of two maps from element symbols to heat values or of two sets
    of compositions.

    Args:
        elem_values_num (dict[str, int | float] | pd.Series | list[str]): Map from
            element symbols to heatmap values or iterable of composition strings/objects
            in the numerator.
        elem_values_denom (dict[str, int | float] | pd.Series | list[str]): Map from
            element symbols to heatmap values or iterable of composition strings/objects
            in the denominator.
        cbar_title (str): Title for the color bar. Defaults to "Element Ratio".
        not_in_numerator (tuple[str, str]): Color and legend description used for
            elements missing from numerator.
        not_in_denominator (tuple[str, str]): See not_in_numerator.
        not_in_either (tuple[str, str]): See not_in_numerator.
        kwargs (Any, optional): kwargs passed to ptable_heatmap.

    Returns:
        ax: The plot's matplotlib Axes.
    """
    elem_values_num = count_elements(elem_values_num)

    elem_values_denom = count_elements(elem_values_denom)

    elem_values = elem_values_num / elem_values_denom

    kwargs["zero_color"] = not_in_numerator[0]
    kwargs["infty_color"] = not_in_denominator[0]
    kwargs["na_color"] = not_in_either[0]

    ax = ptable_heatmap(elem_values, cbar_title=cbar_title, precision=".1f", **kwargs)

    # add legend handles
    for y_pos, color, txt in (
        (1.8, *not_in_numerator),
        (1.1, *not_in_denominator),
        (0.4, *not_in_either),
    ):
        bbox = dict(facecolor=color, edgecolor="gray")
        plt.text(0.8, y_pos, txt, fontsize=12, bbox=bbox)

    return ax


def hist_elemental_prevalence(
    formulas: Sequence[str],
    log: bool = False,
    keep_top: int = None,
    ax: Axes = None,
    bar_values: Literal["percent", "count", None] = "percent",
    **kwargs: Any,
) -> None:
    """Plots a histogram of the prevalence of each element in a materials dataset.

    Adapted from https://github.com/kaaiian/ML_figures (https://git.io/JmbaI).

    Args:
        formulas (list[str]): compositional strings, e.g. ["Fe2O3", "Bi2Te3"].
        log (bool, optional): Whether y-axis is log or linear. Defaults to False.
        keep_top (int | None): Display only the top n elements by prevalence.
        ax (Axes): matplotlib Axes on which to plot. Defaults to None.
        bar_values ('percent'|'count'|None): 'percent' annotates bars with the
            percentage each element makes up in the total element count. 'count'
            displays count itself. None removes bar labels.
        **kwargs (int): Keyword arguments passed to annotate_bar_heights.
    """
    if ax is None:
        ax = plt.gca()

    elem_counts = count_elements(formulas)
    non_zero = elem_counts[elem_counts > 0].sort_values(ascending=False)
    if keep_top is not None:
        non_zero = non_zero.head(keep_top)
        plt.title(f"Top {keep_top} Elements")

    non_zero.plot.bar(width=0.7, edgecolor="black")

    plt.ylabel("log(Element Count)" if log else "Element Count")

    if log:
        plt.yscale("log")

    if bar_values is not None:
        if bar_values == "percent":
            sum_elements = non_zero.sum()
            labels = [f"{el / sum_elements:.1%}" for el in non_zero.values]
        else:
            labels = non_zero.astype(int).to_list()
        annotate_bar_heights(ax, labels=labels, **kwargs)

    return ax
