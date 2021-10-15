import sys
from typing import Any, Dict, Sequence, Union

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
            "Expected map from element symbols to heatmap values or a iterable of "
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
) -> None:
    """Plot a heatmap across the periodic table of elements.

    Inspired by from https://github.com/kaaiian/ML_figures (https://git.io/JmbaI).

    Args:
        elem_values (dict[str, int | float] | pd.Series | list[str]): Map from element
            symbols to heatmap values or iterable of composition strings/objects.
        log (bool, optional): Whether color map scale is log or linear.
        ax (Axes, optional): plt.Axes object. Defaults to None.
        cbar_title (str, optional): Title for colorbar. Defaults to "Element Count".
        cbar_max (float, optional): Maximum value of the colorbar range. Will be ignored
            if smaller than the largest plotted value. For creating multiple plots with
            identical color bars for visual comparison. Defaults to 0.
        cmap (str, optional): Matplotlib colormap name to use. Defaults to "YlGn".

    Raises:
        ValueError: provide either formulas or elem_values, not neither nor both
    """
    elem_values = count_elements(elem_values)

    color_map = get_cmap(cmap)

    n_rows = df_ptable.row.max()
    n_columns = df_ptable.column.max()

    # TODO can we pass as a kwarg and still ensure aspect ratio respected?
    fig = plt.figure(figsize=(0.75 * n_columns, 0.7 * n_rows))

    if ax is None:
        ax = plt.gca()

    rw = rh = 0.9  # rectangle width/height

    norm = LogNorm() if log else Normalize()

    # replace positive and negative infinities with NaN values, then drop all NaNs
    clean_scale = elem_values.replace([np.inf, -np.inf], np.nan).dropna()

    if cbar_max is not None:
        color_scale = [min(clean_scale.to_list()), cbar_max]
    else:
        color_scale = clean_scale.to_list()

    norm.autoscale(color_scale)

    text_style = dict(horizontalalignment="center", fontsize=16, fontweight="semibold")

    for symbol, row, column, *_ in df_ptable.values:

        row = n_rows - row  # makes periodic table right side up
        count = elem_values[symbol]

        # inf (float/0) or NaN (0/0) are expected
        # when passing in elem_values from ptable_heatmap_ratio
        if count == np.inf:
            color = "lightskyblue"  # not in formulas_b
            count_label = r"$\infty$"
        elif pd.isna(count):
            color = "white"  # not in either formulas_a nor formulas_b
            count_label = "0/0"
        else:
            color = color_map(norm(count)) if count > 0 else "#EEE"  # light gray
            # replace shortens scientific notation 1e+01 to 1e1 so it fits inside cells
            count_label = f"{count:.2g}".replace("e+0", "e")

        if row < 3:  # vertical offset for lanthanide + actinide series
            row += 0.5
        rect = Rectangle((column, row), rw, rh, edgecolor="gray", facecolor=color)

        plt.text(column + 0.5 * rw, row + 0.5 * rh, symbol, **text_style)
        plt.text(
            column + 0.5 * rw,
            row + 0.1 * rh,
            count_label,
            fontsize=12,
            horizontalalignment="center",
        )

        ax.add_patch(rect)

    # colorbar position and size: [bar_xpos, bar_ypos, bar_width, bar_height]
    # anchored at lower left corner
    cb_ax = ax.inset_axes([0.18, 0.8, 0.42, 0.05], transform=ax.transAxes)
    # format major and minor ticks
    cb_ax.tick_params(which="both", labelsize=14, width=1)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, orientation="horizontal", cax=cb_ax)
    cbar.outline.set_linewidth(1)
    cb_ax.set_title(cbar_title, pad=10, **text_style)

    plt.ylim(0.3, n_rows + 0.1)
    plt.xlim(0.9, n_columns + 1)

    plt.axis("off")


def ptable_heatmap_ratio(
    elem_values_a: ElemValues, elem_values_b: ElemValues, **kwargs: Any
) -> None:
    """Display the ratio of the normalised prevalence of each element for two sets of
    compositions.

    Args:
        formulas_a (list[str], optional): numerator compositional strings, e.g
            ["Fe2O3", "Bi2Te3"]
        formulas_b (list[str], optional): denominator compositional strings
        elem_values_a (pd.Series | dict[str, int], optional): map from element symbol
            to prevalence count for numerator
        elem_values_b (pd.Series | dict[str, int], optional): map from element symbol
            to prevalence count for denominator
        kwargs (Any, optional): kwargs passed to ptable_heatmap
    """
    elem_values_a = count_elements(elem_values_a)

    elem_values_b = count_elements(elem_values_b)

    elem_values = elem_values_a / elem_values_b

    # normalize elemental distributions, just a scaling factor but
    # makes different ratio plots comparable
    elem_values /= elem_values.sum()

    ptable_heatmap(elem_values, cbar_title="Element Ratio", **kwargs)

    # add legend for the colours
    for y_pos, label, color, txt in [
        [0.4, "white", "white", "not in either"],
        [1.1, "blue", "lightskyblue", "not in 2nd list"],
        [1.8, "gray", "silver", "not in 1st list"],
    ]:
        bbox = {"facecolor": color, "edgecolor": "gray"}
        plt.text(0.8, y_pos, f"{label}: {txt}", fontsize=12, bbox=bbox)


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
        ax (Axes): plt.Axes object. Defaults to None.
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
