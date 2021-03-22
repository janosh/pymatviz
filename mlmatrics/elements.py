from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from pymatgen import Composition

from mlmatrics.utils import ROOT, show_bar_values


def count_elements(formulas: list) -> pd.Series:
    """Count occurrences of each chemical element in a materials dataset.

    Args:
        formulas (list[str]): compositional strings, e.g. ["Fe2O3", "Bi2Te3"]

    Returns:
        pd.Series: Total number of appearances of each element in `formulas`.
    """
    formula2dict = lambda str: pd.Series(
        Composition(str).fractional_composition.as_dict()
    )

    srs = pd.Series(formulas).apply(formula2dict).sum()

    # ensure all elements are present in returned Series (with count zero if they
    # weren't in formulas)
    ptable = pd.read_csv(ROOT + "/data/periodic_table.csv")
    # fill_value=0 required as max(NaN, any int) = NaN
    srs = srs.combine(pd.Series(0, index=ptable.symbol), max, fill_value=0)
    return srs


def ptable_elemental_prevalence(
    formulas: List[str] = None,
    elem_counts: pd.Series = None,
    log: bool = False,
    cbar_title: str = None,
    cmap: str = "YlGn",
) -> None:
    """Display the prevalence of each element in a materials dataset plotted as a
    heatmap over the periodic table. `formulas` xor `elem_counts` must be passed.

    Adapted from https://github.com/kaaiian/ML_figures (https://git.io/JmbaI).

    Args:
        formulas (list[str]): compositional strings, e.g. ["Fe2O3", "Bi2Te3"]
        elem_counts (pd.Series): Map from element symbol to prevalence count
        log (bool, optional): Whether color map scale is log or linear.
        cbar_title (str, optional): Optional Title for colorbar. Defaults to None.
        cmap (str, optional): Matplotlib colormap name to use. Defaults to "YlGn".

    Raises:
        ValueError: provide either formulas or elem_counts, not neither nor both
    """
    if (formulas is None and elem_counts is None) or (
        formulas is not None and elem_counts is not None
    ):
        raise ValueError("provide either formulas or elem_counts, not neither nor both")

    if formulas is not None:
        elem_counts = count_elements(formulas)

    ptable = pd.read_csv(ROOT + "/data/periodic_table.csv")
    cmap = get_cmap(cmap)

    n_rows = ptable.row.max()
    n_columns = ptable.column.max()

    # TODO can we pass as as a kwarg and still ensure aspect ratio respected?
    plt.figure(figsize=(n_columns, n_rows))

    rw = rh = 0.9  # rectangle width/height
    min_count = elem_counts.min()
    max_count = elem_counts.replace([np.inf, -np.inf], np.nan).dropna().max()

    norm = Normalize(
        vmin=0 if log else min_count,
        vmax=np.log10(max_count) if log else max_count,
    )

    text_style = dict(
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=20,
        fontweight="semibold",
    )

    for symbol, row, column, _ in ptable.values:
        row = n_rows - row
        count = elem_counts[symbol]

        if log and count > 0:
            count = np.log10(count)

        # inf or NaN are expected when passing in elem_counts from ptable_elemental_ratio
        if count == 0:  # not in formulas_a
            color = "silver"
        elif count == np.inf:
            color = "lightskyblue"  # not in formulas_b
        elif pd.isna(count):
            color = "white"  # not in either formulas_a nor formulas_b
        else:
            color = cmap(norm(count)) if count != 0 else "silver"

        if row < 3:
            row += 0.5
        rect = Rectangle((column, row), rw, rh, edgecolor="gray", facecolor=color)

        plt.text(column + rw / 2, row + rw / 2, symbol, **text_style)

        plt.gca().add_patch(rect)

    # color bar
    granularity = 20  # number of cells in the color bar
    bar_xpos, bar_ypos = 3.5, 7.8  # bar position
    bar_width, bar_height = 9, 0.35
    cell_width = bar_width / granularity

    for idx in np.arange(granularity) + (1 if log else 0):
        value = idx * max_count / (granularity - 1)
        if log and value > 0:
            value = np.log10(value)

        color = cmap(norm(value)) if value != 0 else "silver"
        x_loc = (idx - (1 if log else 0)) / granularity * bar_width + bar_xpos
        rect = Rectangle(
            (x_loc, bar_ypos), cell_width, bar_height, edgecolor="gray", facecolor=color
        )

        if idx in np.linspace(0, granularity, granularity // 4) + (
            1 if log else 0
        ) or idx == (granularity - (0 if log else 1)):
            text = f"{value:.1f}" if log else f"{value:.0f}"
            plt.text(x_loc + cell_width / 2, bar_ypos - 0.4, text, **text_style)

        plt.gca().add_patch(rect)

    if log:
        plt.text(
            bar_xpos + cell_width / 2, bar_ypos + 0.6, int(min_count), **text_style
        )
        plt.text(x_loc + cell_width / 2, bar_ypos + 0.6, int(max_count), **text_style)

    if cbar_title is None:
        cbar_title = "log(Element Count)" if log else "Element Count"

    plt.text(bar_xpos + bar_width / 2, bar_ypos + 0.7, cbar_title, **text_style)

    plt.ylim(-0.15, n_rows + 0.1)
    plt.xlim(0.85, n_columns + 1.1)

    plt.axis("off")


def ptable_elemental_ratio(
    formulas_a: List[str] = None,
    formulas_b: List[str] = None,
    elem_counts_a: pd.Series = None,
    elem_counts_b: pd.Series = None,
    log: bool = False,
    **kwargs,
) -> None:
    """Display the ratio of the normalised prevalence of each element for two sets of
    compositions.

    Args:
        formulas_a (list[str]): numerator compositional strings, e.g. ["Fe2O3", "Bi2Te3"]
        formulas_b (list[str]): denominator compositional strings, e.g. ["Fe2O3", "Bi2Te3"]
        elem_counts_a (pd.Series): Map from element symbol to prevalence count for numerator
        elem_counts_b (pd.Series): Map from element symbol to prevalence count for denominator
        log (bool, optional): Whether color map scale is log or linear.
        kwargs (dict, optional): kwargs passed to ptable_elemental_prevalence
    """

    if (formulas_a is None and elem_counts_a is None) or (
        formulas_a is not None and elem_counts_a is not None
    ):
        raise ValueError(
            "provide either formulas_a or elem_counts_a, not neither nor both"
        )

    if (formulas_b is None and elem_counts_b is None) or (
        formulas_b is not None and elem_counts_b is not None
    ):
        raise ValueError(
            "provide either formulas_b or elem_counts_b, not neither nor both"
        )

    if formulas_a is not None:
        elem_counts_a = count_elements(formulas_a)

    if formulas_b is not None:
        elem_counts_b = count_elements(formulas_b)

    # normalize elemental distributions, just a scaling factor but
    # makes different ratio plots comparable
    elem_counts_a /= elem_counts_a.sum()
    elem_counts_b /= elem_counts_b.sum()

    elem_counts = elem_counts_a / elem_counts_b

    cbar_title = "log(Element Ratio)" if log else "Element Ratio"

    ptable_elemental_prevalence(
        elem_counts=elem_counts, log=log, cbar_title=cbar_title, **kwargs
    )

    text_style = {"fontsize": 14, "fontweight": "semibold"}

    # add key for the colours
    plt.text(
        0.8,
        2,
        "gray: not in st list",
        **text_style,
        bbox={"facecolor": "silver", "linewidth": 0},
    )
    plt.text(
        0.8,
        1.5,
        "blue: not in 2nd list",
        **text_style,
        bbox={"facecolor": "lightskyblue", "linewidth": 0},
    )
    plt.text(0.8, 1, "white: not in either", **text_style)


def hist_elemental_prevalence(
    formulas: list,
    log: bool = False,
    keep_top: int = None,
    ax: Axes = None,
    bar_values: str = "percent",
    **kwargs,
) -> None:
    """Plots a histogram of the prevalence of each element in a materials dataset.

    Adapted from https://github.com/kaaiian/ML_figures (https://git.io/JmbaI).

    Args:
        formulas (list): compositional strings, e.g. ["Fe2O3", "Bi2Te3"]
        log (bool, optional): Whether y-axis is log or linear. Defaults to False.
        keep_top (int | None): Display only the top n elements by prevalence.
        ax (Axes): plt axes. Defaults to None.
        bar_values (str): One of 'percent', 'count' or None. Annotate bars with the
            percentage each element makes up in the total element count, or use the count
            itself, or display no bar labels.
        **kwargs (int): Keyword arguments passed to show_bar_values.
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
            labels = [f"{100 * el / sum_elements:.1f}%" for el in non_zero.values]
        else:
            labels = non_zero.astype(int).to_list()
        show_bar_values(ax, labels=labels, **kwargs)
