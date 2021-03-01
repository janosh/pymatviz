from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import YlGn
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
    srs = pd.Series(formulas).apply(lambda x: pd.Series(Composition(x).as_dict())).sum()

    # ensure all elements are present in returned Series (with count zero if they
    # weren't in formulas)
    ptable = pd.read_csv(ROOT + "/data/periodic_table.csv")
    # fill_value=0 required as max(NaN, any int) = NaN
    srs = srs.combine(pd.Series(0, index=ptable.symbol), max, fill_value=0)
    return srs


def ptable_elemental_prevalence(
    formulas: List[str] = None, elem_counts: pd.Series = None, log_scale: bool = False
) -> None:
    """Display the prevalence of each element in a materials dataset plotted as a
    heatmap over the periodic table. `formulas` xor `elem_counts` must be passed.

    Adapted from https://github.com/kaaiian/ML_figures.

    Args:
        formulas (list[str]): compositional strings, e.g. ["Fe2O3", "Bi2Te3"]
        elem_counts (pd.Series): Map from element symbol to prevalence count
        log_scale (bool, optional): Whether color map scale is log or linear.
    """
    if (formulas is None and elem_counts is None) or (
        formulas is not None and elem_counts is not None
    ):
        raise ValueError("provide either formulas or elem_counts, not neither nor both")

    if formulas is not None:
        elem_counts = count_elements(formulas)

    ptable = pd.read_csv(ROOT + "/data/periodic_table.csv")

    n_row = ptable.row.max()
    n_column = ptable.column.max()

    plt.figure(figsize=(n_column, n_row))

    rw = rh = 0.9  # rectangle width/height
    count_min = elem_counts.min()
    count_max = elem_counts.max()

    norm = Normalize(
        vmin=0 if log_scale else count_min,
        vmax=np.log(count_max) if log_scale else count_max,
    )

    text_style = dict(
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=20,
        fontweight="semibold",
        color="black",
    )

    for symbol, row, column, _ in ptable.values:
        row = n_row - row
        count = elem_counts[symbol]
        if log_scale and count != 0:
            count = np.log(count)
        color = YlGn(norm(count)) if count != 0 else "silver"

        if row < 3:
            row += 0.5
        rect = Rectangle((column, row), rw, rh, edgecolor="gray", facecolor=color)

        plt.text(column + rw / 2, row + rw / 2, symbol, **text_style)

        plt.gca().add_patch(rect)

    granularity = 20
    x_offset = 3.5
    y_offset = 7.8
    length = 9
    for i in range(granularity):
        value = int(round((i) * count_max / (granularity - 1)))
        if log_scale and value != 0:
            value = np.log(value)
        color = YlGn(norm(value)) if value != 0 else "silver"
        x_loc = i / (granularity) * length + x_offset
        width = length / granularity
        height = 0.35
        rect = Rectangle(
            (x_loc, y_offset), width, height, edgecolor="gray", facecolor=color
        )

        if i in [0, 4, 9, 14, 19]:
            text = f"{value:g}"
            if log_scale:
                text = f"{np.exp(value):g}".replace("e+0", "e")
            plt.text(x_loc + width / 2, y_offset - 0.4, text, **text_style)

        plt.gca().add_patch(rect)

    plt.text(
        x_offset + length / 2,
        y_offset + 0.7,
        "log(Element Count)" if log_scale else "Element Count",
        horizontalalignment="center",
        verticalalignment="center",
        fontweight="semibold",
        fontsize=20,
        color="k",
    )

    plt.ylim(-0.15, n_row + 0.1)
    plt.xlim(0.85, n_column + 1.1)

    plt.axis("off")


def hist_elemental_prevalence(
    formulas: list,
    log_scale: bool = False,
    keep_top: int = None,
    ax: Axes = None,
    bar_values: str = "percent",
    **kwargs,
) -> None:
    """Plots a histogram of the prevalence of each element in a materials dataset.
    Adapted from https://github.com/kaaiian/ML_figures.

    Args:
        formulas (list): compositional strings, e.g. ["Fe2O3", "Bi2Te3"]
        log_scale (bool, optional): Whether y-axis is log or linear. Defaults to False.
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

    plt.ylabel("log(Element Count)" if log_scale else "Element Count")

    if log_scale:
        plt.yscale("log")

    if bar_values is not None:
        if bar_values == "percent":
            sum_elements = non_zero.sum()
            labels = [f"{100 * el / sum_elements:.1f}%" for el in non_zero.values]
        else:
            labels = non_zero.astype(int).to_list()
        show_bar_values(ax, labels=labels, **kwargs)
