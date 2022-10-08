from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import transforms
from matplotlib.ticker import FixedLocator
from pymatgen.core import Structure
from pymatgen.symmetry.groups import SpaceGroup

from pymatviz.ptable import count_elements
from pymatviz.utils import Array, annotate_bars, get_crystal_sys


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from pymatviz.ptable import CountMode, ElemValues


def residual_hist(
    y_true: Array,
    y_pred: Array,
    ax: plt.Axes = None,
    xlabel: str | None = r"Residual ($y_\mathrm{test} - y_\mathrm{pred}$)",
    **kwargs: Any,
) -> plt.Axes:
    r"""Plot the residual distribution overlaid with a Gaussian kernel
    density estimate.

    Adapted from https://github.com/kaaiian/ML_figures (https://git.io/Jmb2O).

    Args:
        y_true (array): ground truth targets
        y_pred (array): model predictions
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        xlabel (str, optional): x-axis label. Defaults to
            'Residual ($y_\mathrm{test} - y_\mathrm{pred}$)
        **kwargs: Additional keyword arguments to pass to matplotlib.Axes.

    Returns:
        ax: The plot's matplotlib Axes.
    """
    ax = ax or plt.gca()

    y_res = y_pred - y_true

    ax.hist(
        y_res, bins=kwargs.pop("bins", 50), density=True, edgecolor="black", **kwargs
    )

    # Gaussian kernel density estimation: evaluates the Gaussian
    # probability density estimated based on the points in y_res
    kde = scipy.stats.gaussian_kde(y_res)
    x_range = np.linspace(min(y_res), max(y_res), 100)

    label = "Gaussian kernel density estimate"
    ax.plot(x_range, kde(x_range), linewidth=3, color="red", label=label)

    ax.set(xlabel=xlabel)
    ax.legend(loc="upper left", framealpha=0.5, handlelength=1)

    return ax


def true_pred_hist(
    y_true: Array,
    y_pred: Array,
    y_std: Array,
    ax: plt.Axes = None,
    cmap: str = "hot",
    truth_color: str = "blue",
    **kwargs: Any,
) -> plt.Axes:
    """Plot a histogram of model predictions with bars colored by the mean uncertainty
    of predictions in that bin. Overlaid by a more transparent histogram of ground truth
    values.

    Args:
        y_true (array): ground truth targets
        y_pred (array): model predictions
        y_std (array): model uncertainty
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        cmap (str, optional): string identifier of a plt colormap. Defaults to 'hot'.
        truth_color (str, optional): Face color to use for y_true bars.
            Defaults to 'blue'.
        **kwargs: Additional keyword arguments to pass to ax.hist().

    Returns:
        ax: The plot's matplotlib Axes.
    """
    ax = ax or plt.gca()

    color_map = getattr(plt.cm, cmap)
    y_true, y_pred, y_std = np.array([y_true, y_pred, y_std])

    _, bin_edges, bars = ax.hist(
        y_pred, alpha=0.8, label=r"$y_\mathrm{pred}$", **kwargs
    )
    kwargs.pop("bins", None)
    ax.hist(
        y_true,
        bins=bin_edges,
        alpha=0.2,
        color=truth_color,
        label=r"$y_\mathrm{true}$",
        **kwargs,
    )

    for xmin, xmax, rect in zip(bin_edges, bin_edges[1:], bars.patches):

        y_preds_in_rect = np.logical_and(y_pred > xmin, y_pred < xmax).nonzero()

        color_value = y_std[y_preds_in_rect].mean()

        rect.set_color(color_map(color_value))

    ax.legend(frameon=False)

    norm = plt.cm.colors.Normalize(vmax=y_std.max(), vmin=y_std.min())
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=color_map), pad=0.075, ax=ax
    )
    cbar.outline.set_linewidth(1)
    cbar.set_label(r"mean $y_\mathrm{std}$ of prediction in bin")
    cbar.ax.yaxis.set_ticks_position("left")

    ax.figure.set_size_inches(12, 7)

    return ax


def spacegroup_hist(
    data: Sequence[int | str] | pd.Series,
    show_counts: bool = True,
    xticks: Literal["all", "crys_sys_edges"] | int = 20,
    include_missing: bool = False,
    ax: plt.Axes = None,
    **kwargs: Any,
) -> plt.Axes:
    """Plot a histogram of spacegroups shaded by crystal system.

    Args:
        data (list[int | str] | pd.Series): A sequence (list, tuple, pd.Series) of
            space group strings or numbers (from 1 - 230) or pymatgen structures.
        show_counts (bool, optional): Whether to count the number of items
            in each crystal system. Defaults to True.
        xticks ('all' | 'crys_sys_edges' | int, optional): Where to add x-ticks. An
            integer will add ticks below that number of tallest bars. Defaults to 20.
            'all' will show below all bars, 'crys_sys_edges' only at the edge from one
            crystal system to another.
        include_missing (bool, optional): Whether to include a 0-height bar for missing
            space groups missing from the data. Currently only implemented for numbers,
            not symbols. Defaults to False.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        kwargs: Keywords passed to pandas.Series.plot.bar().

    Returns:
        ax: The plot's matplotlib Axes.
    """
    ax = ax or plt.gca()

    if isinstance(next(iter(data)), Structure):
        # if 1st sequence item is structure, assume all are
        series = pd.Series(
            struct.get_space_group_info()[1] for struct in data  # type: ignore
        )
    else:
        series = pd.Series(data)

    df = pd.DataFrame(series.value_counts(sort=False))
    df.columns = ["counts"]

    crys_colors = {
        "triclinic": "red",
        "monoclinic": "teal",
        "orthorhombic": "blue",
        "tetragonal": "green",
        "trigonal": "orange",
        "hexagonal": "purple",
        "cubic": "yellow",
    }

    if df.index.is_numeric():  # assume index is space group numbers
        if include_missing:
            df = df.reindex(range(1, 231), fill_value=0)
        else:
            df = df.sort_index()
        df["crystal_sys"] = [get_crystal_sys(x) for x in df.index]
        ax.set(xlim=(0, 230))
        xlabel = "International Spacegroup Number"

    else:  # assume index is space group symbols
        # TODO: figure how to implement include_missing for space group symbols
        # if include_missing:
        #     idx = [SpaceGroup.from_int_number(x).symbol for x in range(1, 231)]
        #     df = df.reindex(idx, fill_value=0)
        df["crystal_sys"] = [SpaceGroup(x).crystal_system for x in df.index]

        # sort df by crystal system going from smallest to largest spacegroup numbers
        # e.g. triclinic (1-2) comes first, cubic (195-230) last
        sys_order = dict(zip(crys_colors, range(len(crys_colors))))
        df = df.loc[df.crystal_sys.map(sys_order).sort_values().index]

        xlabel = "International Spacegroup Symbol"

    ax.set(xlabel=xlabel, ylabel="Count")

    kwargs["width"] = kwargs.get("width", 0.9)  # set default bar width
    # make plot
    df.counts.plot.bar(figsize=[16, 4], ax=ax, **kwargs)

    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/fill_between_demo
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    # count rows per crystal system
    crys_sys_counts = df.groupby("crystal_sys").sum("counts")

    # sort by key order in dict crys_colors
    crys_sys_counts = crys_sys_counts.loc[
        [x for x in crys_colors if x in crys_sys_counts.index]
    ]

    crys_sys_counts["width"] = df.value_counts("crystal_sys")
    ax.set_title("Totals per crystal system", fontdict={"fontsize": 18}, pad=30)
    crys_sys_counts["color"] = pd.Series(crys_colors)

    x0 = 0
    for cryst_sys, count, width, color in crys_sys_counts.itertuples():
        x1 = x0 + width

        for patch in ax.patches[0 if x0 == 1 else x0 : x1 + 1]:
            patch.set_facecolor(color)

        text_kwds = dict(transform=trans, horizontalalignment="center")
        ax.text(
            *[(x0 + x1) / 2, 0.95],
            cryst_sys,
            rotation=90,
            verticalalignment="top",
            fontdict={"fontsize": 14},
            **text_kwds,
        )
        if show_counts:
            ax.text(
                *[(x0 + x1) / 2, 1.02],
                f"{count:,} ({count/len(data):.0%})",
                fontdict={"fontsize": 12},
                **text_kwds,
            )

        ax.fill_between(
            [x0 - 0.5, x1 - 0.5],
            *[0, 1],
            facecolor=color,
            alpha=0.1,
            transform=trans,
            edgecolor="black",
        )
        x0 += width

    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    if xticks == "crys_sys_edges" or isinstance(xticks, int):

        if isinstance(xticks, int):
            # get x_locs of n=xticks tallest bars
            x_indices = df.reset_index().sort_values("counts").tail(xticks).index
        else:
            # add x_locs of n=xticks tallest bars
            x_indices = crys_sys_counts.width.cumsum()

        majorLocator = FixedLocator(x_indices)

        ax.xaxis.set_major_locator(majorLocator)
    plt.xticks(rotation=90)

    return ax


def hist_elemental_prevalence(
    formulas: ElemValues,
    count_mode: CountMode = "element_composition",
    log: bool = False,
    keep_top: int = None,
    ax: plt.Axes = None,
    bar_values: Literal["percent", "count", None] = "percent",
    h_offset: int = 0,
    v_offset: int = 10,
    rotation: int = 45,
    **kwargs: Any,
) -> plt.Axes:
    """Plots a histogram of the prevalence of each element in a materials dataset.

    Adapted from https://github.com/kaaiian/ML_figures (https://git.io/JmbaI).

    Args:
        formulas (list[str]): compositional strings, e.g. ["Fe2O3", "Bi2Te3"].
        count_mode ('composition' | 'fractional_composition' | 'reduced_composition'):
            Reduce or normalize compositions before counting. See count_elements() for
            details. Only used when elem_values is list of composition strings/objects.
        log (bool, optional): Whether y-axis is log or linear. Defaults to False.
        keep_top (int | None): Display only the top n elements by prevalence.
        ax (Axes): matplotlib Axes on which to plot. Defaults to None.
        bar_values ('percent'|'count'|None): 'percent' (default) annotates bars with the
            percentage each element makes up in the total element count. 'count'
            displays count itself. None removes bar labels.
        h_offset (int): Horizontal offset for bar height labels. Defaults to 0.
        v_offset (int): Vertical offset for bar height labels. Defaults to 10.
        rotation (int): Bar label angle. Defaults to 45.
        **kwargs (int): Keyword arguments passed to pandas.Series.plot.bar().

    Returns:
        ax: The plot's matplotlib Axes.
    """
    ax = ax or plt.gca()

    elem_counts = count_elements(formulas, count_mode)
    non_zero = elem_counts[elem_counts > 0].sort_values(ascending=False)
    if keep_top is not None:
        non_zero = non_zero.head(keep_top)
        ax.set_title(f"Top {keep_top} Elements")

    non_zero.plot.bar(width=0.7, edgecolor="black", ax=ax, **kwargs)

    if log:
        ax.set(yscale="log", ylabel="log(Element Count)")
    else:
        ax.set(title="Element Count")

    if bar_values is not None:
        if bar_values == "percent":
            sum_elements = non_zero.sum()
            labels = [f"{el / sum_elements:.1%}" for el in non_zero.values]
        else:
            labels = non_zero.astype(int).to_list()
        annotate_bars(
            ax, labels=labels, h_offset=h_offset, v_offset=v_offset, rotation=rotation
        )

    return ax
