from __future__ import annotations

from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator
from pymatgen.core import Structure
from pymatgen.symmetry.groups import SpaceGroup
from scipy.stats import gaussian_kde

from pymatviz.utils import NumArray, get_crystal_sys


def residual_hist(
    y_true: NumArray,
    y_pred: NumArray,
    ax: Axes = None,
    xlabel: str = None,
    **kwargs: Any,
) -> Axes:
    """Plot the residual distribution overlaid with a Gaussian kernel
    density estimate.

    Adapted from https://github.com/kaaiian/ML_figures (https://git.io/Jmb2O).

    Args:
        y_true (array): ground truth targets
        y_pred (array): model predictions
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        xlabel (str, optional): x-axis label. Defaults to None.

    Returns:
        ax: The plot's matplotlib Axes.
    """
    if ax is None:
        ax = plt.gca()

    y_res = y_pred - y_true
    plt.hist(y_res, bins=35, density=True, edgecolor="black", **kwargs)

    # Gaussian kernel density estimation: evaluates the Gaussian
    # probability density estimated based on the points in y_res
    kde = gaussian_kde(y_res)
    x_range = np.linspace(min(y_res), max(y_res), 100)

    label = "Gaussian kernel density estimate"
    plt.plot(x_range, kde(x_range), lw=3, color="red", label=label)

    plt.xlabel(xlabel or r"Residual ($y_\mathrm{test} - y_\mathrm{pred}$)")
    plt.legend(loc=2, framealpha=0.5, handlelength=1)

    return ax


def true_pred_hist(
    y_true: NumArray,
    y_pred: NumArray,
    y_std: NumArray,
    ax: Axes = None,
    cmap: str = "hot",
    bins: int = 50,
    log: bool = True,
    truth_color: str = "blue",
    **kwargs: Any,
) -> Axes:
    """Plot a histogram of model predictions with bars colored by the mean uncertainty of
    predictions in that bin. Overlaid by a more transparent histogram of ground truth
    values.

    Args:
        y_true (array): ground truth targets
        y_pred (array): model predictions
        y_std (array): model uncertainty
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        cmap (str, optional): string identifier of a plt colormap. Defaults to 'hot'.
        bins (int, optional): Histogram resolution. Defaults to 50.
        log (bool, optional): Whether to log-scale the y-axis. Defaults to True.
        truth_color (str, optional): Face color to use for y_true bars.
            Defaults to 'blue'.

    Returns:
        ax: The plot's matplotlib Axes.
    """
    if ax is None:
        ax = plt.gca()

    color_map = getattr(plt.cm, cmap)
    y_true, y_pred, y_std = np.array([y_true, y_pred, y_std])

    _, bin_edges, bars = ax.hist(
        y_pred, bins=bins, alpha=0.8, label=r"$y_\mathrm{pred}$", **kwargs
    )
    ax.figure.set
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

    if log:
        plt.yscale("log")
    ax.legend(frameon=False)

    norm = plt.cm.colors.Normalize(vmax=y_std.max(), vmin=y_std.min())
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map), pad=0.075)
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
    ax: Axes = None,
    **kwargs: Any,
) -> Axes:
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
        kwargs: Keywords passed to pd.Series.plot.bar().

    Returns:
        ax: The plot's matplotlib Axes.
    """
    if ax is None:
        ax = plt.gca()

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
