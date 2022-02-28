from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from scipy.stats import gaussian_kde

from pymatviz.utils import NumArray


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
    spacegroups: Sequence[int],
    show_counts: bool = True,
    show_minor_xticks: bool = False,
    ax: Axes = None,
    **kwargs: Any,
) -> Axes:
    """Plot a histogram of spacegroups shaded by crystal system.

    (triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic)

    Args:
        spacegroups (array): A list of spacegroup numbers.
        show_counts (bool, optional): Whether to count the number of items
            in each crystal system. Defaults to True.
        show_minor_xticks (bool, optional): Whether to render minor x-ticks half way
            through each crystal system. Defaults to False.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        kwargs: Keywords passed to pd.Series.plot.bar().

    Returns:
        ax: The plot's matplotlib Axes.
    """
    if ax is None:
        ax = plt.gca()

    sg_series = pd.Series(spacegroups)

    sg_series.value_counts().reindex(range(230), fill_value=0).plot.bar(
        figsize=[16, 4], width=1, rot=0, ax=ax, **kwargs
    )

    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/fill_between_demo
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    # https://git.io/JYJcs
    crystal_systems: dict[str, tuple[str, tuple[int, int]]] = {
        "tri-/monoclinic": ("red", (1, 15)),
        "orthorhombic": ("blue", (16, 74)),
        "tetragonal": ("green", (75, 142)),
        "trigonal": ("orange", (143, 167)),
        "hexagonal": ("purple", (168, 194)),
        "cubic": ("yellow", (195, 230)),
    }

    if show_counts:
        spacegroup_ranges = [1] + [x[1][1] for x in crystal_systems.values()]
        crys_sys_counts = sg_series.value_counts(bins=spacegroup_ranges, sort=False)
        # reindex needed for crys_sys_counts[cryst_sys] below
        crys_sys_counts.index = crystal_systems.keys()
        ax.set_title("Totals per crystal system", fontdict={"fontsize": 18}, pad=30)

    for cryst_sys, (color, (x0, x1)) in crystal_systems.items():

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
            count = crys_sys_counts[cryst_sys]
            ax.text(
                *[(x0 + x1) / 2, 1.02],
                f"{count:,} ({count/len(spacegroups):.0%})",
                fontdict={"fontsize": 12},
                **text_kwds,
            )

        ax.fill_between(
            [x0 - 1, x1],
            *[0, 1],
            facecolor=color,
            alpha=0.1,
            transform=trans,
            edgecolor="black",
        )

    ax.set(xlim=(0, 230), xlabel="International Spacegroup Number", ylabel="Count")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    majorLocator = FixedLocator([x[1][1] for x in crystal_systems.values()])
    minorLocator = FixedLocator([sum(x[1]) // 2 for x in crystal_systems.values()])

    ax.xaxis.set_major_locator(majorLocator)
    if show_minor_xticks:
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))

    return ax
