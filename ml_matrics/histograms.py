import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from numpy import ndarray as Array
from scipy.stats import gaussian_kde


def residual_hist(
    y_true: Array, y_pred: Array, ax: Axes = None, xlabel: str = None, **kwargs
) -> Axes:
    """Plot the residual distribution overlayed with a Gaussian kernel
    density estimate.

    Adapted from https://github.com/kaaiian/ML_figures (https://git.io/Jmb2O).

    Args:
        y_true (Array): ground truth targets
        y_pred (Array): model predictions
        ax (Axes, optional): plt axes. Defaults to None.
        xlabel (str, optional): x-axis label. Defaults to None.

    Returns:
        Axes: plt axes with plotted data.
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
    y_true: Array,
    y_pred: Array,
    y_std: Array,
    ax: Axes = None,
    cmap: str = "hot",
    bins: int = 50,
    log: bool = True,
    truth_color: str = "blue",
    **kwargs,
) -> Axes:
    """Plot a histogram of model predictions with bars colored by the average uncertainty of
    predictions in that bin. Overlayed by a more transparent histogram of ground truth values.

    Args:
        y_true (Array): ground truth targets
        y_pred (Array): model predictions
        y_std (Array): model uncertainty
        ax (Axes, optional): plt axes. Defaults to None.
        cmap (str, optional): string identifier of a plt colormap. Defaults to "hot".
        bins (int, optional): Histogram resolution. Defaults to 50.
        log (bool, optional): Whether to log-scale the y-axis. Defaults to True.
        truth_color (str, optional): Face color to use for y_true bars. Defaults to "blue".

    Returns:
        Axes: plt axes with plotted data.
    """

    if ax is None:
        ax = plt.gca()

    cmap = getattr(plt.cm, cmap)
    y_true, y_pred, y_std = np.array([y_true, y_pred, y_std])

    _, bins, bars = ax.hist(
        y_pred, bins=bins, alpha=0.8, label=r"$y_\mathrm{pred}$", **kwargs
    )
    ax.figure.set
    ax.hist(
        y_true,
        bins=bins,
        alpha=0.2,
        color=truth_color,
        label=r"$y_\mathrm{true}$",
        **kwargs,
    )

    for xmin, xmax, rect in zip(bins, bins[1:], bars.patches):

        y_preds_in_rect = np.logical_and(y_pred > xmin, y_pred < xmax).nonzero()

        color_value = y_std[y_preds_in_rect].mean()

        rect.set_color(cmap(color_value))

    if log:
        plt.yscale("log")
    ax.legend(frameon=False)

    norm = plt.cm.colors.Normalize(vmax=y_std.max(), vmin=y_std.min())
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), pad=0.075)
    cbar.outline.set_linewidth(1)
    cbar.set_label(r"mean $y_\mathrm{std}$ of prediction in bin")
    cbar.ax.yaxis.set_ticks_position("left")

    ax.figure.set_size_inches(12, 7)

    return ax


def spacegroup_hist(spacegroups: Array, ax: Axes = None, **kwargs) -> Axes:
    """Plot a histogram of spacegroups shaded by crystal system.

    (triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic)

    Args:
        spacegroups (Array): list, tuple np.array or pd.Series of spacegroup numbers.
        ax (Axes, optional): plt axes. Defaults to None.
        kwargs: Keywords passed to pd.Series.plot.bar().

    Returns:
        Axes: plt axes
    """
    if ax is None:
        ax = plt.gca()

    pd.Series(spacegroups).value_counts().reindex(range(230), fill_value=0).plot.bar(
        figsize=[15, 4], width=1, rot=0, ax=ax, **kwargs
    )

    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/fill_between_demo
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    # https://git.io/JYJcs
    crystal_systems = {
        "tri-/monoclinic": ["red", (1, 15)],
        "orthorhombic": ["blue", (16, 74)],
        "tetragonal": ["green", (75, 142)],
        "trigonal": ["orange", (143, 167)],
        "hexagonal": ["purple", (168, 194)],
        "cubic": ["yellow", (195, 230)],
    }

    for name, [color, rng] in crystal_systems.items():
        x0, x1 = rng
        for patch in ax.patches[0 if x0 == 1 else x0 : x1 + 1]:
            patch.set_facecolor(color)
        ax.text(
            sum(rng) / 2,
            0.95,
            name,
            rotation=90,
            transform=trans,
            verticalalignment="top",
            horizontalalignment="center",
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
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))

    return ax
