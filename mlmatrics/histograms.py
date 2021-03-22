import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
    cb_ax = inset_axes(ax, width="3%", height="50%", loc="center right")

    norm = plt.cm.colors.Normalize(vmax=y_std.max(), vmin=y_std.min())
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)
    cb_ax.yaxis.set_ticks_position("left")

    return ax
