from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from pymatviz.utils import annotate_metrics, df_to_arrays, with_hist


if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.gridspec import GridSpec
    from numpy.typing import ArrayLike


def hist_density(
    x: ArrayLike | str,
    y: ArrayLike | str,
    df: pd.DataFrame | None = None,
    sort: bool = True,
    bins: int = 100,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Return an approximate density of 2d points.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns. Defaults to None.
        sort (bool, optional): Whether to sort points by density so that densest points
            are plotted last. Defaults to True.
        bins (int, optional): Number of bins (histogram resolution). Defaults to 100.

    Returns:
        tuple[array, array]: x and y values (sorted by density) and density itself
    """
    x, y = df_to_arrays(df, x, y)

    data, x_e, y_e = np.histogram2d(x, y, bins=bins)

    zs = scipy.interpolate.interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = zs.argsort()
        x, y, zs = x[idx], y[idx], zs[idx]

    return x, y, zs


def density_scatter(
    x: ArrayLike | str,
    y: ArrayLike | str,
    df: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    sort: bool = True,
    log_cmap: bool = True,
    density_bins: int = 100,
    xlabel: str | None = None,
    ylabel: str | None = None,
    identity: bool = True,
    stats: bool | dict[str, Any] = True,
    **kwargs: Any,
) -> plt.Axes:
    """Scatter plot colored (and optionally sorted) by density.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns. Defaults to None.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        sort (bool, optional): Whether to sort the data. Defaults to True.
        log_cmap (bool, optional): Whether to log the color scale. Defaults to True.
        density_bins (int, optional): How many density_bins to use for the density
            histogram, i.e. granularity of the density color scale. Defaults to 100.
        xlabel (str, optional): x-axis label. Defaults to "Actual".
        ylabel (str, optional): y-axis label. Defaults to "Predicted".
        identity (bool, optional): Whether to add an identity/parity line (y = x).
            Defaults to True.
        stats (bool | dict[str, Any], optional): Whether to display a text box with MAE
            and R^2. Defaults to True. Can be dict to pass kwargs to annotate_metrics().
            E.g. stats=dict(loc="upper left", prefix="Title", prop=dict(fontsize=16)).
        **kwargs: Additional keyword arguments to pass to ax.scatter(). E.g. cmap to
            change the color map.

    Returns:
        ax: The plot's matplotlib Axes.
    """
    if not isinstance(stats, (bool, dict)):
        raise TypeError(f"stats must be bool or dict, got {type(stats)} instead.")
    if xlabel is None:
        xlabel = getattr(x, "name", x if isinstance(x, str) else "Actual")
    if ylabel is None:
        ylabel = getattr(y, "name", y if isinstance(y, str) else "Predicted")

    x, y = df_to_arrays(df, x, y)
    ax = ax or plt.gca()

    x, y, cs = hist_density(x, y, sort=sort, bins=density_bins)

    norm = mpl.colors.LogNorm() if log_cmap else None

    ax.scatter(x, y, c=cs, norm=norm, **kwargs)

    if identity:
        x_mid = sum(ax.get_xlim()) / 2
        ax.axline(
            (x_mid, x_mid),
            slope=1,
            alpha=0.5,
            zorder=0,
            linestyle="dashed",
            color="black",
        )

    if stats:
        annotate_metrics(x, y, ax=ax, **(stats if isinstance(stats, dict) else {}))

    ax.set(xlabel=xlabel, ylabel=ylabel)

    return ax


def scatter_with_err_bar(
    x: ArrayLike | str,
    y: ArrayLike | str,
    df: pd.DataFrame | None = None,
    xerr: ArrayLike | None = None,
    yerr: ArrayLike | None = None,
    ax: plt.Axes | None = None,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    title: str | None = None,
    **kwargs: Any,
) -> plt.Axes:
    """Scatter plot with optional x- and/or y-error bars. Useful when passing model
    uncertainties as yerr=y_std for checking if uncertainty correlates with error, i.e.
    if points farther from the parity line have larger uncertainty.

    Args:
        x (array | str): x-values or dataframe column name
        y (array | str): y-values or dataframe column name
        df (pd.DataFrame, optional): DataFrame with x and y columns. Defaults to None.
        xerr (array, optional): Horizontal error bars. Defaults to None.
        yerr (array, optional): Vertical error bars. Defaults to None.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        xlabel (str, optional): x-axis label. Defaults to "Actual".
        ylabel (str, optional): y-axis label. Defaults to "Predicted".
        title (str, optional): Plot tile. Defaults to None.
        **kwargs: Additional keyword arguments to pass to ax.errorbar().

    Returns:
        ax: The plot's matplotlib Axes.
    """
    x, y = df_to_arrays(df, x, y)
    ax = ax or plt.gca()

    styles = dict(markersize=6, fmt="o", ecolor="g", capthick=2, elinewidth=2)
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, **kwargs, **styles)

    # identity line
    ax.axline((0, 0), (1, 1), alpha=0.5, zorder=0, linestyle="dashed", color="black")

    annotate_metrics(x, y, ax=ax)

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    return ax


def density_hexbin(
    x: ArrayLike | str,
    y: ArrayLike | str,
    df: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    weights: ArrayLike | None = None,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    **kwargs: Any,
) -> plt.Axes:
    """Hexagonal-grid scatter plot colored by point density or by density in third
    dimension passed as weights.

    Args:
        x (array): x-values or dataframe column name.
        y (array): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns. Defaults to None.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        weights (array, optional): If given, these values are accumulated in the bins.
            Otherwise, every point has value 1. Must be of the same length as x and y.
            Defaults to None.
        xlabel (str, optional): x-axis label. Defaults to "Actual".
        ylabel (str, optional): y-axis label. Defaults to "Predicted".
        **kwargs: Additional keyword arguments to pass to ax.hexbin().

    Returns:
        ax: The plot's matplotlib Axes.
    """
    x, y = df_to_arrays(df, x, y)
    ax = ax or plt.gca()

    # the scatter plot
    hexbin = ax.hexbin(x, y, gridsize=75, mincnt=1, bins="log", C=weights, **kwargs)

    cb_ax = ax.inset_axes([0.95, 0.03, 0.03, 0.7])  # [x, y, width, height]
    plt.colorbar(hexbin, cax=cb_ax)
    cb_ax.yaxis.set_ticks_position("left")

    # identity line
    ax.axline((0, 0), (1, 1), alpha=0.5, zorder=0, linestyle="dashed", color="black")

    annotate_metrics(x, y, ax=ax, loc="upper left")

    ax.set(xlabel=xlabel, ylabel=ylabel)

    return ax


def density_scatter_with_hist(
    x: ArrayLike | str,
    y: ArrayLike | str,
    df: pd.DataFrame | None = None,
    cell: GridSpec | None = None,
    bins: int = 100,
    **kwargs: Any,
) -> plt.Axes:
    """Scatter plot colored (and optionally sorted) by density with histograms along
    each dimension.
    """
    x, y = df_to_arrays(df, x, y)
    ax_scatter = with_hist(x, y, cell, bins)
    return density_scatter(x, y, ax=ax_scatter, **kwargs)


def density_hexbin_with_hist(
    x: ArrayLike | str,
    y: ArrayLike | str,
    df: pd.DataFrame | None = None,
    cell: GridSpec | None = None,
    bins: int = 100,
    **kwargs: Any,
) -> plt.Axes:
    """Hexagonal-grid scatter plot colored by density or by third dimension passed
    color_by with histograms along each dimension.
    """
    x, y = df_to_arrays(df, x, y)
    ax_scatter = with_hist(x, y, cell, bins)
    return density_hexbin(x, y, ax=ax_scatter, **kwargs)


def residual_vs_actual(
    y_true: ArrayLike | str,
    y_pred: ArrayLike | str,
    df: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    xlabel: str = r"Actual value",
    ylabel: str = r"Residual ($y_\mathrm{true} - y_\mathrm{pred}$)",
    **kwargs: Any,
) -> plt.Axes:
    r"""Plot targets on the x-axis vs residuals (y_err = y_true - y_pred) on the y-axis.

    Args:
        y_true (array): Ground truth values
        y_pred (array): Model predictions
        df (pd.DataFrame, optional): DataFrame with y_true and y_pred columns.
            Defaults to None.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        xlabel (str, optional): x-axis label. Defaults to "Actual value".
        ylabel (str, optional): y-axis label. Defaults to
            `'Residual ($y_\mathrm{true} - y_\mathrm{pred}$)'`.
        **kwargs: Additional keyword arguments passed to plt.plot()

    Returns:
        ax: The plot's matplotlib Axes.
    """
    y_true, y_pred = df_to_arrays(df, y_true, y_pred)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    ax = ax or plt.gca()

    y_err = y_true - y_pred

    ax.plot(y_true, y_err, "o", alpha=0.5, label=None, mew=1.2, ms=5.2, **kwargs)
    ax.axline(
        [1, 0], [2, 0], linestyle="dashed", color="black", alpha=0.5, label="ideal"
    )

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="lower right")

    return ax
