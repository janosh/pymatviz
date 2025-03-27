"""Parity, residual and density plots."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.interpolate
import scipy.stats
from matplotlib.colors import LogNorm

import pymatviz as pmv
from pymatviz.utils import bin_df_cols, df_to_arrays


if TYPE_CHECKING:
    from typing import Any, Literal

    from matplotlib.gridspec import GridSpec
    from numpy.typing import ArrayLike


def _hist_density(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    sort: bool = True,
    bins: int = 100,
    method: str = "nearest",
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Return an approximate density of 2d points.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns. Defaults to None.
        sort (bool, optional): Whether to sort points by density so that densest points
            are plotted last. Defaults to True.
        bins (int, optional): Number of bins (histogram resolution). Defaults to 100.
        method (str, optional): Interpolation method. Defaults to "nearest".
            See scipy.interpolate.interpn() for options.

    Returns:
        tuple[np.array, np.array, np.array]: x and y values (sorted by density) and
            density itself
    """
    xs, ys = df_to_arrays(df, x, y)

    counts, x_bins, y_bins = np.histogram2d(xs, ys, bins=bins)

    # get bin centers
    points = 0.5 * (x_bins[1:] + x_bins[:-1]), 0.5 * (y_bins[1:] + y_bins[:-1])
    zs = scipy.interpolate.interpn(
        points, counts, np.vstack([xs, ys]).T, method=method, bounds_error=False
    )

    # sort points by density, so that the densest points are plotted last
    if sort:
        sort_idx = zs.argsort()
        xs, ys, zs = xs[sort_idx], ys[sort_idx], zs[sort_idx]

    return xs, ys, zs


def density_scatter(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    log_density: bool = True,
    hist_density_kwargs: dict[str, Any] | None = None,
    color_bar: bool | dict[str, Any] = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] = True,
    stats: bool | dict[str, Any] = True,
    **kwargs: Any,
) -> plt.Axes:
    """Scatter plot colored by density using matplotlib backend.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns. Defaults to None.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.
        log_density (bool, optional): Whether to log the density color scale.
            Defaults to True.
        hist_density_kwargs (dict, optional): Passed to hist_density(). Use to change
            sort (by density, default True), bins (default 100), or method (for
            interpolation, default "nearest"). matplotlib backend only.
        color_bar (bool | dict, optional): Whether to add a color bar. Defaults to True.
            If dict, unpacked into ax.figure.colorbar(). E.g. dict(label="Density").
        xlabel (str, optional): x-axis label. Defaults to "Actual".
        ylabel (str, optional): y-axis label. Defaults to "Predicted".
        identity_line (bool | dict[str, Any], optional): Whether to add an parity line
            (y = x). Defaults to True. Pass a dict to customize line properties.
        best_fit_line (bool | dict[str, Any], optional): Whether to add a best-fit line.
            Defaults to True. Pass a dict to customize line properties.
        stats (bool | dict[str, Any], optional): Whether to display a text box with MAE
            and R^2. Defaults to True. Can be dict to pass kwargs to annotate_metrics().
            E.g. stats=dict(loc="upper left", prefix="Title", prop=dict(fontsize=16)).
        **kwargs: Passed to ax.scatter().

    Returns:
        plt.Axes: The plot object.
    """
    if not isinstance(stats, bool | dict):
        raise TypeError(f"stats must be bool or dict, got {type(stats)} instead.")
    if xlabel is None:
        xlabel = getattr(x, "name", x if isinstance(x, str) else "Actual")
    if ylabel is None:
        ylabel = getattr(y, "name", y if isinstance(y, str) else "Predicted")

    xs, ys = df_to_arrays(df, x, y)
    ax = ax or plt.gca()

    xs, ys, cs = _hist_density(xs, ys, **(hist_density_kwargs or {}))

    # decrease marker size
    scatter_defaults = dict(s=6, norm=LogNorm() if log_density else None)
    ax.scatter(xs, ys, c=cs, **scatter_defaults | kwargs)

    ax.set(xlabel=xlabel, ylabel=ylabel)

    if color_bar:
        kwds = dict(label="Density") if color_bar is True else color_bar
        color_bar = ax.figure.colorbar(ax.collections[0], **kwds)

    pmv.powerups.enhance_parity_plot(
        ax,
        xs=xs,
        ys=ys,
        identity_line=identity_line,
        best_fit_line=best_fit_line,
        stats=stats,
    )

    return ax


def density_scatter_plotly(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    density: Literal["kde", "empirical"] | None = None,
    log_density: bool | None = None,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] | None = None,
    stats: bool | dict[str, Any] = True,
    n_bins: int | None | Literal[False] = None,
    bin_counts_col: str | None = None,
    facet_col: str | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    hover_format: str = ".3f",
    **kwargs: Any,
) -> go.Figure:
    """Scatter plot colored by density using plotly backend.

    This function uses binning as implemented in bin_df_cols() to reduce the number of
    points plotted which enables plotting millions of data points and reduced file size
    for interactive plots. All outlier points will be plotted as is but overlapping
    points (tolerance for overlap determined by n_bins) will be merged into a single
    point with a new column bin_counts_col counting the number of points in that bin.

    Args:
        x (str): x-values dataframe column name.
        y (str): y-values dataframe column name.
        df (pd.DataFrame): DataFrame with x and y columns.
        density ('kde' | 'interpolate' | 'empirical'): Determines the method for
            calculating and displaying density. Default is 'empirical' when n_bins
            is provided, else 'kde' for kernel density estimation.
        log_density (bool | None): Whether to apply logarithmic scaling to density.
            If None, automatically set based on density range.
        identity_line (bool | dict[str, Any], optional): Whether to add a parity line
            (y = x). Defaults to True. Pass a dict to customize line properties.
        best_fit_line (bool | dict[str, Any], optional): Whether to add a best-fit line.
            Defaults to True. Pass a dict to customize line properties.
        stats (bool | dict[str, Any], optional): Whether to display a text box with MAE
            and R^2. Defaults to True. Can be dict to pass kwargs to annotate_metrics().
            E.g. stats=dict(loc="upper left", prefix="Title", font=dict(size=16)).
        n_bins (int | None | False, optional): Number of bins for histogram.
            If None, automatically enables binning mode if the number of datapoints
            exceeds 1000, else defaults to False (no binning).
            If int, uses that number of bins.
            If False, performs no binning. Defaults to None.
        bin_counts_col (str, optional): Column name for bin counts. Defaults to
            "Point Density". Will be used as color bar title.
        facet_col (str | None, optional): Column name to use for creating faceted
            subplots. If provided, the plot will be split into multiple subplots based
            on unique values in this column. Defaults to None.
        colorbar_kwargs (dict, optional): Passed to fig.layout.coloraxis.colorbar.
            E.g. dict(thickness=15) to make colorbar thinner.
        hover_format (str, optional): Format specifier for the point density values in
            the hover tooltip. Defaults to ".3f" (3 decimal places). Can be any valid
            Python format specifier, e.g. ".2e" for scientific notation, ".0f" for
            integers, etc. See https://docs.python.org/3/library/string.html#format-specification-mini-language
            for more options.
        **kwargs: Passed to px.scatter().

    Returns:
        go.Figure: The plot object.
    """
    bin_counts_col = bin_counts_col or "Point Density"

    if not isinstance(stats, bool | dict):
        raise TypeError(f"stats must be bool or dict, got {type(stats)} instead.")

    if n_bins is None:  # auto-enable binning depending on data size
        n_bins = 200 if len(df) > 1000 else False

    density = density or ("empirical" if n_bins else "kde")

    if facet_col:
        # Group the dataframe based on the facet column
        grouped = df.groupby(facet_col)
        binned_dfs = []

        for group_name, group_df in grouped:
            binned_df = _bin_and_calculate_density(
                group_df, x, y, density, n_bins, bin_counts_col
            )
            binned_df[facet_col] = group_name  # Add the facet column back
            binned_dfs += [binned_df]

        # Merge all binned dataframes
        df_plot = pd.concat(binned_dfs, ignore_index=True)
    else:
        df_plot = _bin_and_calculate_density(df, x, y, density, n_bins, bin_counts_col)

    color_vals = df_plot[bin_counts_col]

    if log_density is None:
        log_density = np.log10(color_vals.max()) - np.log10(color_vals.min()) > 2

    if log_density:
        color_vals = np.log10(color_vals + 1)

    # Check if a custom hovertemplate is provided in kwargs
    custom_hovertemplate = kwargs.pop("hovertemplate", None)

    kwargs = dict(color_continuous_scale="Viridis") | kwargs

    fig = px.scatter(
        df_plot,
        x=x,
        y=y,
        color=color_vals,
        facet_col=facet_col,
        custom_data=[bin_counts_col],
        **kwargs,
    )

    colorbar_defaults = dict(thickness=15)
    fig.layout.coloraxis.colorbar.update(colorbar_defaults | (colorbar_kwargs or {}))

    # Apply custom hover template if provided
    if custom_hovertemplate:
        for trace in fig.data:
            trace.hovertemplate = custom_hovertemplate

    # For non-log density case, apply hover formatting directly
    elif not log_density:
        for trace in fig.data:
            # Create a custom hover template with formatted point density
            trace.hovertemplate = f"{x}: %{{x}}<br>{y}: %{{y}}<br>{bin_counts_col}: "
            f"%{{customdata[0]:{hover_format}}}<extra></extra>"

    if log_density:
        _update_colorbar_for_log_density(
            fig, color_vals, bin_counts_col, x, y, hover_format, custom_hovertemplate
        )

    pmv.powerups.enhance_parity_plot(
        fig, identity_line=identity_line, best_fit_line=best_fit_line, stats=stats
    )
    return fig


def _bin_and_calculate_density(
    df: pd.DataFrame,
    x: str,
    y: str,
    density: str,
    n_bins: int | None | Literal[False],
    bin_counts_col: str,
) -> pd.DataFrame:
    """Helper function to bin data and calculate density."""
    if n_bins:
        density_col = "bin_counts_kde" if density == "kde" else ""
        df_plot = bin_df_cols(
            df,
            bin_by_cols=[x, y],
            n_bins=n_bins,
            bin_counts_col=bin_counts_col,
            density_col=density_col,
        ).sort_values(bin_counts_col)
        # sort by counts so densest points are plotted last

        if density_col in df_plot:
            df_plot[bin_counts_col] = df_plot[density_col]
        elif density != "empirical":
            raise ValueError(f"Unknown {density=}")
    else:
        df_plot = df
        values = df[[x, y]].dropna().T
        if density == "kde":
            model_kde = scipy.stats.gaussian_kde(values)
            df_plot[bin_counts_col] = model_kde(df_plot[[x, y]].T)
        else:
            warnings.warn(
                f"no need to use density scatter if binning is disabled and {density=}",
                stacklevel=2,
            )
            df_plot[bin_counts_col] = np.ones(len(df_plot))

    return df_plot


def _update_colorbar_for_log_density(
    fig: go.Figure,
    color_vals: np.ndarray,
    bin_counts_col: str,
    x: str = "x",
    y: str = "y",
    hover_format: str = ".3f",
    custom_hovertemplate: str | None = None,
) -> None:
    """Helper function to update colorbar for log density.

    Creates evenly spaced tick labels on a logarithmic scale across the full data range.
    Also updates hover tooltip to display point density in specified format.

    Args:
        fig (go.Figure): The plotly figure to update
        color_vals (np.ndarray): The logged color values
        bin_counts_col (str): The name of the column containing bin counts
        x (str, optional): Name of the x-axis variable. Defaults to "x".
        y (str, optional): Name of the y-axis variable. Defaults to "y".
        hover_format (str, optional): Format specifier for the point density values in
            the hover tooltip. Defaults to ".3f" (3 decimal places).
        custom_hovertemplate (str | None, optional): Custom hover template provided by
            the user. If provided, the hover template will be modified to include
            point density information.
    """
    from pymatviz.utils.data import si_fmt

    # Get the actual (non-logged) min and max counts from the original data
    # color_vals are already log10(counts + 1), so we need to convert back
    logged_min = color_vals.min()
    logged_max = color_vals.max()

    # The actual values in the data (before logging)
    actual_min = max(10 ** (logged_min) - 1, 1)  # Ensure min is at least 1
    actual_max = 10 ** (logged_max) - 1

    # Generate tick positions that will be evenly spaced on a log scale
    # For large ranges, use powers of 10 and intermediate values
    if np.log10(actual_max) - np.log10(actual_min) > 2:
        # Start with powers of 10
        decades = range(
            int(np.floor(np.log10(actual_min))), int(np.ceil(np.log10(actual_max))) + 1
        )
        tick_values = [10**decade for decade in decades]

        # Add intermediate values (2x and 5x) for each decade
        intermediate_ticks = []
        for power in decades[:-1]:  # Skip the last decade
            intermediate_ticks.extend([2 * 10**power, 5 * 10**power])

        tick_values.extend(intermediate_ticks)
        tick_values = sorted(tick_values)

        # Filter out values outside our range
        tick_values = [v for v in tick_values if actual_min <= v <= actual_max]
    else:
        # For smaller ranges, use more points
        num_ticks = min(
            10, max(5, int(np.log10(actual_max) - np.log10(actual_min)) * 5)
        )
        tick_values = np.logspace(np.log10(actual_min), np.log10(actual_max), num_ticks)

    # Format tick labels using si_fmt for consistent formatting
    # and manually strip trailing zeros for cleaner display
    tick_labels = []
    for val in tick_values:
        # float precision based on magnitude: small (large) values use 1 (0) decimals
        formatted = si_fmt(val, fmt=".1f") if val < 10 else si_fmt(val, fmt=".0f")

        # Remove trailing .0 if present
        formatted = formatted.removesuffix(".0")

        tick_labels.append(formatted)

    # Calculate the tick positions in the transformed (logged) scale
    # The transformation applied is log10(x + 1), so we need to apply the same
    tick_positions = np.log10(np.array(tick_values) + 1)

    # Update the colorbar with the correct tick positions and labels
    colorbar = fig.layout.coloraxis.colorbar
    colorbar.update(tickvals=tick_positions, ticktext=tick_labels)

    # Apply hover formatting to all traces
    for trace in fig.data:
        # Use default template if None, otherwise use custom template
        template = (
            f"{x}: %{{x}}<br>{y}: %{{y}}"
            if custom_hovertemplate is None
            else custom_hovertemplate
        )

        # Split at <extra> tag if present
        parts = template.split("<extra>", 1)
        base = parts[0]
        extra = f"<extra>{parts[1]}" if len(parts) > 1 else "<extra></extra>"

        # Combine template with density info
        trace.hovertemplate = (
            f"{base}<br>{bin_counts_col}: %{{customdata[0]:{hover_format}}}{extra}"
        )

    colorbar.title = bin_counts_col.replace(" ", "<br>")


def scatter_with_err_bar(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    xerr: ArrayLike | None = None,
    yerr: ArrayLike | None = None,
    ax: plt.Axes | None = None,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] = True,
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
        identity_line (bool | dict[str, Any], optional): Whether to add an parity line
            (y = x). Defaults to True. Pass a dict to customize line properties.
        best_fit_line (bool | dict[str, Any], optional): Whether to add a best-fit line.
            Defaults to True. Pass a dict to customize line properties.
        xlabel (str, optional): x-axis label. Defaults to "Actual".
        ylabel (str, optional): y-axis label. Defaults to "Predicted".
        title (str, optional): Plot tile. Defaults to None.
        **kwargs: Additional keyword arguments to pass to ax.errorbar().

    Returns:
        plt.Axes: matplotlib Axes object
    """
    xs, ys = df_to_arrays(df, x, y)
    ax = ax or plt.gca()

    styles = dict(markersize=6, fmt="o", ecolor="g", capthick=2, elinewidth=2)
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, **kwargs, **styles)

    if identity_line:
        pmv.powerups.add_identity_line(
            ax, **(identity_line if isinstance(identity_line, dict) else {})
        )
    if best_fit_line:
        pmv.powerups.add_best_fit_line(
            ax, **(best_fit_line if isinstance(best_fit_line, dict) else {})
        )

    pmv.powerups.annotate_metrics(xs, ys, fig=ax)

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

    return ax


def density_hexbin(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    weights: ArrayLike | None = None,
    gridsize: int = 75,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] = True,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    cbar_label: str | None = "Density",
    # [x, y, width, height] anchored at lower left corner
    cbar_coords: tuple[float, float, float, float] = (0.95, 0.03, 0.03, 0.7),
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
        gridsize (int, optional): Number of hexagons in the x and y directions.
            Defaults to 75.
        identity_line (bool | dict[str, Any], optional): Whether to add an parity line
            (y = x). Defaults to True. Pass a dict to customize line properties.
        best_fit_line (bool | dict[str, Any], optional): Whether to add a best-fit line.
            Defaults to True. Pass a dict to customize line properties.
        xlabel (str, optional): x-axis label. Defaults to "Actual".
        ylabel (str, optional): y-axis label. Defaults to "Predicted".
        cbar_label (str, optional): Color bar label. Defaults to "Density".
        cbar_coords (tuple[float, float, float, float], optional): Color bar position
            and size: [x, y, width, height] anchored at lower left corner. Defaults to
            (0.18, 0.8, 0.42, 0.05).
        **kwargs: Additional keyword arguments to pass to ax.hexbin().

    Returns:
        plt.Axes: matplotlib Axes object
    """
    xs, ys = df_to_arrays(df, x, y)
    ax = ax or plt.gca()

    # the scatter plot
    hexbin = ax.hexbin(
        xs, ys, gridsize=gridsize, mincnt=1, bins="log", C=weights, **kwargs
    )

    cb_ax = ax.inset_axes(cbar_coords)
    plt.colorbar(hexbin, cax=cb_ax)
    cb_ax.yaxis.set_ticks_position("left")
    if cbar_label:
        # make title vertical
        cb_ax.set_title(cbar_label, rotation=90, pad=10)

    if identity_line:
        pmv.powerups.add_identity_line(
            ax, **(identity_line if isinstance(identity_line, dict) else {})
        )
    if best_fit_line:
        pmv.powerups.add_best_fit_line(
            ax, **(best_fit_line if isinstance(best_fit_line, dict) else {})
        )

    pmv.powerups.annotate_metrics(xs, ys, fig=ax, loc="upper left")

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_aspect("equal")

    return ax


def density_scatter_with_hist(
    x: ArrayLike | str,
    y: ArrayLike | str,
    df: pd.DataFrame | None = None,
    cell: GridSpec | None = None,
    bins: int = 100,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> plt.Axes:
    """Scatter plot colored (and optionally sorted) by density with histograms along
    each dimension.
    """
    xs, ys = df_to_arrays(df, x, y)
    ax_scatter = pmv.powerups.with_marginal_hist(xs, ys, cell, bins, fig=ax)
    return density_scatter(xs, ys, ax=ax_scatter, **kwargs)


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
    xs, ys = df_to_arrays(df, x, y)
    ax_scatter = pmv.powerups.with_marginal_hist(xs, ys, cell, bins)
    return density_hexbin(xs, ys, ax=ax_scatter, **kwargs)


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
        plt.Axes: matplotlib Axes object
    """
    y_true, y_pred = df_to_arrays(df, y_true, y_pred)
    if not isinstance(y_true, np.ndarray):
        raise TypeError(f"Expect y_true as np.ndarray, got {type(y_true).__name__}")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"Expect y_pred as np.ndarray, got {type(y_pred).__name__}")
    ax = ax or plt.gca()

    y_err = y_true - y_pred

    ax.plot(y_true, y_err, "o", alpha=0.5, label=None, mew=1.2, ms=5.2, **kwargs)
    ax.axline(
        [1, 0], [2, 0], linestyle="dashed", color="black", alpha=0.5, label="ideal"
    )

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="lower right")

    return ax
