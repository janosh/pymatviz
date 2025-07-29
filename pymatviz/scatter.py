"""Parity and density plots."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats
from plotly.subplots import make_subplots

import pymatviz as pmv
from pymatviz.process_data import bin_df_cols, df_to_arrays


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import ArrayLike


def _get_axis_labels(
    x: ArrayLike | str, y: ArrayLike | str, df: pd.DataFrame | None = None
) -> tuple[str, str]:
    """Extract axis labels from data or column names."""
    if df is not None:
        xlabel = getattr(df[x], "name", x)
        ylabel = getattr(df[y], "name", y)
    else:
        xlabel = getattr(x, "name", x if isinstance(x, str) else "Actual")
        ylabel = getattr(y, "name", y if isinstance(y, str) else "Predicted")
    return xlabel, ylabel


def _create_marginal_subplots() -> go.Figure:
    """Create subplot structure for marginal histograms."""
    return make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.85, 0.15],
        row_heights=[0.15, 0.85],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )


def _add_marginal_histograms(
    subplot_fig: go.Figure,
    xs: np.ndarray,
    ys: np.ndarray,
    bins: int,
    xlabel: str,
    ylabel: str,
) -> None:
    """Add marginal histograms to subplot figure."""
    # Top histogram (x-axis marginal)
    hist_x, bin_edges_x = np.histogram(xs, bins=bins)
    bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
    subplot_fig.add_bar(
        x=bin_centers_x,
        y=hist_x,
        opacity=0.7,
        marker_line_width=0,
        showlegend=False,
        row=1,
        col=1,
    )

    # Right histogram (y-axis marginal)
    hist_y, bin_edges_y = np.histogram(ys, bins=bins)
    bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
    subplot_fig.add_bar(
        x=hist_y,
        y=bin_centers_y,
        opacity=0.7,
        marker_line_width=0,
        orientation="h",
        showlegend=False,
        row=2,
        col=2,
    )

    # Remove lines from all axes
    subplot_fig.update_layout(
        showlegend=False,
        xaxis=dict(showline=False, title=xlabel, showticklabels=False),
        yaxis=dict(showline=False, showticklabels=True),
        xaxis2=dict(showline=False, showticklabels=True),
        yaxis2=dict(showline=False, title=ylabel, showticklabels=False),
        xaxis3=dict(showline=False),
        yaxis3=dict(showline=False),
        xaxis4=dict(showline=False),
        yaxis4=dict(showline=False, showticklabels=False),
    )


def density_scatter(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    density: Literal["kde", "empirical"] | None = None,
    log_density: bool | None = None,
    n_bins: int | None | Literal[False] = None,
    bin_counts_col: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] = True,
    stats: bool | dict[str, Any] = True,
    colorbar_kwargs: dict[str, Any] | None = None,
    hover_format: str = ".3f",
    **kwargs: Any,
) -> go.Figure:
    """Scatter plot colored by density using Plotly.

    This function uses binning to reduce the number of points plotted which enables
    plotting millions of data points and reduced file size for interactive plots.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns. Defaults to None.
        density ('kde' | 'empirical'): Determines the method for calculating and
            displaying density. Default is 'empirical' when n_bins is provided,
            else 'kde' for kernel density estimation.
        log_density (bool | None): Whether to apply logarithmic scaling to density.
            If None, automatically set based on density range.
        n_bins (int | None | False, optional): Number of bins for histogram.
            If None, automatically enables binning mode if the number of datapoints
            exceeds 1000, else defaults to False (no binning).
        bin_counts_col (str, optional): Column name for bin counts. Defaults to
            "Point<br>Density". Will be used as color bar title.
        xlabel (str, optional): x-axis label. Auto-detected from data if None.
        ylabel (str, optional): y-axis label. Auto-detected from data if None.
        identity_line (bool | dict[str, Any], optional): Whether to add a parity line
            (y = x). Defaults to True. Pass a dict to customize line properties.
        best_fit_line (bool | dict[str, Any], optional): Whether to add a best-fit line.
            Defaults to True. Pass a dict to customize line properties.
        stats (bool | dict[str, Any], optional): Whether to display a text box with MAE
            and R^2. Defaults to True. Can be dict to pass kwargs to annotate_metrics().
        colorbar_kwargs (dict, optional): Passed to fig.layout.coloraxis.colorbar.
        hover_format (str, optional): Format specifier for the point density values.
        **kwargs: Passed to px.scatter().

    Returns:
        go.Figure: The plot object.
    """
    if not isinstance(stats, bool | dict):
        raise TypeError(f"stats must be bool or dict, got {type(stats)} instead.")

    # Convert arrays to DataFrame if needed
    if df is None:
        xs, ys = df_to_arrays(df, x, y)
        if xlabel is None or ylabel is None:
            auto_xlabel, auto_ylabel = _get_axis_labels(x, y)
            xlabel = xlabel or auto_xlabel
            ylabel = ylabel or auto_ylabel

        # Create column names for DataFrame
        x_col = xlabel if isinstance(xlabel, str) else "x"
        y_col = ylabel if isinstance(ylabel, str) else "y"
        df_data = pd.DataFrame({x_col: xs, y_col: ys})
        x, y = x_col, y_col
    else:
        df_data = df
    if xlabel is None or ylabel is None:
        auto_xlabel, auto_ylabel = _get_axis_labels(x, y, df)
        xlabel = xlabel or auto_xlabel
        ylabel = ylabel or auto_ylabel

    return density_scatter_plotly(
        df_data,
        x=x,
        y=y,
        density=density,
        log_density=log_density,
        identity_line=identity_line,
        best_fit_line=best_fit_line,
        stats=stats,
        n_bins=n_bins,
        bin_counts_col=bin_counts_col,
        colorbar_kwargs=colorbar_kwargs,
        hover_format=hover_format,
        **kwargs,
    )


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
            "Point<br>Density". Will be used as color bar title.
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
    bin_counts_col = bin_counts_col or "Point<br>Density"

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
            trace.hovertemplate = (
                f"{x}: %{{x}}<br>{y}: %{{y}}<br>{bin_counts_col}: "
                f"%{{customdata[0]:{hover_format}}}<extra></extra>"
            )

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
    # Handle empty dataframes
    if len(df) == 0:
        df_empty = df.copy()
        df_empty[bin_counts_col] = []
        return df_empty

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
        df_plot = df.copy()
        values = df[[x, y]].dropna().T
        if density == "kde" and len(values.columns) > 1:
            model_kde = scipy.stats.gaussian_kde(values)
            df_plot[bin_counts_col] = model_kde(df_plot[[x, y]].T)
        else:
            if density == "kde" and len(values.columns) <= 1:
                warnings.warn(
                    "Not enough data points for KDE, using empirical density",
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


def density_hexbin(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    weights: ArrayLike | None = None,
    gridsize: int = 75,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] = True,
    stats: bool | dict[str, Any] = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Hexagonal-grid scatter plot colored by point density using Plotly.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns. Defaults to None.
        weights (array, optional): If given, these values are accumulated in the bins.
            Otherwise, every point has value 1. Must be of the same length as x and y.
        gridsize (int, optional): Number of hexagons in the x and y directions.
            Defaults to 75.
        identity_line (bool | dict[str, Any], optional): Whether to add a parity line
            (y = x). Defaults to True. Pass a dict to customize line properties.
        best_fit_line (bool | dict[str, Any], optional): Whether to add a best-fit line.
            Defaults to True. Pass a dict to customize line properties.
        stats (bool | dict[str, Any], optional): Whether to display a text box with MAE
            and R^2. Defaults to True.
        xlabel (str, optional): x-axis label. Auto-detected if None.
        ylabel (str, optional): y-axis label. Auto-detected if None.
        colorbar_kwargs (dict, optional): Passed to fig.layout.coloraxis.colorbar.
        **kwargs: Additional keyword arguments passed to go.Scatter().

    Returns:
        go.Figure: Plotly Figure object
    """
    if not isinstance(stats, bool | dict):
        raise TypeError(f"stats must be bool or dict, got {type(stats)} instead.")

    xs, ys = df_to_arrays(df, x, y)
    xlabel, ylabel = _get_axis_labels(x, y, df)

    # Use numpy's histogram2d for initial binning, then convert to hex coordinates
    hist, x_edges, y_edges = np.histogram2d(xs, ys, bins=gridsize, weights=weights)

    # Create hexagonal grid from rectangular bins
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Convert to hexagonal tessellation pattern
    hex_x, hex_y, hex_counts = [], [], []
    hex_width = x_edges[1] - x_edges[0]

    for i, x_center in enumerate(x_centers):
        for j, y_center in enumerate(y_centers):
            count = hist[i, j]
            if count > 0:
                # Apply hexagonal offset for alternating rows
                x_offset = hex_width / 2 if j % 2 == 1 else 0
                hex_x.append(x_center + x_offset)
                hex_y.append(y_center)
                hex_counts.append(count)

    x_plot, y_plot, z_plot = np.array(hex_x), np.array(hex_y), np.array(hex_counts)

    # Create the scatter plot with hexagon markers
    fig = go.Figure()

    # Calculate marker size to prevent overlap
    # Make markers much smaller than the hex boundaries to avoid visual overlap
    # Scale inversely with gridsize - more hexagons = smaller markers
    base_size = 400 / gridsize  # Doubled the base scaling factor
    marker_size = min(12, max(4, base_size))  # Doubled size range: 4-12

    scatter_defaults = dict(
        mode="markers",
        marker=dict(
            size=marker_size,
            color=z_plot,
            colorscale="Viridis",
            colorbar=dict(title="Density"),
            symbol="hexagon",
            line=dict(width=0),  # Remove marker borders to eliminate gaps
        ),
        hovertemplate=f"{xlabel}: %{{x}}<br>{ylabel}: %{{y}}<br>Density: "
        f"%{{marker.color}}<extra></extra>",
    )

    fig.add_scatter(
        x=x_plot,
        y=y_plot,
        **scatter_defaults | kwargs,
    )

    # Update colorbar if custom kwargs provided
    if colorbar_kwargs:
        fig.update_traces(marker_colorbar=colorbar_kwargs)

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=False,
    )

    pmv.powerups.enhance_parity_plot(
        fig,
        xs=xs,
        ys=ys,
        identity_line=identity_line,
        best_fit_line=best_fit_line,
        stats=stats,
    )

    return fig


def density_scatter_with_hist(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    bins: int = 100,
    density: Literal["kde", "empirical"] | None = None,
    log_density: bool | None = None,
    n_bins: int | None | Literal[False] = None,
    **kwargs: Any,
) -> go.Figure:
    """Scatter plot colored by density with marginal histograms along each dimension.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns. Defaults to None.
        bins (int, optional): Number of bins for marginal histograms. Defaults to 100.
        density ("kde" | "empirical" | None): Method for density calculation.
        log_density (bool | None): Whether to log-scale the density colors.
        n_bins (int | None | False): Number of bins for empirical density calculation.
        **kwargs: Additional arguments passed to the main density scatter plot.

    Returns:
        go.Figure: Plotly Figure with marginal histograms.
    """
    arrays = df_to_arrays(df, x, y)
    xs, ys = arrays[0], arrays[1]
    # Ensure we have numpy arrays
    if isinstance(xs, dict):
        xs = next(iter(xs.values()))
    if isinstance(ys, dict):
        ys = next(iter(ys.values()))
    # Convert to numpy arrays
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xlabel, ylabel = _get_axis_labels(x, y, df)

    # Create the main density scatter plot
    scatter_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["marginal_x", "marginal_y"]
    }
    fig = density_scatter(
        x,
        y,
        df=df,
        density=density,
        log_density=log_density,
        n_bins=n_bins,
        xlabel=xlabel,
        ylabel=ylabel,
        **scatter_kwargs,
    )

    # Create subplot with marginals
    subplot_fig = _create_marginal_subplots()

    # Add main plot and marginals
    for trace in fig.data:
        subplot_fig.add_trace(trace, row=2, col=1)
    _add_marginal_histograms(subplot_fig, xs, ys, bins, xlabel, ylabel)

    # Copy colorbar settings
    if hasattr(fig.layout, "coloraxis"):
        subplot_fig.layout.coloraxis = fig.layout.coloraxis

    return subplot_fig


def density_hexbin_with_hist(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    bins: int = 100,
    gridsize: int = 75,
    **kwargs: Any,
) -> go.Figure:
    """Hexagonal-grid scatter plot colored by density with marginal histograms along
    each dimension.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns. Defaults to None.
        bins (int, optional): Number of bins for marginal histograms. Defaults to 100.
        gridsize (int, optional): Number of hexagons in the x and y directions.
            Defaults to 75.
        **kwargs: Passed to density_hexbin().

    Returns:
        go.Figure: Plotly Figure with marginal histograms.
    """
    arrays = df_to_arrays(df, x, y)
    xs, ys = arrays[0], arrays[1]
    # Ensure we have numpy arrays
    if isinstance(xs, dict):
        xs = next(iter(xs.values()))
    if isinstance(ys, dict):
        ys = next(iter(ys.values()))
    # Convert to numpy arrays
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xlabel, ylabel = _get_axis_labels(x, y, df)

    # Create the main hexbin plot
    hex_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["marginal_x", "marginal_y"]
    }
    fig = density_hexbin(
        x, y, df=df, gridsize=gridsize, xlabel=xlabel, ylabel=ylabel, **hex_kwargs
    )

    # Create subplot with marginals
    subplot_fig = _create_marginal_subplots()

    # Add main plot and marginals
    for trace in fig.data:
        subplot_fig.add_trace(trace, row=2, col=1)
    _add_marginal_histograms(subplot_fig, xs, ys, bins, xlabel, ylabel)

    # Copy colorbar settings
    if hasattr(fig.layout, "coloraxis"):
        subplot_fig.layout.coloraxis = fig.layout.coloraxis

    return subplot_fig
