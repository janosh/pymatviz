"""Parity and density plots."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.stats

from pymatviz.powerups import enhance_parity_plot
from pymatviz.process_data import bin_df_cols, df_to_arrays
from pymatviz.widgets.binned_scatter_plot import BinnedScatterPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


if TYPE_CHECKING:
    from typing import Any, Literal

    from numpy.typing import ArrayLike

DENSITY_COL = "Point density"


def _get_axis_labels(
    x: ArrayLike | str, y: ArrayLike | str, df: pd.DataFrame | None = None
) -> tuple[str, str]:
    """Axis labels from column names, array names (e.g. pd.Series) or defaults."""
    if df is not None:
        return str(x), str(y)
    xlabel = str(getattr(x, "name", x if isinstance(x, str) else "Actual"))
    ylabel = str(getattr(y, "name", y if isinstance(y, str) else "Predicted"))
    return xlabel, ylabel


def _parity_data(
    x: ArrayLike | str,
    y: ArrayLike | str,
    df: pd.DataFrame | None,
    xlabel: str | None,
    ylabel: str | None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Finite 1D x/y float arrays (NaN rows dropped) and axis labels."""
    xs, ys = (np.asarray(arr, dtype=float) for arr in df_to_arrays(df, x, y))
    if xs.ndim != 1 or ys.shape != xs.shape:
        raise ValueError(
            f"Expected 1D x and y of equal length, got {xs.shape=}, {ys.shape=}"
        )
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not finite.any():
        raise ValueError("Parity plots need at least one finite (x, y) point")
    auto_xlabel, auto_ylabel = _get_axis_labels(x, y, df)
    return xs[finite], ys[finite], xlabel or auto_xlabel, ylabel or auto_ylabel


def _point_density(
    xs: np.ndarray,
    ys: np.ndarray,
    density: Literal["kde", "empirical"],
    n_bins: int | Literal[False],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points to plot (the first sample in each bin when n_bins, else the inputs)
    with their density, sorted by density so the densest points are drawn last.
    Binned KDE densities are scaled to sum to the number of samples.
    """
    if density not in ("kde", "empirical"):
        raise ValueError(f"Unknown {density=}, must be 'kde' or 'empirical'")
    if n_bins:
        df_bins = bin_df_cols(
            pd.DataFrame({"x": xs, "y": ys}),
            bin_by_cols=["x", "y"],
            n_bins=n_bins,
            bin_counts_col=DENSITY_COL,
        )
        plot_x, plot_y = df_bins["x"].to_numpy(), df_bins["y"].to_numpy()
        point_density = df_bins[DENSITY_COL].to_numpy(dtype=float)
    else:
        plot_x, plot_y, point_density = xs, ys, np.ones(len(xs))
    if density == "kde":
        try:  # needs >= 2 points not all on one line (non-singular covariance)
            kde = scipy.stats.gaussian_kde(np.vstack([xs, ys]))
            point_density = kde(np.vstack([plot_x, plot_y]))
            if n_bins:
                point_density = point_density / point_density.sum() * len(xs)
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"KDE impossible for {len(xs)} point(s) ({exc}), using empirical "
                "density",
                stacklevel=3,
            )
    order = np.argsort(point_density, kind="stable")
    return plot_x[order], plot_y[order], point_density[order]


def density_scatter(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    density: Literal["kde", "empirical"] | None = None,
    log_density: bool | None = None,
    n_bins: int | Literal[False] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] | None = None,
    stats: bool | dict[str, Any] = True,
    **widget_options: Any,
) -> ScatterPlotWidget:
    """Parity scatter plot with points colored by density.

    Binning reduces the number of plotted points, enabling millions of data points
    with small payloads.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns.
        density ("kde" | "empirical" | None): Density estimate used for colors.
            Defaults to "empirical" (bin counts) when binning, else "kde" (Gaussian
            kernel density estimate).
        log_density (bool | None): Log-scale the color mapping. None (default)
            enables it when positive densities span more than 2 decades.
        n_bins (int | False | None): Number of bins per axis. None (default) bins
            into 200 bins only for more than 1000 points.
        xlabel (str, optional): x-axis label. Auto-detected from data if None.
        ylabel (str, optional): y-axis label. Auto-detected from data if None.
        identity_line (bool | dict): Add a y = x line. A dict is passed to
            add_identity_line(). Defaults to True.
        best_fit_line (bool | dict | None): Add a least-squares fit line of the
            unbinned data. A dict is passed to add_best_fit_line(). None (default)
            adds it if R² > 0.3.
        stats (bool | dict): Show MAE and R² of the unbinned data. A dict is passed
            to annotate_metrics(). Defaults to True.
        **widget_options: Passed to ScatterPlotWidget, e.g. color_scale, color_bar
            (default title "Point density") or point styles via styles.

    Returns:
        ScatterPlotWidget: The density scatter widget.
    """
    xs, ys, xlabel, ylabel = _parity_data(x, y, df, xlabel, ylabel)
    if n_bins is None:  # auto-enable binning depending on data size
        n_bins = 200 if len(xs) > 1000 else False
    plot_x, plot_y, point_density = _point_density(
        xs, ys, density or ("empirical" if n_bins else "kde"), n_bins
    )
    if log_density is None:
        positive = point_density[point_density > 0]
        log_density = bool(
            positive.size and np.log10(positive.max() / positive.min()) > 2
        )
    color_scale = {
        "scheme": "interpolateViridis",
        "type": "log" if log_density else "linear",
    }
    widget = ScatterPlotWidget(
        series=[
            {
                "x": plot_x,
                "y": plot_y,
                "color_values": point_density,
                "markers": "points",
            }
        ],
        **{
            "x_axis": {"label": xlabel},
            "y_axis": {"label": ylabel},
            "color_scale": color_scale,
            "color_bar": {"title": DENSITY_COL},
        }
        | widget_options,
    )
    return enhance_parity_plot(
        widget,
        xs,
        ys,
        identity_line=identity_line,
        best_fit_line=best_fit_line,
        stats=stats,
    )


def density_hexbin(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    bin_px: float | None = None,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] | None = None,
    stats: bool | dict[str, Any] = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **widget_options: Any,
) -> BinnedScatterPlotWidget:
    """Parity plot of points binned into a screen-space density map. Bins are
    recomputed on zoom and switch to individual points when few remain in view.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns.
        bin_px (float, optional): Bin size in screen pixels. Defaults to
            matterviz' default (2.8 px).
        identity_line (bool | dict): Add a y = x line. A dict is passed to
            add_identity_line(). Defaults to True.
        best_fit_line (bool | dict | None): Add a least-squares fit line. A dict is
            passed to add_best_fit_line(). None (default) adds it if R² > 0.3.
        stats (bool | dict): Show MAE and R². A dict is passed to
            annotate_metrics(). Defaults to True.
        xlabel (str, optional): x-axis label. Auto-detected if None.
        ylabel (str, optional): y-axis label. Auto-detected if None.
        **widget_options: Passed to BinnedScatterPlotWidget, e.g. marginals,
            color_bar or density (merged with bin_px).

    Returns:
        BinnedScatterPlotWidget: The binned density widget.
    """
    xs, ys, xlabel, ylabel = _parity_data(x, y, df, xlabel, ylabel)
    density = {"bin_px": bin_px} if bin_px is not None else {}
    widget_options["density"] = density | widget_options.get("density", {})
    widget = BinnedScatterPlotWidget(
        series=[{"x": xs, "y": ys}],
        **{
            "x_axis": {"label": xlabel},
            "y_axis": {"label": ylabel},
            "color_bar": {"title": "Count"},
        }
        | widget_options,
    )
    return enhance_parity_plot(
        widget,
        xs,
        ys,
        identity_line=identity_line,
        best_fit_line=best_fit_line,
        stats=stats,
    )


def _histogram_marginals(bins: int) -> dict[str, dict[str, Any]]:
    """Top x and right y marginal histograms with the given bin count."""
    if not isinstance(bins, int) or bins < 1:
        raise ValueError(f"{bins=} must be a positive int")
    return {side: {"type": "histogram", "bins": bins} for side in ("top", "right")}


def density_scatter_with_hist(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    bins: int = 100,
    bin_px: float = 1.5,
    **kwargs: Any,
) -> BinnedScatterPlotWidget:
    """Fine-grained density parity plot with marginal x/y histograms.

    MatterViz' ScatterPlot widget doesn't expose marginals, so this renders a
    BinnedScatterPlot with small (default 1.5 px) bins, which looks like a
    density-colored scatter plot and stays fast for millions of points.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns.
        bins (int): Number of bins per marginal histogram. Defaults to 100.
        bin_px (float): Density bin size in screen pixels. Defaults to 1.5.
        **kwargs: Passed to density_hexbin().

    Returns:
        BinnedScatterPlotWidget: Density widget with marginal histograms.
    """
    return density_hexbin(
        x, y, df=df, bin_px=bin_px, marginals=_histogram_marginals(bins), **kwargs
    )


def density_hexbin_with_hist(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    bins: int = 100,
    **kwargs: Any,
) -> BinnedScatterPlotWidget:
    """Binned density parity plot with marginal x/y histograms.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns.
        bins (int): Number of bins per marginal histogram. Defaults to 100.
        **kwargs: Passed to density_hexbin(), e.g. bin_px.

    Returns:
        BinnedScatterPlotWidget: Density widget with marginal histograms.
    """
    return density_hexbin(x, y, df=df, marginals=_histogram_marginals(bins), **kwargs)
