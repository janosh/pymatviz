"""Parity and density plots."""

from __future__ import annotations

import warnings
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from pymatviz.powerups import enhance_parity_plot
from pymatviz.utils import df_to_arrays
from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.binned_scatter_plot import BinnedScatterPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


if TYPE_CHECKING:
    from typing import Any

    import pandas as pd
    from numpy.typing import ArrayLike


def _parity_data(
    x: ArrayLike | str,
    y: ArrayLike | str,
    df: pd.DataFrame | None,
    xlabel: str | None,
    ylabel: str | None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Finite 1D x/y float arrays (NaN rows dropped) and axis labels (given ones,
    else column names, array names like a pd.Series' or Actual/Predicted).
    """
    xs, ys = (np.asarray(arr, dtype=float) for arr in df_to_arrays(df, x, y))
    if xs.ndim != 1 or ys.shape != xs.shape:
        raise ValueError(
            f"Expected 1D x and y of equal length, got {xs.shape=}, {ys.shape=}"
        )
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not finite.any():
        raise ValueError("Parity plots need at least one finite (x, y) point")

    def auto_label(arr: ArrayLike | str, default: str) -> str:
        """Column name, array name (unnamed pd.Series: default) or default."""
        if df is not None or isinstance(arr, str):
            return str(arr)
        name = getattr(arr, "name", None)
        return default if name is None else str(name)

    xlabel = xlabel or auto_label(x, "Actual")
    ylabel = ylabel or auto_label(y, "Predicted")
    return xs[finite], ys[finite], xlabel, ylabel


# Gaussian KDE costs O(n^2): beyond this many points, density_hexbin bins in the
# browser instead
MAX_KDE_POINTS = 10_000


def density_scatter(
    x: ArrayLike | str,
    y: ArrayLike | str,
    *,
    df: pd.DataFrame | None = None,
    log: bool | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] | None = None,
    stats: bool | dict[str, Any] = True,
    **kwargs: Any,
) -> ScatterPlotWidget:
    """Parity scatter plot with points colored by their Gaussian kernel density
    estimate, densest drawn last. For more than 10k points, use density_hexbin,
    which bins in the browser.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns.
        log (bool | None): Log-scale the color mapping. None (default)
            enables it when positive densities span more than 2 decades.
        xlabel (str, optional): x-axis label. Auto-detected from data if None.
        ylabel (str, optional): y-axis label. Auto-detected from data if None.
        identity_line (bool | dict): Add a y = x line. A dict is passed to
            add_identity_line(). Defaults to True.
        best_fit_line (bool | dict | None): Add a least-squares fit line. A dict is
            passed to add_best_fit_line(). None (default) adds it if R² > 0.3.
        stats (bool | dict): Show MAE and R². A dict is passed to
            annotate_metrics(). Defaults to True.
        **kwargs: Passed to ScatterPlotWidget, e.g. color_scale (a scheme name or
            dict merged over the default Viridis scale with the log/linear type),
            color_bar (default title "Point density") or point styles via styles.

    Returns:
        ScatterPlotWidget: The density scatter widget.

    Raises:
        ValueError: For more than MAX_KDE_POINTS points.
    """
    xs, ys, xlabel, ylabel = _parity_data(x, y, df, xlabel, ylabel)
    if len(xs) > MAX_KDE_POINTS:
        raise ValueError(
            f"density_scatter estimates densities for up to {MAX_KDE_POINTS:,} "
            f"points, got {len(xs):,}: use density_hexbin, which bins in the browser"
        )
    import scipy.stats

    try:  # no KDE for fewer than 2 points or all on one line
        density = scipy.stats.gaussian_kde(np.vstack([xs, ys]))(np.vstack([xs, ys]))
    except (ValueError, np.linalg.LinAlgError) as exc:
        warnings.warn(
            f"KDE impossible for {len(xs)} point(s) ({exc}), using uniform colors",
            stacklevel=2,
        )
        density = np.ones(len(xs))
    order = np.argsort(density, kind="stable")  # densest points drawn last
    plot_x, plot_y, point_density = xs[order], ys[order], density[order]
    if log is None:
        positive = point_density[point_density > 0]
        log = bool(positive.size and np.log10(positive.max() / positive.min()) > 2)
    color_scale = {
        "scheme": "interpolateViridis",
        "type": "log" if log else "linear",
    }
    if isinstance(kwargs.get("color_scale"), str):  # bare scheme keeps the log type
        kwargs["color_scale"] = {"scheme": kwargs["color_scale"]}
    widget = ScatterPlotWidget(
        series=[
            {
                "x": plot_x,
                "y": plot_y,
                "color_values": point_density,
                "markers": "points",
            }
        ],
        **with_defaults(
            {
                "x_axis": {"label": xlabel},
                "y_axis": {"label": ylabel},
                "color_scale": color_scale,
                "color_bar": {"title": "Point density"},
            },
            kwargs,
        ),
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
    marginal_bins: int | None = None,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] | None = None,
    stats: bool | dict[str, Any] = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **kwargs: Any,
) -> BinnedScatterPlotWidget:
    """Parity plot of points binned into a screen-space density map. Bins are
    recomputed on zoom and switch to individual points when few remain in view.

    Args:
        x (array | str): x-values or dataframe column name.
        y (array | str): y-values or dataframe column name.
        df (pd.DataFrame, optional): DataFrame with x and y columns.
        bin_px (float, optional): Bin size in screen pixels. Defaults to
            matterviz' default (2.8 px). Small bins (e.g. 1.5) look like a
            density-colored scatter plot that stays fast for millions of points.
        marginal_bins (int, optional): Add top x and right y marginal histograms
            with this many bins. Defaults to None (no marginals).
        identity_line (bool | dict): Add a y = x line. A dict is passed to
            add_identity_line(). Defaults to True.
        best_fit_line (bool | dict | None): Add a least-squares fit line. A dict is
            passed to add_best_fit_line(). None (default) adds it if R² > 0.3.
        stats (bool | dict): Show MAE and R². A dict is passed to
            annotate_metrics(). Defaults to True.
        xlabel (str, optional): x-axis label. Auto-detected if None.
        ylabel (str, optional): y-axis label. Auto-detected if None.
        **kwargs: Passed to BinnedScatterPlotWidget, e.g. marginals (merged over
            the marginal_bins histograms per side), color_bar or density (merged
            with bin_px).

    Returns:
        BinnedScatterPlotWidget: The binned density widget.

    Raises:
        ValueError: If marginal_bins is not a positive int.
    """
    xs, ys, xlabel, ylabel = _parity_data(x, y, df, xlabel, ylabel)
    marginals = {}
    if marginal_bins is not None:
        is_bool = isinstance(marginal_bins, bool)
        if is_bool or not isinstance(marginal_bins, Integral) or marginal_bins < 1:
            raise ValueError(f"{marginal_bins=} must be a positive int")
        marginals = {
            side: {"type": "histogram", "bins": int(marginal_bins)}
            for side in ("top", "right")
        }
    widget = BinnedScatterPlotWidget(
        series=[{"x": xs, "y": ys}],
        **with_defaults(
            {
                "x_axis": {"label": xlabel},
                "y_axis": {"label": ylabel},
                "color_bar": {"title": "Count"},
                "density": {"bin_px": bin_px} if bin_px is not None else {},
                **({"marginals": marginals} if marginals else {}),
            },
            kwargs,
        ),
    )
    return enhance_parity_plot(
        widget,
        xs,
        ys,
        identity_line=identity_line,
        best_fit_line=best_fit_line,
        stats=stats,
    )
