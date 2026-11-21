"""Parity and density plots."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pymatviz.powerups import enhance_parity_plot
from pymatviz.utils.data import df_to_arrays
from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.binned_scatter_plot import BinnedScatterPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


if TYPE_CHECKING:
    from collections.abc import Sequence
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
        import scipy.stats

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
    log: bool | None = None,
    bins: int | Literal[False] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    identity_line: bool | dict[str, Any] = True,
    best_fit_line: bool | dict[str, Any] | None = None,
    stats: bool | dict[str, Any] = True,
    **kwargs: Any,
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
        log (bool | None): Log-scale the color mapping. None (default)
            enables it when positive densities span more than 2 decades.
        bins (int | False | None): Number of bins per axis. None (default) bins
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
        **kwargs: Passed to ScatterPlotWidget, e.g. color_scale (a scheme name or
            dict merged over the default Viridis scale with the log/linear type),
            color_bar (default title "Point density") or point styles via styles.

    Returns:
        ScatterPlotWidget: The density scatter widget.
    """
    xs, ys, xlabel, ylabel = _parity_data(x, y, df, xlabel, ylabel)
    if bins is None:  # auto-enable binning depending on data size
        bins = 200 if len(xs) > 1000 else False
    plot_x, plot_y, point_density = _point_density(
        xs, ys, density or ("empirical" if bins else "kde"), bins
    )
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
                "color_bar": {"title": DENSITY_COL},
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
        if not isinstance(marginal_bins, int) or marginal_bins < 1:
            raise ValueError(f"{marginal_bins=} must be a positive int")
        marginals = {
            side: {"type": "histogram", "bins": marginal_bins}
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


def bin_df_cols(
    df_in: pd.DataFrame,
    bin_by_cols: Sequence[str],
    *,
    group_by_cols: Sequence[str] = (),
    n_bins: int | Sequence[int] = 100,
    bin_counts_col: str = "bin_counts",
    density_col: str = "",
    verbose: bool = False,
) -> pd.DataFrame:
    """Bin columns of a DataFrame.

    Args:
        df_in (pd.DataFrame): Input dataframe to bin.
        bin_by_cols (Sequence[str]): Columns to bin.
        group_by_cols (Sequence[str]): Additional columns to group by. Defaults to ().
        n_bins (int): Number of bins to use. Defaults to 100.
        bin_counts_col (str): Column name for bin counts. Defaults to "bin_counts".
        density_col (str): Column name for density values. Defaults to "".
        verbose (bool): If True, print df length reduction. Defaults to False.

    Returns:
        pd.DataFrame: Binned DataFrame with original index name and values.
    """
    if isinstance(n_bins, int):
        # broadcast integer n_bins to all bin_by_cols
        n_bins = [n_bins] * len(bin_by_cols)

    if len(bin_by_cols) != len(n_bins):
        raise ValueError(f"{len(bin_by_cols)=} != {len(n_bins)=}")

    # Preserve the original index
    orig_index_name = df_in.index.name or "index"
    # Reset index so it participates in groupby. If the index name is already in the
    # columns, we it'll participate already and be set back to the index at the end.
    if orig_index_name not in df_in:
        df_in = df_in.reset_index()

    # Bins are separate group keys rather than added columns, so df_in needn't be
    # copied to stay unmodified
    bin_keys = [
        pd.Series(pd.cut(df_in[col].values, bins=bins), index=df_in.index).rename(
            f"{col}_bins"
        )
        for col, bins in zip(bin_by_cols, n_bins, strict=True)
    ]
    group = df_in.groupby(by=[*bin_keys, *group_by_cols], observed=True)

    df_bin = group.first().dropna()
    df_bin[bin_counts_col] = group.size()
    df_bin = df_bin.reset_index()

    if verbose:
        print(  # noqa: T201
            f"{1 - len(df_bin) / len(df_in):.1%} sample reduction from binning: from "
            f"{len(df_in):,} to {len(df_bin):,}"
        )

    if density_col:
        # compute kernel density estimate for each bin
        values = df_in[bin_by_cols].dropna().T
        import scipy.stats

        kde = scipy.stats.gaussian_kde(values.astype(float))

        xy_binned = df_bin[bin_by_cols].T
        density = kde(xy_binned.astype(float))
        # scale density to sum to the number of data points (values is transposed,
        # so its columns are the data points)
        df_bin[density_col] = density / density.sum() * values.shape[1]

    # Set the index back to the original index name
    return df_bin.set_index(orig_index_name)
