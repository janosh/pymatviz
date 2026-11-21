"""Uncertainty calibration visualizations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pymatviz.powerups import add_identity_line
from pymatviz.utils.data import df_to_arrays
from pymatviz.widgets._normalize import with_defaults
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import ArrayLike


def _load_regression_data(
    y_true: ArrayLike | str,
    y_pred: ArrayLike | str,
    y_std: ArrayLike | Mapping[str, ArrayLike] | str | Sequence[str],
    df: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Resolve y_true/y_pred/y_std into arrays, pulling columns from df when given."""
    row_mask = None
    # a single df column is labeled by its name, a lone array as "std"
    std_name = y_std if isinstance(y_std, str) else "std"
    if isinstance(y_std, str) or (
        df is not None
        and isinstance(y_std, Sequence | pd.Index)
        and all(isinstance(column, str) for column in y_std)
    ):
        true_vals, pred_vals, y_std = df_to_arrays(df, y_true, y_pred, y_std)  # ty: ignore
    else:
        true_vals, pred_vals = df_to_arrays(df, y_true, y_pred)
        if df is not None:
            row_mask = df[[y_true, y_pred]].notna().all(axis=1).to_numpy()

    true_vals, pred_vals = np.asarray(true_vals), np.asarray(pred_vals)
    std_arrays = {
        str(key): np.asarray(std)
        for key, std in (
            y_std if isinstance(y_std, Mapping) else {std_name: y_std}
        ).items()
    }
    if row_mask is not None:
        for key, std in std_arrays.items():
            if std.ndim != 1 or len(std) != len(row_mask):
                raise ValueError(
                    f"Uncertainties for {key!r} must have shape {(len(row_mask),)}, "
                    f"got {std.shape}"
                )
        std_arrays = {key: std[row_mask] for key, std in std_arrays.items()}
    for name, values in [
        ("y_true", true_vals),
        ("y_pred", pred_vals),
        *std_arrays.items(),
    ]:
        if values.ndim != 1 or values.shape != true_vals.shape:
            raise ValueError(
                f"{name} must be 1D with shape {true_vals.shape}, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
    return true_vals, pred_vals, std_arrays


def qq_gaussian(
    y_true: ArrayLike | str,
    y_pred: ArrayLike | str,
    y_std: ArrayLike | Mapping[str, ArrayLike] | str | Sequence[str],
    *,
    df: pd.DataFrame | None = None,
    identity_line: bool | dict[str, Any] = True,
    **kwargs: Any,
) -> ScatterPlotWidget:
    """Q-Q Gaussian plot for uncertainty calibration assessment.

    Plots the observed vs expected proportion of z-scored residuals below each
    standard normal quantile. The area between each curve and the diagonal (the
    miscalibration area) is shaded and shown in the legend.

    Args:
        y_true: Ground truth targets (array or df column name).
        y_pred: Model predictions (array or df column name).
        y_std: Positive standard deviations: one array/column, or a dict (or list of
            df column names) of several uncertainty estimates to compare.
        df: DataFrame containing data columns.
        identity_line: Show the perfect calibration line y = x. A dict is passed
            to add_identity_line(), e.g. {"style": {"color": "blue"}}.
        **kwargs: Passed to ScatterPlotWidget, e.g. x_axis, y_axis, styles.

    Returns:
        ScatterPlotWidget: Q-Q calibration plot.
    """
    y_true, y_pred, y_std = _load_regression_data(y_true, y_pred, y_std, df)
    res = y_pred - y_true
    if len(res) == 0:
        raise ValueError("Q-Q calibration requires non-empty data")
    eps = 1e-10
    exp_proportions = np.linspace(eps, 1 - eps, 100)
    from scipy.stats import norm

    quantiles = norm.ppf(exp_proportions)

    series, fill_regions = [], []
    for series_idx, (key, std) in enumerate(y_std.items()):
        if np.any(std <= 0):
            raise ValueError(f"Uncertainties for {key!r} must be positive")
        z_scored = np.sort(res / std)
        obs_proportions = np.searchsorted(z_scored, quantiles, side="right") / res.size
        miscal_area = np.trapezoid(
            np.abs(obs_proportions - exp_proportions), x=exp_proportions
        )
        series.append(
            {
                "x": exp_proportions,
                "y": obs_proportions,
                "markers": "line",
                "label": f"{key} (miscal: {miscal_area:.2f})",
                "line_style": {"width": 2},
                "metadata": {"miscal_area": float(miscal_area)},
            }
        )
        fill_regions.append(
            {
                "upper": {"type": "series", "series_idx": series_idx},
                "lower": {"type": "data", "values": exp_proportions},
                "fill": "gray",
                "fill_opacity": 0.2,
                "show_in_legend": False,
            }
        )

    widget = ScatterPlotWidget(
        series=series,
        **with_defaults(
            {
                "fill_regions": fill_regions,
                "x_axis": {"label": "Theoretical Quantile"},
                "y_axis": {"label": "Observed Quantile"},
                "x_range": [0, 1],
                "y_range": [0, 1],
            },
            kwargs,
        ),
    )
    if identity_line:
        line_kwargs = identity_line if isinstance(identity_line, dict) else {}
        style = {"color": "red"} | line_kwargs.get("style", {})
        add_identity_line(widget, **line_kwargs | {"style": style})
    return widget


def error_decay_with_uncert(
    y_true: ArrayLike | str,
    y_pred: ArrayLike | str,
    y_std: ArrayLike | Mapping[str, ArrayLike] | str | Sequence[str],
    *,
    df: pd.DataFrame | None = None,
    percentiles: bool = True,
    **kwargs: Any,
) -> ScatterPlotWidget:
    """Error decay plot as the most uncertain samples are excluded.

    Compares the MAE of the retained samples when excluding them in order of
    decreasing uncertainty to the optimal (decreasing error) and random orders. The
    random baseline is the mean absolute error with a band of the exact standard
    deviation of the running MAE over random exclusion orders (sampling without
    replacement), so no shuffling is needed.

    Args:
        y_true: Ground truth targets (array or df column name).
        y_pred: Model predictions (array or df column name).
        y_std: Uncertainties: one array/column, or a dict (or list of df column
            names) of several uncertainty estimates to compare.
        df: DataFrame containing data columns.
        percentiles: Show excluded samples as a percentage rather than a count.
        **kwargs: Passed to ScatterPlotWidget, e.g. x_axis, y_axis, styles.

    Returns:
        ScatterPlotWidget: Error decay plot.
    """
    y_true, y_pred, y_std = _load_regression_data(y_true, y_pred, y_std, df)
    abs_err = np.abs(y_true - y_pred)
    n_samples = len(abs_err)
    if n_samples == 0:
        raise ValueError("error_decay_with_uncert needs non-empty data")
    retained = np.arange(1, n_samples + 1)
    sample_indices = (
        np.linspace(0, n_samples - 1, min(100, n_samples), dtype=int)
        if percentiles
        else np.arange(n_samples)
    )
    xs = (n_samples - retained[sample_indices]).astype(float)
    if percentiles:
        xs *= 100 / n_samples

    if reserved := {"error", "random"} & set(y_std):
        raise ValueError(f"y_std keys {sorted(reserved)} clash with baseline labels")
    decays = {
        key: abs_err[np.argsort(std)].cumsum() / retained for key, std in y_std.items()
    }
    decays["error"] = np.sort(abs_err).cumsum() / retained  # optimal decay
    series: list[dict[str, Any]] = [
        {"x": xs, "y": decay[sample_indices], "markers": "line", "label": key}
        for key, decay in decays.items()
    ]

    # std of the mean of k samples drawn without replacement from n with population
    # variance s^2: s * sqrt((n - k) / (k * (n - 1))), 0 for k = n (finite population
    # correction). This is the n_shuffles -> inf limit of shuffling abs_err.
    finite_pop_corr = (n_samples - retained) / (retained * max(n_samples - 1, 1))
    rand_std = (abs_err.std() * np.sqrt(finite_pop_corr))[sample_indices]
    series.append(
        {
            "x": xs,
            "y": np.full(len(xs), abs_err.mean()),
            "markers": "line",
            "label": "random",
            "line_style": {"dash": "6 4"},
        }
    )
    error_band = {
        "series": {"type": "series", "series_idx": len(series) - 1},
        "error": rand_std,
        "fill": "gray",
        "fill_opacity": 0.2,
    }
    return ScatterPlotWidget(
        series=series,
        **with_defaults(
            {
                "error_bands": [error_band],
                "x_axis": {
                    "label": "Excluded samples (%)"
                    if percentiles
                    else "Excluded samples"
                },
                "y_axis": {"label": "MAE"},
                "y_range": [0, None],
            },
            kwargs,
        ),
    )
