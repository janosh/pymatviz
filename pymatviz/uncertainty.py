"""Uncertainty calibration visualizations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from pymatviz.process_data import df_to_arrays


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
            y_std if isinstance(y_std, Mapping) else {"std": y_std}
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
    fig: go.Figure | None = None,
    identity_line: bool | dict[str, Any] = True,
) -> go.Figure:
    """Q-Q Gaussian plot for uncertainty calibration assessment.

    Args:
        y_true: Ground truth targets
        y_pred: Model predictions
        y_std: Positive standard deviations (single array or dict for multiple)
        df: DataFrame containing data columns
        fig: Existing plotly figure to add to
        identity_line: Show perfect calibration line

    Returns:
        go.Figure: plotly Figure with Q-Q plot
    """
    y_true, y_pred, y_std = _load_regression_data(y_true, y_pred, y_std, df)

    fig = fig or go.Figure()
    res = y_pred - y_true
    if len(res) == 0:
        raise ValueError("Q-Q calibration requires non-empty data")
    eps = 1e-10
    exp_proportions = np.linspace(eps, 1 - eps, 100)
    quantiles = norm.ppf(exp_proportions)

    if identity_line:
        line_props = (
            identity_line.get("line_kwargs", {})
            if isinstance(identity_line, dict)
            else {}
        )
        fig.add_scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(color=line_props.get("color", "red"), width=1, dash="dash"),
            showlegend=False,
        )

    for key, std in y_std.items():
        if np.any(std <= 0):
            raise ValueError(f"Uncertainties for {key!r} must be positive")
        z_scored = np.sort(res / std)
        obs_proportions = np.searchsorted(z_scored, quantiles, side="right") / res.size
        miscal_area = np.trapezoid(
            np.abs(obs_proportions - exp_proportions), x=exp_proportions
        )

        # Invisible reference line for fill
        fig.add_scatter(
            x=exp_proportions,
            y=exp_proportions,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )

        # Q-Q line with fill
        fig.add_scatter(
            x=exp_proportions,
            y=obs_proportions,
            mode="lines",
            name=f"{key} (miscal: {miscal_area:.2f})",
            line=dict(width=2),
            opacity=0.8,
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.2)",
        )

    fig.update_layout(
        xaxis=dict(title="Theoretical Quantile", range=[0, 1]),
        yaxis=dict(title="Observed Quantile", range=[0, 1]),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=0,
        ),
    )
    return fig


def error_decay_with_uncert(
    y_true: ArrayLike | str,
    y_pred: ArrayLike | str,
    y_std: ArrayLike | Mapping[str, ArrayLike] | str | Sequence[str],
    *,
    df: pd.DataFrame | None = None,
    n_rand: int = 100,
    percentiles: bool = True,
    fig: go.Figure | None = None,
) -> go.Figure:
    """Error decay plot as uncertain samples are excluded.

    Args:
        y_true: Ground truth targets
        y_pred: Model predictions
        y_std: Uncertainties (single array or dict for multiple)
        df: DataFrame containing data columns
        n_rand: Random shuffles for baseline
        percentiles: Show excluded samples as a percentage rather than a count
        fig: Existing plotly figure to add to

    Returns:
        Plotly figure with error decay plot
    """
    y_true, y_pred, y_std = _load_regression_data(y_true, y_pred, y_std, df)

    fig = fig or go.Figure()
    abs_err = np.abs(y_true - y_pred)
    n_samples = len(abs_err)
    if n_samples == 0 or n_rand < 1:
        raise ValueError(
            f"Expected non-empty data and n_rand >= 1, got {n_samples=}, {n_rand=}"
        )
    retained = np.arange(1, n_samples + 1)
    sample_indices = (
        np.linspace(0, n_samples - 1, min(100, n_samples), dtype=int)
        if percentiles
        else np.arange(n_samples)
    )
    xs = (n_samples - retained[sample_indices]).astype(float)
    if percentiles:
        xs *= 100 / n_samples

    # Add uncertainty-based decay lines
    for key, std in y_std.items():
        decay = abs_err[np.argsort(std)].cumsum() / retained
        fig.add_scatter(x=xs, y=decay[sample_indices], mode="lines", name=key)

    # Optimal error-based decay
    decay_optimal = np.sort(abs_err).cumsum() / retained
    fig.add_scatter(x=xs, y=decay_optimal[sample_indices], mode="lines", name="error")

    # Add random baseline with confidence interval
    rand_mean = abs_err.mean()
    abs_err_tile = np.tile(abs_err, [n_rand, 1])
    rng = np.random.default_rng(seed=0)
    for row in abs_err_tile:
        rng.shuffle(row)
    rand_std = (abs_err_tile.cumsum(1).std(0) / retained)[sample_indices]

    # Random mean line
    fig.add_scatter(
        x=[xs[-1], xs[0]],
        y=[rand_mean, rand_mean],
        mode="lines",
        name="random (mean)",
        line=dict(dash="dash"),
        showlegend=False,
    )

    # Random confidence interval
    fig.add_scatter(
        x=xs,
        y=rand_mean + rand_std,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=xs,
        y=rand_mean - rand_std,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(128,128,128,0.2)",
        line=dict(width=0),
        name="random",
        hoverinfo="skip",
    )

    fig.layout.xaxis = dict(
        title="Excluded samples (%)" if percentiles else "Excluded samples"
    )
    fig.layout.yaxis = dict(title="MAE", rangemode="tozero")

    return fig
