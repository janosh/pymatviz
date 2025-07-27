"""Uncertainty calibration visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from pymatviz.process_data import df_to_arrays


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike


def qq_gaussian(
    y_true: ArrayLike | str,
    y_pred: ArrayLike | str,
    y_std: ArrayLike | dict[str, ArrayLike] | str | Sequence[str],
    *,
    df: pd.DataFrame | None = None,
    fig: go.Figure | None = None,
    identity_line: bool | dict[str, Any] = True,
) -> go.Figure:
    """Q-Q Gaussian plot for uncertainty calibration assessment.

    Args:
        y_true: Ground truth targets
        y_pred: Model predictions
        y_std: Uncertainties (single array or dict for multiple)
        df: DataFrame containing data columns
        fig: Existing plotly figure to add to
        identity_line: Show perfect calibration line

    Returns:
        go.Figure: plotly Figure with Q-Q plot
    """
    if isinstance(y_std, str | pd.Index):
        arrays = df_to_arrays(df, y_true, y_pred, y_std)
        y_true, y_pred, y_std = arrays[0], arrays[1], arrays[2]
    else:
        arrays = df_to_arrays(df, y_true, y_pred)
        y_true, y_pred = arrays[0], arrays[1]

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)  # Type narrowing

    fig = fig or go.Figure()
    if not isinstance(y_std, dict):
        y_std = {"std": y_std}

    # Calculate Q-Q data
    res = y_pred - y_true  # Signed residuals
    eps = 1e-10
    exp_proportions = np.linspace(eps, 1 - eps, 100)

    if identity_line:
        line_props = (
            identity_line.get("line_kwargs", {})
            if isinstance(identity_line, dict)
            else {}
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                line=dict(color=line_props.get("color", "red"), width=1, dash="dash"),
                showlegend=False,
            )
        )

    for key, std in y_std.items():
        z_scored = (res / std).reshape(-1, 1)
        obs_proportions = np.mean(z_scored <= norm.ppf(exp_proportions), axis=0)
        miscal_area = np.trapezoid(np.abs(obs_proportions - exp_proportions), dx=0.01)

        # Invisible reference line for fill
        fig.add_trace(
            go.Scatter(
                x=exp_proportions,
                y=exp_proportions,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Q-Q line with fill
        fig.add_trace(
            go.Scatter(
                x=exp_proportions,
                y=obs_proportions,
                mode="lines",
                name=f"{key} (miscal: {miscal_area:.2f})",
                line=dict(width=2),
                opacity=0.8,
                fill="tonexty",
                fillcolor="rgba(128,128,128,0.2)",
            )
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
    y_std: ArrayLike | dict[str, ArrayLike] | str | Sequence[str],
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
        percentiles: Use percentiles vs sample count on x-axis
        fig: Existing plotly figure to add to

    Returns:
        Plotly figure with error decay plot
    """
    if isinstance(y_std, str | pd.Index):
        arrays = df_to_arrays(df, y_true, y_pred, y_std)
        y_true, y_pred, y_std = arrays[0], arrays[1], arrays[2]
    else:
        arrays = df_to_arrays(df, y_true, y_pred)
        y_true, y_pred = arrays[0], arrays[1]

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)  # Type narrowing

    fig = fig or go.Figure()
    if not isinstance(y_std, dict):
        y_std = {"std": y_std}

    # Calculate error decay curves
    abs_err = np.abs(y_true - y_pred)
    n_samples = len(abs_err)
    xs = list(range(100 if percentiles else n_samples, 0, -1))

    # Add uncertainty-based decay lines
    for key, std in y_std.items():
        # Sort by uncertainty and calculate cumulative error
        decay = abs_err[np.argsort(std)].cumsum() / np.arange(1, n_samples + 1)
        if percentiles:
            decay = np.percentile(decay, xs[::-1])
        fig.add_scatter(x=xs, y=decay, mode="lines", name=key)

    # Optimal error-based decay
    decay_optimal = np.sort(abs_err).cumsum() / np.arange(1, n_samples + 1)
    if percentiles:
        decay_optimal = np.percentile(decay_optimal, xs[::-1])
    fig.add_scatter(x=xs, y=decay_optimal, mode="lines", name="error")

    # Add random baseline with confidence interval
    rand_mean = abs_err.mean()
    # Calculate random std dev
    abs_err_tile = np.tile(abs_err, [n_rand, 1])
    rng = np.random.default_rng(seed=0)
    for row in abs_err_tile:
        rng.shuffle(row)
    rand_std = abs_err_tile.cumsum(1).std(0) / np.arange(1, n_samples + 1)

    if percentiles:
        rand_std = np.percentile(rand_std, xs[::-1])

    x_range = [1, 100] if percentiles else [n_samples, 0]
    x_fill = xs[::-1] if percentiles else xs

    # Random mean line
    fig.add_scatter(
        x=x_range,
        y=[rand_mean, rand_mean],
        mode="lines",
        name="random (mean)",
        line=dict(dash="dash"),
        showlegend=False,
    )

    # Random confidence interval
    fig.add_scatter(
        x=x_fill,
        y=rand_mean + rand_std,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=x_fill,
        y=rand_mean - rand_std,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(128,128,128,0.2)",
        line=dict(width=0),
        name="random",
        hoverinfo="skip",
    )

    fig.layout.xaxis = dict(
        title="Confidence percentiles" if percentiles else "Excluded samples"
    )
    fig.layout.yaxis = dict(title="MAE", range=[0, rand_mean * 1.3])

    return fig
