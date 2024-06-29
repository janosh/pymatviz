"""Visualizations for assessing the quality of model uncertainty estimates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from pymatviz.powerups import add_identity_line
from pymatviz.utils import df_to_arrays


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike


def qq_gaussian(
    y_true: ArrayLike | str,
    y_pred: ArrayLike | str,
    y_std: ArrayLike | dict[str, ArrayLike] | str | Sequence[str],
    *,
    df: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    identity_line: bool | dict[str, Any] = True,
) -> plt.Axes:
    """Plot the Gaussian quantile-quantile (Q-Q) plot of one (passed as array) or
    multiple (passed as dict) sets of uncertainty estimates for a single pair of ground
    truth targets `y_true` and model predictions `y_pred`.

    Overconfidence relative to a Gaussian distribution is visualized as shaded
    areas below the parity line, underconfidence (oversized uncertainties) as
    shaded areas above the parity line.

    The measure of calibration is how well the uncertainty percentiles conform
    to those of a normal distribution.

    Inspired by https://git.io/JufOz.
    Info on Q-Q plots: https://wikipedia.org/wiki/Q-Q_plot

    Args:
        y_true (array | str): Ground truth targets
        y_pred (array | str): Model predictions
        y_std (array | dict[str, array] | str | list[str]): Model uncertainties either
            as array(s) (single or dict with labels if you have multiple sources of
            uncertainty) or column names in df.
        df (pd.DataFrame, optional): DataFrame with y_true, y_pred and y_std columns.
        ax (Axes): matplotlib Axes on which to plot. Defaults to None.
        identity_line (bool | dict[str, Any], optional): Whether to add a parity line
            (y = x). Defaults to True. Pass a dict to customize line properties.

    Returns:
        plt.Axes: matplotlib Axes object
    """
    if isinstance(y_std, (str, pd.Index)):
        y_true, y_pred, y_std = df_to_arrays(df, y_true, y_pred, y_std)
    else:
        y_true, y_pred = df_to_arrays(df, y_true, y_pred)
    assert isinstance(y_true, np.ndarray)  # noqa: S101
    assert isinstance(y_pred, np.ndarray)  # noqa: S101
    ax = ax or plt.gca()

    if not isinstance(y_std, dict):
        y_std = {"std": y_std}

    res = np.abs(y_pred - y_true)
    resolution = 100

    lines = []  # collect plotted lines to show second legend with miscalibration areas
    for key, std in y_std.items():
        z_scored = (np.array(res) / std).reshape(-1, 1)

        exp_proportions = np.linspace(0, 1, resolution)
        gaussian_upper_bound = norm.ppf(0.5 + exp_proportions / 2)
        obs_proportions = np.mean(z_scored <= gaussian_upper_bound, axis=0)

        [line] = ax.plot(
            exp_proportions, obs_proportions, linewidth=2, alpha=0.8, label=key
        )
        ax.fill_between(
            exp_proportions, y1=obs_proportions, y2=exp_proportions, alpha=0.2
        )
        miscal_area = np.trapz(  # noqa: NPY201
            np.abs(obs_proportions - exp_proportions), dx=1 / resolution
        )
        lines.append([line, miscal_area])

    if identity_line:  # guiding line for perfect calibration
        add_identity_line(
            ax, **(identity_line if isinstance(identity_line, dict) else {})
        )

    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set(xlabel="Theoretical Quantile", ylabel="Observed Quantile")

    legend1 = ax.legend(loc="upper left", frameon=False)
    # Multiple legends on the same axes:
    # https://matplotlib.org/3.3.3/tutorials/intermediate/legend_guide.html#multiple-legends-on-the-same-axes
    ax.add_artist(legend1)

    lines, areas = zip(*lines)  # type: ignore[assignment]

    if len(lines) > 1:
        legend2 = ax.legend(
            lines,
            [f"{area:.2f}" for area in areas],
            title="Miscalibration areas",
            loc="lower right",
            ncol=2,
            frameon=False,
        )
        # https://stackoverflow.com/a/44620643
        legend2._legend_box.align = "left"  # noqa: SLF001
    else:
        ax.legend(
            lines,
            [f"Miscalibration area: {areas[0]:.2f}"],
            loc="lower right",
            frameon=False,
        )

    return ax


def get_err_decay(
    y_true: ArrayLike, y_pred: ArrayLike, n_rand: int = 100
) -> tuple[ArrayLike, ArrayLike]:
    """Calculate the model's error curve as samples are excluded from the calculation
    based on their absolute error.

    Use in combination with get_std_decay to see what the error drop curve would look
    like if model error and uncertainty were perfectly rank-correlated.

    Args:
        y_true (array): ground truth targets
        y_pred (array): model predictions
        n_rand (int, optional): Number of randomly ordered sample exclusions over which
            to average to estimate dummy performance. Defaults to 100.

    Returns:
        Tuple[array, array]: Drop off in errors as data points are dropped based on
            model uncertainties and randomly, respectively.
    """
    abs_err = np.abs(y_true - y_pred)
    # increasing count of the number of samples in each element of cumsum()
    n_inc = range(1, len(abs_err) + 1)

    decay_by_err = np.sort(abs_err).cumsum() / n_inc

    # error decay for random exclusion of samples
    abs_err_tile = np.tile(abs_err, [n_rand, 1])

    for row in abs_err_tile:
        np.random.default_rng(seed=0).shuffle(row)  # shuffle rows of ae_tile in place

    rand = abs_err_tile.cumsum(1) / n_inc

    return decay_by_err, rand.std(0)


def get_std_decay(y_true: ArrayLike, y_pred: ArrayLike, y_std: ArrayLike) -> ArrayLike:
    """Calculate the drop in model error as samples are excluded from the calculation
    based on the model's uncertainty.

    For model's able to estimate their own uncertainty well, meaning predictions of
    larger error are associated with larger uncertainty, the error curve should fall
    off sharply at first as the highest-error points are discarded and slowly towards
    the end where only small-error samples with little uncertainty remain.

    Note that even perfect model uncertainties would not mean this error drop curve
    coincides exactly with the one returned by get_err_decay as in some cases the model
    may have made an accurate prediction purely by chance in which case the error is
    small yet a good uncertainty estimate would still be large, leading the same sample
    to be excluded at different x-axis locations and thus the get_std_decay curve to lie
    higher.

    Args:
        y_true (array): ground truth targets
        y_pred (array): model predictions
        y_std (array): model's predicted uncertainties

    Returns:
        array: Error decay as data points are excluded by order of largest to smallest
            model uncertainties.
    """
    abs_err = np.abs(y_true - y_pred)

    # indices that sort y_std in ascending uncertainty
    y_std_sort = np.argsort(y_std)

    # increasing count of the number of samples in each element of cumsum()
    n_inc = range(1, len(abs_err) + 1)

    return abs_err[y_std_sort].cumsum() / n_inc


def error_decay_with_uncert(
    y_true: ArrayLike | str,
    y_pred: ArrayLike | str,
    y_std: ArrayLike | dict[str, ArrayLike] | str | Sequence[str],
    *,
    df: pd.DataFrame | None = None,
    n_rand: int = 100,
    percentiles: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot for assessing the quality of uncertainty estimates. If a model's uncertainty
    is well calibrated, i.e. strongly correlated with its error, removing the most
    uncertain predictions should make the mean error decay similarly to how it decays
    when removing the predictions of largest error.

    Args:
        y_true (array | str): Ground truth regression targets.
        y_pred (array | str): Model predictions.
        y_std (array | dict[str, ArrayLike] | str | list[str]): Model uncertainties.
            Can be single or multiple uncertainties (e.g. aleatoric/epistemic/total
            uncertainty) as dict.
        n_rand (int, optional): Number of shuffles from which to compute std.dev.
            of error decay by random ordering. Defaults to 100.
        df (pd.DataFrame, optional): DataFrame with y_true, y_pred and y_std columns.
        percentiles (bool, optional): Whether the x-axis shows percentiles or number
            of remaining samples in the MAE calculation. Defaults to True.
        ax (Axes): matplotlib Axes on which to plot. Defaults to None.

    Note: If you're not happy with the default y_max of 1.1 * rand_mean, where rand_mean
    is mean of random sample exclusion, use ax.set(ylim=[None, some_value *
    ax.get_ylim()[1]]).

    Returns:
        plt.Axes: matplotlib Axes object with plotted model error drop curve based on
            excluding data points by order of large to small model uncertainties.
    """
    if isinstance(y_std, (str, pd.Index)):
        y_true, y_pred, y_std = df_to_arrays(df, y_true, y_pred, y_std)
    else:
        y_true, y_pred = df_to_arrays(df, y_true, y_pred)
    assert isinstance(y_true, np.ndarray)  # for mypy  # noqa: S101
    assert isinstance(y_pred, np.ndarray)  # noqa: S101

    ax = ax or plt.gca()

    xs = range(100 if percentiles else len(y_true), 0, -1)

    if not isinstance(y_std, dict):
        y_std = {"std": y_std}

    for key in y_std:
        decay_by_std = get_std_decay(y_true, y_pred, y_std[key])

        if percentiles:
            decay_by_std = np.percentile(decay_by_std, xs[::-1])

        ax.plot(xs, decay_by_std, label=key)

    decay_by_err, rand_std = get_err_decay(y_true, y_pred, n_rand)

    if percentiles:
        decay_by_err, rand_std = (
            np.percentile(ys, xs[::-1]) for ys in [decay_by_err, rand_std]
        )

    rand_mean = np.abs(y_true - y_pred).mean()
    rand_hi, rand_lo = rand_mean + rand_std, rand_mean - rand_std

    ax.plot(xs, decay_by_err, label="error")
    ax.plot([1, 100] if percentiles else [len(xs), 0], [rand_mean, rand_mean])
    ax.fill_between(
        xs[::-1] if percentiles else xs, rand_hi, rand_lo, alpha=0.2, label="random"
    )
    ax.set(ylim=[0, rand_mean.mean() * 1.3], ylabel="MAE")

    # n: Number of remaining points in error calculation after discarding the
    # (len(y_true) - n) most uncertain/hightest-error points
    ax.set(xlabel="Confidence percentiles" if percentiles else "Excluded samples")
    ax.legend(loc="lower left")

    return ax
