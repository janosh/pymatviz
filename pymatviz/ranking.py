from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes

from pymatviz.utils import NumArray


def get_err_decay(
    y_true: NumArray, y_pred: NumArray, n_rand: int = 100
) -> tuple[NumArray, NumArray]:
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
    ae_tile = np.tile(abs_err, [n_rand, 1])

    for row in ae_tile:
        np.random.shuffle(row)  # shuffle rows of ae_tile in place

    rand = ae_tile.cumsum(1) / n_inc

    return decay_by_err, rand.std(0)


def get_std_decay(y_true: NumArray, y_pred: NumArray, y_std: NumArray) -> NumArray:
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

    decay_by_std = abs_err[y_std_sort].cumsum() / n_inc

    return decay_by_std


def err_decay(
    y_true: NumArray,
    y_pred: NumArray,
    y_stds: NumArray | dict[str, NumArray],
    n_rand: int = 100,
    percentiles: bool = True,
    ax: Axes = None,
) -> Axes:
    """Plot for assessing the quality of uncertainty estimates. If a model's
    uncertainty is well calibrated, i.e. strongly correlated with its error,
    removing the most uncertain predictions should make the mean error decay
    similarly to how it decays when removing the predictions of largest error.

    Args:
        y_true (array): Ground truth regression targets.
        y_pred (array): Model predictions.
        y_stds (array | dict[str, NumArray]): Model uncertainties. Can be a single or
            multiple types (e.g. aleatoric/epistemic/total uncertainty) in dict form.
        n_rand (int, optional): Number of shuffles from which to compute std.dev.
            of error decay by random ordering. Defaults to 100.
        percentiles (bool, optional): Whether the x-axis shows percentiless or number
            of remaining samples in the MAE calculation. Defaults to True.
        ax (Axes): matplotlib Axes on which to plot. Defaults to None.

    Returns:
        ax: matplotlib Axes object with plotted model error drop curve based on
            excluding data points by order of large to small model uncertainties.
    """
    if ax is None:
        ax = plt.gca()

    xs = range(100 if percentiles else len(y_true), 0, -1)

    if isinstance(y_stds, np.ndarray):
        y_stds = {"std": y_stds}

    for key, y_std in y_stds.items():
        decay_by_std = get_std_decay(y_true, y_pred, y_std)

        if percentiles:
            decay_by_std = np.percentile(decay_by_std, xs[::-1])

        plt.plot(xs, decay_by_std, label=key)

    decay_by_err, rand_std = get_err_decay(y_true, y_pred, n_rand)

    rand_mean = np.abs(y_true - y_pred).mean()

    if percentiles:
        decay_by_err, rand_std = (
            np.percentile(ys, xs[::-1]) for ys in [decay_by_err, rand_std]
        )

    rand_hi, rand_lo = rand_mean + rand_std, rand_mean - rand_std
    ax.plot(xs, decay_by_err, label="error")
    ax.plot([1, 100] if percentiles else [len(xs), 0], [rand_mean, rand_mean])
    ax.fill_between(
        xs[::-1] if percentiles else xs, rand_hi, rand_lo, alpha=0.2, label="random"
    )
    ax.set(ylim=[0, rand_mean.mean() * 1.1], ylabel="MAE")

    # n: Number of remaining points in err calculation after discarding the
    # (len(y_true) - n) most uncertain/hightest-error points
    ax.set(xlabel="Confidence percentiles" if percentiles else "Excluded samples")
    ax.legend(loc="lower left")

    return ax
