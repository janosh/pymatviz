from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from ml_matrics.utils import NumArray


def get_err_decay(
    y_true: NumArray, y_pred: NumArray, n_rand: int = 100
) -> Tuple[NumArray, NumArray]:
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
    y_stds: Union[NumArray, Dict[str, NumArray]],
    title: str = None,
    n_rand: int = 100,
    percentiles: bool = True,
) -> None:
    """Plot for assessing the quality of uncertainty estimates. If a model's
    uncertainty is well calibrated, i.e. strongly correlated with its error,
    removing the most uncertain predictions should make the mean error decay
    similarly to how it decays when removing the predictions of largest error.

    Args:
        y_true (NumArray): Ground truth regression targets.
        y_pred (NumArray): Model predictions.
        y_stds (NumArray | dict[str, NumArray]): Model uncertainties. Can be a single or
            multiple types (e.g. aleatoric/epistemic/total uncertainty) in dict form.
        title (str, optional): Plot title. Defaults to None.
        n_rand (int, optional): Number of shuffles from which to compute std.dev.
            of error decay by random ordering. Defaults to 100.
        percentiles (bool, optional): Whether the x-axis shows percentiless or number
            of remaining samples in the MAE calculation. Defaults to True.
    """
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
    plt.plot(xs, decay_by_err, label="error")
    plt.plot([1, 100] if percentiles else [len(xs), 0], [rand_mean, rand_mean])
    plt.fill_between(
        xs[::-1] if percentiles else xs, rand_hi, rand_lo, alpha=0.2, label="random"
    )
    plt.ylim([0, rand_mean.mean() * 1.1])

    # n: Number of remaining points in err calculation after discarding the
    # (len(y_true) - n) most uncertain/hightest-error points
    plt.xlabel("Confidence percentiles" if percentiles else "Excluded samples")
    plt.ylabel("MAE")
    plt.title(title)
    plt.legend(loc="lower left")
