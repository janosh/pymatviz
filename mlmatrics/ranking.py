import matplotlib.pyplot as plt
import numpy as np


def get_err_decay(y_test, y_pred, n_rand=100):
    abs_err = np.abs(y_test - y_pred)
    # increasing count of the number of samples in each element of cumsum()
    n_inc = range(1, len(abs_err) + 1)

    decay_by_err = np.sort(abs_err).cumsum() / n_inc

    # error decay for random exclusion of samples
    ae_tile = np.tile(abs_err, [n_rand, 1])
    [np.random.shuffle(row) for row in ae_tile]  # shuffle rows of ae_tile in place

    rand = ae_tile.cumsum(1) / n_inc
    rand_mean, rand_std = rand.mean(0), rand.std(0)

    return decay_by_err, rand_mean, rand_std


def get_std_decay(y_test, y_pred, y_std):
    abs_err = np.abs(y_test - y_pred)

    # indices that sort y_std in ascending uncertainty
    y_std_sort = np.argsort(y_std)

    # increasing count of the number of samples in each element of cumsum()
    n_inc = range(1, len(abs_err) + 1)

    decay_by_std = abs_err[y_std_sort].cumsum() / n_inc

    return decay_by_std


def err_decay(y_test, y_pred, y_stds, title=None, n_rand=100, percentile=True):
    """
    Plot for assessing the quality of uncertainty estimates. If a model's
    uncertainty is well calibrated, i.e. strongly correlated with its error,
    removing the most uncertain predictions should make the mean error decay
    similarly to how it decays when removing the predictions of largest error
    """
    xs = range(100 if percentile else len(y_test), 0, -1)

    prctile = lambda seq: np.percentile(seq, xs[::-1])

    if type(y_stds) != dict:
        y_stds = {"std": y_stds}

    for key, y_std in y_stds.items():
        decay_by_std = get_std_decay(y_test, y_pred, y_std)

        if percentile:
            decay_by_std = prctile(decay_by_std)

        plt.plot(xs, decay_by_std, label=key)

    decay_by_err, rand_mean, rand_std = get_err_decay(y_test, y_pred, n_rand)

    if percentile:
        decay_by_err, rand_mean, rand_std = [
            prctile(ys) for ys in [decay_by_err, rand_mean, rand_std]
        ]

    rand_hi, rand_lo = rand_mean + rand_std, rand_mean - rand_std
    plt.plot(xs, decay_by_err, label="error")
    plt.plot(xs[::-1] if percentile else xs, rand_mean)
    plt.fill_between(
        xs[::-1] if percentile else xs, rand_hi, rand_lo, alpha=0.2, label="random"
    )
    plt.ylim([0, rand_mean.mean() * 1.1])

    # n: Number of remaining points in err calculation after discarding the
    # (len(y_test) - n) most uncertain/hightest-error points
    plt.xlabel("error/confidence percentile" if percentile else "excluded samples")
    plt.ylabel("$\\epsilon_\\mathrm{MAE}$")
    plt.title(title)
    plt.legend()
