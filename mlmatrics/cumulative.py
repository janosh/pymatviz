import matplotlib.pyplot as plt
import numpy as np


def add_dropdown(ax, percentile, err):
    per = int(percentile * (err.shape[0] - 1) / 100 + 0.5)
    ax.plot((0, err[per]), (percentile, percentile), "--", color="grey", alpha=0.4)
    ax.plot((err[per], err[per]), (0, percentile), "--", color="grey", alpha=0.4)


def cum_res(res, ax=None):
    if ax is None:
        ax = plt.gca()

    n_data = len(res)

    low = int(0.05 * (n_data - 1) + 0.5)
    up = int(0.95 * (n_data - 1) + 0.5)

    d_low = res[low] - res[int(0.97 * low)]
    d_up = res[int(1.03 * up)] - res[up]

    d_max = max(d_low, d_up)

    lim = max(abs(res[up] + d_max), abs(res[low] - d_max))

    ax.plot(res, np.arange(n_data) / n_data * 100)
    ax.fill_between(res[low:up], (np.arange(n_data) / n_data * 100)[low:up], alpha=0.3)

    ax.plot((0, 0), (0, 100), "--", color="grey", alpha=0.4)
    ax.set_ylim((0, 100))
    ax.set_xlim((-lim, lim))
    ax.plot((ax.get_xlim()[0], 0), (50, 50), "--", color="grey", alpha=0.4)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Percentile")
    ax.set_title("Cumulative Residual")


def cum_err(err, ax=None, legend_title=None):
    if ax is None:
        ax = plt.gca()

    n_data = len(err)
    lim = np.percentile(err, 98)

    ax.plot(err, np.arange(n_data) / n_data * 100)

    ax.set_ylim((0, 100))
    ax.set_xlim((0, lim))

    add_dropdown(ax, 50, err)
    add_dropdown(ax, 75, err)

    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Percentile")
    ax.set_title("Cumulative Error")
    ax.legend(title=legend_title, frameon=False)


def cum_err_cum_res(targets, preds, title=None):

    res = np.sort(preds - targets)
    err = np.sort(np.abs(preds - targets))

    fig, [ax_res, ax_err] = plt.subplots(1, 2, figsize=(12, 5))

    cum_res(res, ax_res)
    cum_err(err, ax_err, title)

    return fig, [ax_res, ax_err]
