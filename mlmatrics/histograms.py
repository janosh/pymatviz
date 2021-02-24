import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray as Array
from scipy.stats import gaussian_kde


def residual_hist(
    y_true: Array, y_pred: Array, ax: Axes = None, xlabel: str = None
) -> Axes:

    if ax is None:
        ax = plt.gca()

    y_res = y_pred - y_true
    plt.hist(y_res, bins=35, density=True, edgecolor="black")

    # Gaussian kernel density estimation: evaluates the Gaussian
    # probability density estimated based on the points in y_res
    kde = gaussian_kde(y_res)
    x_range = np.linspace(min(y_res), max(y_res), 100)

    label = "Gaussian kernel density estimate"
    plt.plot(x_range, kde(x_range), lw=3, color="red", label=label)

    plt.xlabel(xlabel or r"Residual ($y_\mathrm{test} - y_\mathrm{pred}$)")
    plt.legend(loc=2, framealpha=0.5, handlelength=1)

    return ax
