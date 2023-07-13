from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt


if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def marchenko_pastur_pdf(x: float, gamma: float, sigma: float = 1) -> float:
    """Generate Marchenko-Pastur probability distribution which describes the density of
    singular values of large rectangular random matrices.

    See https://wikipedia.org/wiki/Marchenko-Pastur_distribution.

    By comparing the eigenvalue distribution of a correlation matrix to this
    PDF, one can gauge the significance of correlations.

    Args:
        x (float): Position at which to compute probability density.
        gamma (float): Also referred to as lambda. The distribution's main parameter
            that measures how well sampled the data is.
        sigma (float, optional): Standard deviation of random variables assumed
            to be independent identically distributed. Defaults to 1 as
            appropriate for correlation matrices.

    Returns:
        float: Marchenko-Pastur density for given gamma at x
    """
    lambda_m = (sigma * (1 - np.sqrt(1 / gamma))) ** 2  # Largest eigenvalue
    lambda_p = (sigma * (1 + np.sqrt(1 / gamma))) ** 2  # Smallest eigenvalue

    prefac = gamma / (2 * np.pi * sigma**2 * x)
    root = np.sqrt((lambda_p - x) * (x - lambda_m))
    unit_step = x > lambda_p or x < lambda_m

    return prefac * root * (0 if unit_step else 1)


def marchenko_pastur(
    matrix: ArrayLike,
    gamma: float,
    sigma: float = 1,
    filter_high_evals: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the eigenvalue distribution of a symmetric matrix (usually a correlation
    matrix) against the Marchenko Pastur distribution.

    The probability of a random matrix having eigenvalues >= (1 + sqrt(gamma))^2 in the
    absence of any signal is vanishingly small. Thus, if eigenvalues larger than that
    appear, they correspond to statistically significant signals.

    Args:
        matrix (ArrayLike): 2d array
        gamma (float): The Marchenko-Pastur ratio of random variables to observation
            count. E.g. for N=1000 variables and p=500 observations of each,
            gamma = p/N = 1/2.
        sigma (float, optional): Standard deviation of random variables. Defaults to 1.
        filter_high_evals (bool, optional): Whether to filter out eigenvalues larger
            than theoretical random maximum. Useful for focusing the plot on the area
            of the MP PDF. Defaults to False.
        ax (Axes, optional): matplotlib Axes on which to plot. Defaults to None.

    Returns:
        ax: The plot's matplotlib Axes.
    """
    ax = ax or plt.gca()

    # use eigvalsh for speed since correlation matrix is symmetric
    evals = np.linalg.eigvalsh(matrix)

    lambda_m = (sigma * (1 - np.sqrt(1 / gamma))) ** 2  # Largest eigenvalue
    lambda_p = (sigma * (1 + np.sqrt(1 / gamma))) ** 2  # Smallest eigenvalue

    if filter_high_evals:
        # Remove eigenvalues larger than those expected in a purely random matrix
        evals = evals[evals <= lambda_p + 1]

    ax.hist(evals, bins=50, edgecolor="black", density=True)

    # Plot the theoretical density
    mp_pdf = np.vectorize(lambda x: marchenko_pastur_pdf(x, gamma, sigma))
    x = np.linspace(max(1e-4, lambda_m), lambda_p, 200)
    ax.plot(x, mp_pdf(x), linewidth=5)

    # Compute and display matrix rank
    # A ratio less than one indicates an undersampled set of RVs
    rank = np.linalg.matrix_rank(matrix)
    n_rows = len(matrix)

    plt.text(
        *[0.95, 0.9],
        f"rank deficiency: {rank}/{n_rows} {'(None)' if n_rows == rank else ''}",
        transform=ax.transAxes,
        ha="right",
    )

    return ax
