"""Generate dummy data for testing and prototyping."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class RegressionData(NamedTuple):
    """Regression data containing: y_true, y_pred and y_std."""

    y_true: NDArray[np.float64]
    y_pred: NDArray[np.float64]
    y_std: NDArray[np.float64]


def regression(
    n_samples: int = 500,
    true_mean: float = 5,
    true_std: float = 4,
    pred_slope: float = 1.2,
    pred_intercept: float = -2,
    seed: int = 0,
) -> RegressionData:
    """Generate dummy regression data for testing and prototyping.

    This function creates synthetic data to simulate a regression task:
    - `y_true`: Sampled from a normal distribution with mean 5 and
        standard deviation 4.
    - `y_pred`: Linearly related to `y_true` with a slope of 1.2 and
        additional Gaussian noise.
    - `y_std`: Residuals scaled by random noise, representing variability
        in predictions.

    Parameters:
        n_samples (int): Number of samples to generate. Default is 500.

    Returns:
        RegressionData: A named tuple containing y_true, y_pred, and y_std.
    """
    np_rng = np.random.default_rng(seed=seed)

    y_true = np_rng.normal(loc=true_mean, scale=true_std, size=n_samples)

    noise = np_rng.normal(loc=0, scale=1, size=n_samples)
    y_pred = pred_slope * y_true + pred_intercept + noise

    # Generate realistic positive uncertainties
    # Base uncertainty that's correlated with absolute residuals but always positive
    abs_residuals = np.abs(y_true - y_pred)
    y_std = (
        0.5 + 0.3 * abs_residuals + 0.8 * np_rng.exponential(scale=1, size=n_samples)
    )

    return RegressionData(y_true=y_true, y_pred=y_pred, y_std=y_std)
