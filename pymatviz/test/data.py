"""Generate dummy data for testing and prototyping."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


SEED = 0


class RegressionData(NamedTuple):
    """Regression data containing: y_true, y_pred and y_std."""

    y_true: NDArray
    y_pred: NDArray
    y_std: NDArray


def get_regression_data(n_samples: int = 500) -> RegressionData:
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
    np_rng = np.random.default_rng(seed=SEED)

    y_true = np_rng.normal(5, 4, n_samples)
    y_pred = 1.2 * y_true - 2 * np_rng.normal(0, 1, n_samples)
    y_std = (y_true - y_pred) * 10 * np_rng.normal(0, 0.1, n_samples)

    return RegressionData(y_true=y_true, y_pred=y_pred, y_std=y_std)
