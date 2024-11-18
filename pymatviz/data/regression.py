"""Dummy regression data for testing and prototyping."""

import numpy as np


_rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)

y_true = np_rng.normal(5, 4, _rand_regression_size)
y_pred = 1.2 * y_true - 2 * np_rng.normal(0, 1, _rand_regression_size)
y_std = (y_true - y_pred) * 10 * np_rng.normal(0, 0.1, _rand_regression_size)
