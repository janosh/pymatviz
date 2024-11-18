"""Dummy regression data for testing and prototyping."""

import numpy as np


n_samples = 500
np_rng = np.random.default_rng(seed=0)

y_true = np_rng.normal(5, 4, n_samples)
y_pred = 1.2 * y_true - 2 * np_rng.normal(0, 1, n_samples)
y_std = (y_true - y_pred) * 10 * np_rng.normal(0, 0.1, n_samples)
