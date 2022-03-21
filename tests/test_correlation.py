from __future__ import annotations

import numpy as np

from pymatviz import marchenko_pastur


def test_marchenko_pastur():

    n_rows, n_cols = 50, 100
    rand_mat = np.random.normal(0, 1, size=(n_rows, n_cols))
    corr_mat = np.corrcoef(rand_mat)

    marchenko_pastur(corr_mat, gamma=n_cols / n_rows)


def test_marchenko_pastur_filter_high_evals():

    n_rows, n_cols = 50, 100
    rand_mat = np.random.normal(0, 1, size=(n_rows, n_cols))
    corr_mat = np.corrcoef(rand_mat)

    marchenko_pastur(corr_mat, gamma=n_cols / n_rows, filter_high_evals=True)
