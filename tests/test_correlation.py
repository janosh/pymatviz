import numpy as np

from mlmatrics import marchenko_pastur


def test_marchenko_pastur():

    r_rows, n_cols = 50, 100
    rand_mat = np.random.normal(0, 1, size=(r_rows, n_cols))
    corr_mat = np.corrcoef(rand_mat)

    marchenko_pastur(corr_mat, gamma=n_cols / r_rows)


def test_marchenko_pastur_filter_high_evals():

    r_rows, n_cols = 50, 100
    rand_mat = np.random.normal(0, 1, size=(r_rows, n_cols))
    corr_mat = np.corrcoef(rand_mat)

    marchenko_pastur(corr_mat, gamma=n_cols / r_rows, filter_high_evals=True)
