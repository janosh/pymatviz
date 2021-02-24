from mlmatrics import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    residual_hist,
    scatter_with_err_bar,
)

from . import y_pred, y_true


def test_density_scatter():
    density_scatter(y_true, y_pred)


def test_density_scatter_with_hist():
    density_scatter_with_hist(y_true, y_pred)


def test_density_hexbin():
    density_hexbin(y_true, y_pred)


def test_density_hexbin_with_hist():
    density_hexbin_with_hist(y_true, y_pred)


def test_scatter_with_err_bar():
    scatter_with_err_bar(y_true, y_pred, yerr=y_true - y_pred)


def test_residual_hist():
    residual_hist(y_true, y_pred)
