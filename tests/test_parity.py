from pymatviz import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)

from .conftest import y_pred, y_true


def test_density_scatter():
    density_scatter(y_true, y_pred)
    density_scatter(y_true, y_pred, log=True)
    density_scatter(y_true, y_pred, sort=True)
    density_scatter(y_true, y_pred, color_map="Greens")


def test_density_scatter_with_hist():
    density_scatter_with_hist(y_true, y_pred)


def test_density_hexbin():
    density_hexbin(y_true, y_pred)


def test_density_hexbin_with_hist():
    density_hexbin_with_hist(y_true, y_pred)


def test_scatter_with_err_bar():
    scatter_with_err_bar(y_true, y_pred, yerr=y_true - y_pred)
    scatter_with_err_bar(y_true, y_pred, xerr=y_true - y_pred)


def test_residual_vs_actual():
    residual_vs_actual(y_true, y_pred)
