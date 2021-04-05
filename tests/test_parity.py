from ml_matrics import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)

from . import y_pred, y_true


def test_density_scatter():
    density_scatter(y_true, y_pred)


def test_density_scatter_log():
    density_scatter(y_true, y_pred, log=True)


def test_density_scatter_sort():
    density_scatter(y_true, y_pred, sort=True)


def test_density_scatter_color_map():
    density_scatter(y_true, y_pred, color_map="Greens")


def test_density_scatter_with_hist():
    density_scatter_with_hist(y_true, y_pred)


def test_density_hexbin():
    density_hexbin(y_true, y_pred)


def test_density_hexbin_with_hist():
    density_hexbin_with_hist(y_true, y_pred)


def test_scatter_with_yerr_bar():
    scatter_with_err_bar(y_true, y_pred, yerr=y_true - y_pred)


def test_scatter_with_xerr_bar():
    scatter_with_err_bar(y_true, y_pred, xerr=y_true - y_pred)


def test_residual_vs_actual():
    residual_vs_actual(y_true, y_pred)
