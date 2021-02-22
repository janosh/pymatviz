from mlmatrics import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
)

from . import xs, y_pred


def test_density_scatter():
    density_scatter(xs, y_pred)


def test_density_scatter_with_hist():
    density_scatter_with_hist(xs, y_pred)


def test_density_hexbin():
    density_hexbin(xs, y_pred)


def test_density_hexbin_with_hist():
    density_hexbin_with_hist(xs, y_pred)
