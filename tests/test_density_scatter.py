from mlmatrics import (
    density_scatter,
    density_scatter_hex,
    density_scatter_hex_with_hist,
    density_scatter_with_hist,
)

from . import xs, y_pred


def test_density_scatter():
    density_scatter(xs, y_pred)


def test_density_scatter_with_hist():
    density_scatter_with_hist(xs, y_pred)


def test_density_scatter_hex():
    density_scatter_hex(xs, y_pred)


def test_density_scatter_hex_with_hist():
    density_scatter_hex_with_hist(xs, y_pred)
