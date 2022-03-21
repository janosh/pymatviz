from pymatviz import cum_err, cum_res

from .conftest import y_pred, y_true


def test_cum_err():
    cum_err(y_pred, y_true)


def test_cum_res():
    cum_res(y_pred, y_true)
