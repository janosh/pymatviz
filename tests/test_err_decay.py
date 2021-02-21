from mlmatrics import err_decay

from . import y_pred, y_true


def test_err_decay():
    err_decay(y_true, y_pred, y_true - y_pred)
