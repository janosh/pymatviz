from mlmatrics import err_decay

from . import y_pred, y_true


def test_err_decay():
    err_decay(y_true, y_pred, y_true - y_pred)


def test_err_decay_with_n_rand_10():
    err_decay(y_true, y_pred, y_true - y_pred, n_rand=10)


def test_err_decay_with_percentile_false():
    err_decay(y_true, y_pred, y_true - y_pred, percentiles=False)
