from matplotlib.pyplot import Axes

from ml_matrics import err_decay

from ._helpers import y_pred, y_true


y_std_mock = y_true - y_pred


def test_err_decay():
    ax = err_decay(y_true, y_pred, y_std_mock)

    assert isinstance(ax, Axes)

    err_decay(y_true, y_pred, {"y_std_mock": y_std_mock})

    err_decay(y_true, y_pred, y_std_mock, n_rand=10)

    err_decay(y_true, y_pred, y_std_mock, percentiles=False)
