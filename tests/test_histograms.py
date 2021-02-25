from mlmatrics import residual_hist

from . import y_pred, y_true


def test_residual_hist():
    residual_hist(y_true, y_pred)
