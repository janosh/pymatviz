from mlmatrics import residual_hist, true_pred_hist

from . import y_pred, y_true


def test_residual_hist():
    residual_hist(y_true, y_pred)


def test_true_pred_hist():
    true_pred_hist(y_true, y_pred, y_true - y_pred)
