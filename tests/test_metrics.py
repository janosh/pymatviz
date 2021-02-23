from mlmatrics import regression_metrics

from . import y_pred, y_true


def test_regression_metrics():
    regression_metrics(y_pred, y_true, verbose=False)
