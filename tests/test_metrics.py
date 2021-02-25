import numpy as np

from mlmatrics import regression_metrics

from . import y_pred, y_true


def test_regression_metrics():
    metrics = regression_metrics(y_true, y_pred, verbose=False)
    assert metrics["mae"]
    assert metrics["rmse"]
    assert metrics["r2"]


def test_regression_metrics_ensemble():
    # simulate 2-model ensemble by duplicating predictions along 0-axis
    y_preds = np.tile(y_pred, (2, 1))
    metrics = regression_metrics(y_true, y_preds, verbose=False)
    assert metrics["single"]
    assert metrics["ensemble"]
