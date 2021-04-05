import numpy as np

from ml_matrics import classification_metrics, regression_metrics

from . import y_binary, y_pred, y_proba, y_true


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


def test_classification_metrics():
    y_probs = np.expand_dims(y_proba, axis=(0, 2))
    metrics = classification_metrics(y_binary, y_probs, verbose=False)
    assert metrics["acc"]
    assert metrics["roc_auc"]
    assert metrics["precision"]
    assert metrics["recall"]
    assert metrics["f1"]


def test_classification_metrics_ensemble():
    y_probs = np.expand_dims(y_proba, axis=(0, 2))
    y_probs = np.tile(y_probs, (2, 1, 1))
    metrics = classification_metrics(y_binary, y_probs, verbose=False)
    assert metrics["single"]
    assert metrics["ensemble"]
