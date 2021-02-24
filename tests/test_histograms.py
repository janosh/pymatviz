from mlmatrics import residual_vs_actual

from . import y_pred, y_true


def test_residual_vs_actual():
    residual_vs_actual(y_true, y_pred)
