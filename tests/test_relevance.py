from mlmatrics import precision_recall_curve, roc_curve

from . import y_binary, y_proba


def test_roc_curve():
    roc_curve(y_binary, y_proba)


def test_precision_recall_curve():
    precision_recall_curve(y_binary, y_proba)
