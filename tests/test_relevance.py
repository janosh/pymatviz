from matplotlib.axes import Axes

from pymatviz import precision_recall_curve, roc_curve

from .conftest import y_binary, y_proba


def test_roc_curve():
    roc_auc, ax = roc_curve(y_binary, y_proba)
    assert isinstance(roc_auc, float)
    assert isinstance(ax, Axes)


def test_precision_recall_curve():
    precision, ax = precision_recall_curve(y_binary, y_proba)
    assert isinstance(precision, float)
    assert isinstance(ax, Axes)
