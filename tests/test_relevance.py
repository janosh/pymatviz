from matplotlib.axes import Axes

from ml_matrics import precision_recall_curve, roc_curve

from . import y_binary, y_proba


def test_roc_curve():
    roc_auc, ax = roc_curve(y_binary, y_proba)
    assert type(roc_auc) == float
    assert isinstance(ax, Axes)


def test_precision_recall_curve():
    precision, ax = precision_recall_curve(y_binary, y_proba)
    assert type(precision) == float
    assert isinstance(ax, Axes)
