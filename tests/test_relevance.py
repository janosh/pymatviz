from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from pymatviz import precision_recall_curve, roc_curve

from .conftest import y_binary, y_proba


@pytest.mark.parametrize("ax", [None, plt.gca()])
def test_roc_curve(ax: plt.Axes) -> None:
    roc_auc, ax = roc_curve(y_binary, y_proba, ax=ax)
    assert isinstance(roc_auc, float)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "ROC Curve"
    assert ax.get_xlabel() == "False Positive Rate"
    assert ax.get_ylabel() == "True Positive Rate"


@pytest.mark.parametrize("ax", [None, plt.gca()])
def test_precision_recall_curve(ax: plt.Axes) -> None:
    precision, ax = precision_recall_curve(y_binary, y_proba, ax=ax)
    assert isinstance(precision, float)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Precision-Recall Curve"
    assert ax.get_xlabel() == "Recall"
    assert ax.get_ylabel() == "Precision"
