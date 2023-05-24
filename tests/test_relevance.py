from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest

from pymatviz import precision_recall_curve, roc_curve
from tests.conftest import df_x_y_clf


if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike


@pytest.mark.parametrize("ax", [None, plt.gca()])
@pytest.mark.parametrize("df, y_binary, y_proba", df_x_y_clf)
def test_roc_curve(
    df: pd.DataFrame | None,
    y_binary: str | ArrayLike,
    y_proba: str | ArrayLike,
    ax: plt.Axes,
) -> None:
    roc_auc, ax = roc_curve(y_binary, y_proba, df=df, ax=ax)
    assert isinstance(roc_auc, float)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "ROC Curve"
    assert ax.get_xlabel() == "False Positive Rate"
    assert ax.get_ylabel() == "True Positive Rate"


@pytest.mark.parametrize("ax", [None, plt.gca()])
@pytest.mark.parametrize("df, y_binary, y_proba", df_x_y_clf)
def test_precision_recall_curve(
    df: pd.DataFrame | None,
    y_binary: str | ArrayLike,
    y_proba: str | ArrayLike,
    ax: plt.Axes,
) -> None:
    precision, ax = precision_recall_curve(y_binary, y_proba, df=df, ax=ax)
    assert isinstance(precision, float)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Precision-Recall Curve"
    assert ax.get_xlabel() == "Recall"
    assert ax.get_ylabel() == "Precision"
