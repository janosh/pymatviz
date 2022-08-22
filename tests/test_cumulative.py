from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from pymatviz import cum_err, cum_res

from .conftest import y_pred, y_true


@pytest.mark.parametrize("alpha", [None, 0.5])
def test_cum_err(alpha: float) -> None:
    ax = cum_err(y_pred, y_true, alpha=alpha)
    assert isinstance(ax, plt.Axes)


def test_cum_res():
    ax = cum_res(y_pred, y_true)
    assert isinstance(ax, plt.Axes)
