from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from pymatviz import cumulative_error, cumulative_residual
from tests.conftest import y_pred, y_true


@pytest.mark.parametrize("alpha", [None, 0.5])
def test_cumulative_error(alpha: float) -> None:
    ax = cumulative_error(y_true - y_pred, alpha=alpha)
    assert isinstance(ax, plt.Axes)


def test_cumulative_residual() -> None:
    ax = cumulative_residual(abs(y_true - y_pred))
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) == 3
    assert ax.get_xlabel() == "Residual"
    assert ax.get_ylabel() == "Percentile"
    assert ax.get_title() == "Cumulative Residual"
    assert ax.get_ylim() == (0, 100)
