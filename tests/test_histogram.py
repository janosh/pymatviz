from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest

from pymatviz.histogram import elements_hist, histogram
from pymatviz.typing import BACKENDS, MATPLOTLIB
from tests.conftest import df_regr, y_true


if TYPE_CHECKING:
    import numpy as np

    from pymatviz.typing import Backend


def test_hist_elemental_prevalence(glass_formulas: list[str]) -> None:
    """Test elements histogram with various parameters."""
    fig = elements_hist(glass_formulas)
    assert isinstance(fig, go.Figure)

    fig = elements_hist(glass_formulas, log_y=True)
    assert isinstance(fig, go.Figure)

    fig = elements_hist(glass_formulas, keep_top=10)
    assert isinstance(fig, go.Figure)

    fig = elements_hist(glass_formulas, keep_top=10, show_values="count")
    assert isinstance(fig, go.Figure)

    fig = elements_hist(glass_formulas, show_values=None)
    assert isinstance(fig, go.Figure)

    # Test with custom parameters
    fig = elements_hist(glass_formulas, bar_width=0.5, opacity=0.9)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("log_y", [True, False])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("bins", [20, 100])
@pytest.mark.parametrize("values", [y_true, df_regr.y_true])
def test_histogram(
    values: np.ndarray | pd.Series, log_y: bool, backend: Backend, bins: int
) -> None:
    fig = histogram(values, backend=backend, log_y=log_y, bins=bins)
    if backend == MATPLOTLIB:
        assert isinstance(fig, plt.Figure)
        y_min, y_max = fig.axes[0].get_ylim()
        y_min_exp, y_max_exp = {
            (True, 20): (1.82861565, 13.1246825),
            (True, 100): (0.93303299, 4.28709385),
            (False, 20): (0, 12.6),
            (False, 100): (0, 4.2),
        }[log_y, bins]
        assert y_min == pytest.approx(y_min_exp)
        assert y_max == pytest.approx(y_max_exp)

        if isinstance(values, pd.Series):
            assert fig.axes[0].get_xlabel() == values.name
        assert fig.axes[0].get_ylabel() == "Count"
    else:
        assert isinstance(fig, go.Figure)
        dev_fig = fig.full_figure_for_development(warn=False)
        y_min, y_max = dev_fig.layout.yaxis.range
        y_min_exp, y_max_exp = {
            (True, 20): (0.257799370, 1.12241187),
            (True, 100): (-0.03344777, 0.63550776),
            (False, 20): (0, 12.6315789),
            (False, 100): (0, 4.21052631),
        }[log_y, bins]
        assert y_min == pytest.approx(y_min_exp)
        assert y_max == pytest.approx(y_max_exp)

        if isinstance(values, pd.Series):
            assert fig.layout.xaxis.title.text == values.name
        assert fig.layout.yaxis.title.text == "Count"
