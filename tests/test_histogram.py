from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import pytest

from pymatviz.histogram import elements_hist, histogram
from tests.conftest import df_regr, y_true


if TYPE_CHECKING:
    from collections.abc import Sequence


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
@pytest.mark.parametrize("bins", [20, 100])
@pytest.mark.parametrize("values", [y_true.tolist(), df_regr.y_true.tolist()])
def test_histogram(values: Sequence[float], log_y: bool, bins: int) -> None:
    """Test histogram function with Plotly backend."""
    fig = histogram(values, log_y=log_y, bins=bins)
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

    assert fig.layout.yaxis.title.text == "Count"
