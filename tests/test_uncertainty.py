from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pymatviz.uncertainty import error_decay_with_uncert, qq_gaussian
from tests.conftest import DfOrArrays, xs, y_pred, y_true


if TYPE_CHECKING:
    from numpy.typing import ArrayLike


@pytest.mark.parametrize("y_std", [y_true - y_pred, {"y_std_mock": y_true - y_pred}])
@pytest.mark.parametrize("percentiles", [True, False])
def test_error_decay_with_uncert(
    df_or_arrays: DfOrArrays,
    y_std: ArrayLike | dict[str, ArrayLike],
    percentiles: bool,
) -> None:
    """Test error decay with uncertainty plot."""
    df, x, y = df_or_arrays
    if df is None and isinstance(y_std, str | pd.Index):
        y_std = y_true - y_pred

    fig = error_decay_with_uncert(x, y, y_std, df=df, percentiles=percentiles)

    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == (
        "Confidence percentiles" if percentiles else "Excluded samples"
    )
    assert fig.layout.yaxis.title.text == "MAE"
    assert len(fig.data) >= 2  # At least uncertainty line and error line

    # Check for essential traces
    trace_names = {trace.name for trace in fig.data if trace.name}
    assert "error" in trace_names
    assert "random" in trace_names or "random (mean)" in trace_names


@pytest.mark.parametrize("y_std", [xs, {"foo": xs, "bar": 0.1 * xs}])
def test_qq_gaussian(
    df_or_arrays: DfOrArrays, y_std: ArrayLike | dict[str, ArrayLike]
) -> None:
    """Test Q-Q gaussian plot."""
    df, x, y = df_or_arrays
    if df is None and isinstance(y_std, str | pd.Index):
        y_std = xs

    fig = qq_gaussian(x, y, y_std, df=df)

    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "Theoretical Quantile"
    assert fig.layout.yaxis.title.text == "Observed Quantile"
    assert fig.layout.xaxis.range == (0, 1)
    assert fig.layout.yaxis.range == (0, 1)

    # Check legend positioning and miscalibration measures
    assert fig.layout.legend.x == 0.02
    assert fig.layout.legend.y == 0.98
    assert fig.layout.legend.borderwidth == 0

    # Check miscalibration measures in legend labels
    legend_traces = [
        trace
        for trace in fig.data
        if hasattr(trace, "name") and trace.name and "miscal:" in trace.name
    ]
    assert len(legend_traces) > 0

    # Check Q-Q line properties
    qq_traces = [
        trace
        for trace in fig.data
        if trace.mode == "lines" and trace.name and "miscal:" in trace.name
    ]
    for trace in qq_traces:
        assert trace.line.width == 2
        assert trace.opacity == 0.8

    # Check fill areas
    fill_traces = [trace for trace in fig.data if trace.fill and trace.fill != "none"]
    assert len(fill_traces) > 0


def test_qq_gaussian_multiple_uncertainties() -> None:
    """Test Q-Q plot with multiple uncertainties."""
    n_samples = 100
    np_rng = np.random.default_rng(seed=0)
    y_true = np_rng.normal(0, 1, n_samples)
    y_pred = y_true + np_rng.normal(0, 0.1, n_samples)

    uncertainties = {
        "epistemic": np.abs(np_rng.normal(0.2, 0.05, n_samples)),
        "aleatoric": np.abs(np_rng.normal(0.15, 0.03, n_samples)),
    }

    fig = qq_gaussian(y_true, y_pred, uncertainties)
    trace_names = {trace.name for trace in fig.data if trace.name}

    # Check for both uncertainty types in legend
    assert any("epistemic" in name for name in trace_names)
    assert any("aleatoric" in name for name in trace_names)
    assert len(fig.data) >= 4  # 2 Q-Q lines + 2 fill areas + identity line
