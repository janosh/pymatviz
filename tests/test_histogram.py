from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.graph_objects as go
import pytest

from pymatviz.histogram import elements_hist, histogram
from tests.conftest import df_regr, y_true


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


@pytest.mark.parametrize(
    ("kwargs", "expected_x", "expected_y", "expected_text", "expected_yaxis"),
    [
        (
            {},
            ["O", "Fe", "Li", "P"],
            [7, 3, 1, 1],
            ("58%", "25%", "8%", "8%"),
            "linear",
        ),
        (
            {"log_y": True},
            ["O", "Fe", "Li", "P"],
            [7, 3, 1, 1],
            ("58%", "25%", "8%", "8%"),
            "log",
        ),
        ({"keep_top": 2}, ["O", "Fe"], [7, 3], ("70%", "30%"), "linear"),
        (
            {"show_values": "count"},
            ["O", "Fe", "Li", "P"],
            [7, 3, 1, 1],
            ("7", "3", "1", "1"),
            "linear",
        ),
        ({"show_values": None}, ["O", "Fe", "Li", "P"], [7, 3, 1, 1], None, "linear"),
    ],
)
def test_hist_elemental_prevalence(
    kwargs: dict[str, Any],
    expected_x: list[str],
    expected_y: list[int],
    expected_text: tuple[str, ...] | None,
    expected_yaxis: str,
) -> None:
    """Test elements histogram counts, labels, and y-axis options."""
    fig = elements_hist(["Fe2O3", "LiFePO4"], **kwargs)
    trace = fig.data[0]

    assert list(trace.x) == expected_x
    assert list(trace.y) == expected_y
    assert trace.text == expected_text
    assert fig.layout.yaxis.type == expected_yaxis


def test_hist_elemental_prevalence_style_options() -> None:
    """Test elements histogram forwards bar style options."""
    fig = elements_hist(["Fe2O3", "LiFePO4"], bar_width=0.5, opacity=0.9)
    trace = fig.data[0]

    assert trace.width == 0.5
    assert trace.opacity == 0.9


@pytest.mark.parametrize("log_y", [True, False])
@pytest.mark.parametrize("bins", [20, 100])
@pytest.mark.parametrize("values", [y_true, df_regr.y_true])
def test_histogram(
    values: Sequence[float] | np.ndarray | pd.Series, log_y: bool, bins: int
) -> None:
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


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({}, "must not be empty"),
        ([], "non-empty 1D"),
        ([[1, 2], [3, 4]], "non-empty 1D"),
    ],
)
def test_histogram_rejects_invalid_values(values: Any, match: str) -> None:
    """Histogram rejects empty and non-1D inputs."""
    with pytest.raises(ValueError, match=match):
        histogram(values)


def test_histogram_preserves_zero_x_range_boundary() -> None:
    """Histogram treats zero x_range bounds as explicit values."""
    fig = histogram([1, 2, 3], x_range=(0, None), bins=3)
    assert fig.data[0].x[0] == 0


@pytest.mark.parametrize(
    ("series_name", "expected_title"),
    [(None, "Value"), ("my_col", "my_col"), (0, "0"), ("", "")],
)
def test_histogram_series_xaxis_title(
    series_name: str | int | None, expected_title: str
) -> None:
    """Histogram uses series name as x-axis title; only None falls back to 'Value'."""
    fig = histogram(pd.Series([1.0, 2.0, 3.0], name=series_name))
    assert fig.layout.xaxis.title.text == expected_title
    assert fig.data[0].name == expected_title
