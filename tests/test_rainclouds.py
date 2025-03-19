from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import pymatviz as pmv


if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture
def sample_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed=0)
    return {
        "A": rng.normal(0, 1, 100),
        "B": rng.normal(2, 1.5, 80),
        "C": rng.normal(-1, 0.5, 120),
    }


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(seed=0)
    return pd.DataFrame(
        {
            "value": rng.normal(0, 1, 100),
            "category": rng.choice(["X", "Y", "Z"], 100),
            "extra": rng.uniform(0, 1, 100),
        }
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"orientation": "v"},
        {"alpha": 0.5, "width_viol": 0.4, "width_box": 0.2},
        {"jitter": 0.02, "point_size": 5, "bw": 0.3, "cut": 0.1},
        {"scale": "count", "rain_offset": -0.2, "offset": 0.1},
        {"hover_data": ["extra"]},
    ],
)
def test_rainclouds_basic(
    sample_data: dict[str, np.ndarray], kwargs: dict[str, Any]
) -> None:
    fig = pmv.rainclouds(sample_data, **kwargs)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(sample_data) * 3


@pytest.mark.parametrize(
    ("data_input", "expected_traces"),
    [
        ({"A": [1, 2, 3], "B": [4, 5, 6]}, 6),  # 2 groups, 3 traces each
        ({"X": [1], "Y": [2], "Z": [3]}, 9),  # 3 groups, 3 traces each
        ({"A": [1, 2, 3, 10]}, 3),  # 1 group, 3 traces
    ],
)
def test_rainclouds_data_shapes(
    data_input: dict[str, Sequence[float]], expected_traces: int
) -> None:
    if any(len(vals) < 2 for vals in data_input.values()):
        with pytest.raises(
            ValueError, match="`dataset` input should have multiple elements"
        ):
            pmv.rainclouds(data_input)  # type: ignore[arg-type]
    else:
        fig = pmv.rainclouds(data_input)  # type: ignore[arg-type]
        assert len(fig.data) == expected_traces


def test_rainclouds_with_dataframe(sample_dataframe: pd.DataFrame) -> None:
    data = {
        "X": (sample_dataframe, "value"),
        "Y": sample_dataframe[sample_dataframe["category"] == "Y"]["value"],
    }
    fig = pmv.rainclouds(data, hover_data=["category", "extra"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 6


@pytest.mark.parametrize(
    ("orientation", "expected_axis"), [("h", "yaxis"), ("v", "xaxis")]
)
def test_rainclouds_orientation(
    sample_data: dict[str, np.ndarray],
    orientation: Literal["h", "v"],
    expected_axis: str,
) -> None:
    fig = pmv.rainclouds(sample_data, orientation=orientation)
    assert getattr(fig.layout, expected_axis).ticktext == tuple(sample_data)


@pytest.mark.parametrize(
    ("scale", "expected_range"),
    [
        ("area", (0, 1)),
        ("count", None),
        ("width", (0, 1)),
    ],
)
def test_rainclouds_scale(
    sample_data: dict[str, np.ndarray],
    scale: Literal["area", "count", "width"],
    expected_range: tuple[float, float] | None,
) -> None:
    fig = pmv.rainclouds(sample_data, scale=scale)
    violin_trace = next(
        trace
        for trace in fig.data
        if trace.type == "scatter" and trace.fill == "toself"
    )
    y_range = np.array(violin_trace.y)
    if expected_range is not None:
        assert np.min(y_range) >= expected_range[0]
        assert np.max(y_range) <= expected_range[1]


@pytest.mark.parametrize(
    ("hover_data", "expected_columns"),
    [
        (None, ["value"]),
        (["category"], ["value", "category"]),
        ({"Group": ["extra"]}, ["value", "extra"]),
    ],
)
def test_rainclouds_hover_data(
    sample_dataframe: pd.DataFrame,
    hover_data: Sequence[str] | dict[str, Sequence[str]] | None,
    expected_columns: Sequence[str],
) -> None:
    data = {"Group": (sample_dataframe, "value")}
    fig = pmv.rainclouds(data, hover_data=hover_data)  # type: ignore[arg-type]
    scatter_trace = next(
        trace for trace in fig.data if getattr(trace, "mode", None) == "markers"
    )
    assert all(
        all(col in text for col in expected_columns) for text in scatter_trace.hovertext
    )


@pytest.mark.parametrize(
    ("data_input", "expected_msg"),
    [
        ({"A": "not a sequence"}, "`dataset` input should have multiple elements"),
        (
            {},
            "max() iterable argument is empty"
            if sys.version_info >= (3, 12)
            else "max() arg is an empty sequence",
        ),
    ],
)
def test_rainclouds_invalid_input(
    data_input: dict[str, Any], expected_msg: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        pmv.rainclouds(data_input)


def test_rainclouds_long_labels(sample_data: dict[str, np.ndarray]) -> None:
    long_labels = {
        f"Very long label {idx}": data
        for idx, (_, data) in enumerate(sample_data.items())
    }
    fig = pmv.rainclouds(long_labels)
    assert fig.layout.yaxis.tickangle == -90


@pytest.mark.parametrize(
    ("show_violin", "show_box", "show_points", "n_expected_traces"),
    [
        (True, True, True, 3),
        (False, True, True, 2),
        (True, False, True, 2),
        (True, True, False, 2),
        (False, False, True, 1),
        (False, True, False, 1),
        (True, False, False, 1),
        (False, False, False, 0),
    ],
)
def test_rainclouds_trace_visibility(
    sample_data: dict[str, np.ndarray],
    show_violin: bool,
    show_box: bool,
    show_points: bool,
    n_expected_traces: int,
) -> None:
    fig = pmv.rainclouds(
        sample_data,
        show_violin=show_violin,
        show_box=show_box,
        show_points=show_points,
    )
    assert len(fig.data) == len(sample_data) * n_expected_traces
