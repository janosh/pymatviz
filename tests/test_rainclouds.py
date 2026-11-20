from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import pymatviz as pmv
from pymatviz.widgets.box_plot import BoxPlotWidget


@pytest.fixture
def sample_data() -> dict[str, np.ndarray]:
    """Three normal samples of different size, location and spread."""
    rng = np.random.default_rng(seed=0)
    return {
        "A": rng.normal(0, 1, 100),
        "B": rng.normal(2, 1.5, 80),
        "C": rng.normal(-1, 0.5, 120),
    }


def test_rainclouds_series_and_bandwidth(sample_data: dict[str, np.ndarray]) -> None:
    """One series per dataset with raw samples and scipy-convention bandwidth."""
    widget = pmv.rainclouds(sample_data, bw=0.3, width_box=0.2)
    assert isinstance(widget, BoxPlotWidget)
    assert [srs["label"] for srs in widget.series] == list(sample_data)
    for srs, values in zip(widget.series, sample_data.values(), strict=True):
        np.testing.assert_array_equal(srs["y"], values)
        # scipy gaussian_kde(bw_method=0.3) kernel std = 0.3 * sample std (ddof=1)
        assert srs["bandwidth"] == pytest.approx(0.3 * values.std(ddof=1), rel=1e-15)
        assert srs["box_width"] == 0.2


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {},
            {
                "orientation": "horizontal",
                "kind": "violin+box",
                "side": "positive",
                "point_side": "negative",
                "points": "strip",
                "show_outliers": False,
                "violin_width": 0.6,
                "violin_style": {"opacity": 0.7},
                "sample_style": {"radius": 2, "opacity": 0.5},
            },
        ),
        ({"orientation": "vertical"}, {"orientation": "vertical"}),
        (  # user style dicts merge over the alpha/point_size defaults
            {
                "alpha": 0.4,
                "violin_style": {"stroke_width": 2},
                "sample_style": {"radius": 5},
            },
            {
                "violin_style": {"opacity": 0.4, "stroke_width": 2},
                "sample_style": {"radius": 5, "opacity": 0.5},
            },
        ),
        ({"show_box": False}, {"kind": "violin"}),
        ({"show_violin": False}, {"kind": "box"}),
        ({"show_points": False}, {"points": "none", "show_outliers": True}),
        (
            {"points": "swarm", "side": "negative"},
            {"points": "swarm", "side": "negative"},
        ),
        (
            {"alpha": 0.4, "point_size": 5, "width_viol": 0.5},
            {"violin_style": {"opacity": 0.4}, "violin_width": 0.5},
        ),
    ],
)
def test_rainclouds_options(
    sample_data: dict[str, np.ndarray], kwargs: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Raincloud options map onto BoxPlotWidget traits; kwargs override defaults."""
    widget = pmv.rainclouds(sample_data, **kwargs)
    for key, val in expected.items():
        assert getattr(widget, key) == val, key
    if "point_size" in kwargs:
        assert widget.sample_style["radius"] == kwargs["point_size"]


def test_rainclouds_with_dataframe() -> None:
    """(DataFrame, column) tuples, Series, lists and number tuples are accepted."""
    df_in = pd.DataFrame({"value": [1.0, 2.0, 4.0], "other": [0, 0, 0]})
    widget = pmv.rainclouds(
        {
            "df": (df_in, "value"),
            "series": df_in["value"] * 2,
            "list": [3, 5],
            "tuple": (6, 7),
        }
    )
    assert [srs["y"] for srs in widget.series] == [
        [1, 2, 4],
        [2, 4, 8],
        [3, 5],
        [6, 7],
    ]


@pytest.mark.parametrize(
    ("data", "kwargs", "match"),
    [
        ({"a": [1, 2]}, {"show_violin": False, "show_box": False}, "show_violin"),
        ({}, {}, "at least one dataset"),
        ({"a": []}, {}, "non-empty"),
        ({"a": [1, float("nan")]}, {}, "finite"),
    ],
)
def test_rainclouds_invalid(
    data: dict[str, Any], kwargs: dict[str, Any], match: str
) -> None:
    """Invalid options and empty datasets raise."""
    with pytest.raises(ValueError, match=match):
        pmv.rainclouds(data, **kwargs)


@pytest.mark.parametrize("values", [[3.0], [2.0, 2.0, 2.0]])
def test_rainclouds_zero_spread_uses_default_bandwidth(values: list[float]) -> None:
    """Zero-variance samples omit bandwidth so MatterViz picks a positive one."""
    (srs,) = pmv.rainclouds({"a": values}).series
    assert "bandwidth" not in srs
