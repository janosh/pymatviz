"""Tests for colorbar tick formatting in ptable plot functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

import pymatviz as pmv


if TYPE_CHECKING:
    import plotly.graph_objects as go


def colorbar_markers(fig: go.Figure) -> list[go.scatter.Marker]:
    """Markers of the (invisible) traces that render a colorbar."""
    return [
        trace.marker
        for trace in fig.data
        if getattr(getattr(trace, "marker", None), "showscale", False)
    ]


def test_ptable_heatmap_colorbar_formatting() -> None:
    """Test that ptable_heatmap uses SI suffixes for colorbar ticks."""
    test_data = {"O": 10_000, "C": 100_000, "H": 1_000_000, "N": 10_000_000}

    colorbar = pmv.ptable_heatmap(test_data).data[-1].colorbar
    assert colorbar.tickformat == "~s"
    assert colorbar.title.text is None

    # log scale colorbar gets SI-formatted tick labels
    colorbar = pmv.ptable_heatmap(test_data, log=True).data[-1].colorbar
    expected = ("10k", "20k", "50k", "100k", "200k", "500k", "1M", "2M", "5M", "10M")
    assert colorbar.ticktext == expected


@pytest.mark.parametrize(
    ("colorbar", "expected_tickformat"),
    [(None, "~s"), ({"tickformat": ".1~s"}, ".1~s")],
)
def test_ptable_heatmap_splits_colorbar_formatting(
    colorbar: dict[str, str] | None,
    expected_tickformat: str,
) -> None:
    """ptable_heatmap_splits should use SI suffixes for colorbar ticks."""
    test_data = pd.DataFrame({"Dataset 1": {"Fe": 1_000, "O": 10_000, "C": 100_000}})
    test_data["Dataset 2"] = test_data["Dataset 1"] * 2

    fig = pmv.ptable_heatmap_splits(test_data, colorbar=colorbar)

    markers = colorbar_markers(fig)
    assert [marker.colorbar.title.text for marker in markers] == [
        "Dataset 1",
        "Dataset 2",
    ]
    assert {marker.colorbar.tickformat for marker in markers} == {expected_tickformat}


@pytest.mark.parametrize("log", [False, True])
def test_ptable_hists_colorbar_formatting(log: bool) -> None:
    """ptable_hists colorbar uses SI suffixes, log=True also SI-formats x ticks."""
    values = np.logspace(2, 6, 100)  # 100 to 1,000,000
    test_data = {"Fe": values, "O": values, "C": values}

    fig = pmv.ptable_hists(test_data, log=log, colorbar={"title": "Test"})

    (marker,) = colorbar_markers(fig)
    assert marker.colorbar.tickformat == "~s"
    assert marker.colorbar.title.text == "Test"
    x_tick_formats = {axis.tickformat for axis in fig.select_xaxes()}
    assert x_tick_formats == {".4s" if log else ".2g"}


def test_ptable_scatter_colorbar_si_formatting() -> None:
    """Test that ptable_scatter uses SI suffixes for colorbar ticks."""
    xy_color = ([1, 2, 3], [4, 5, 6], [100, 1_000, 10_000])
    test_data = {"Fe": xy_color, "O": xy_color, "C": xy_color}

    fig = pmv.ptable_scatter(test_data, colorbar={"tickformat": ".1~s"})

    (marker,) = colorbar_markers(fig)
    assert marker.colorbar.tickformat == ".1~s"
