"""Tests for colorbar tick formatting in ptable plot functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import pymatviz as pmv


if TYPE_CHECKING:
    from plotly.graph_objects import Figure


def get_main_colorbar_trace(fig: Figure) -> go.Heatmap:
    """Get the main colorbar trace from a figure.

    This finds the primary heatmap trace with a colorbar.
    """
    return next(
        trace
        for trace in fig.data
        if hasattr(trace, "colorbar")
        and hasattr(trace.colorbar, "tickformat")
        and (not hasattr(trace, "showlegend") or not trace.showlegend)
    )


def get_colorbar_traces_by_title(
    fig: Figure, dataset_name: str
) -> list[go.Scatter | go.Heatmap]:
    """Get colorbar traces for a specific dataset from a figure.

    This finds traces with colorbars that have the dataset name in their title.
    """
    return [
        trace
        for trace in fig.data
        if (
            hasattr(trace, "marker")
            and hasattr(trace.marker, "colorbar")
            and hasattr(trace.marker.colorbar, "title")
            and dataset_name in (trace.marker.colorbar.title.text or "")
        )
    ]


def get_scatter_colorbar_trace(fig: Figure) -> go.Scatter:
    """Get the colorbar trace from a scatter plot figure."""
    return next(
        trace
        for trace in fig.data
        if hasattr(getattr(trace, "marker", None), "colorbar")
    )


def test_ptable_heatmap_colorbar_formatting() -> None:
    """Test that ptable_heatmap uses SI suffixes for colorbar ticks."""
    # Create test data with a wide range of values
    test_data = {"O": 10_000, "C": 100_000, "H": 1_000_000, "N": 10_000_000}

    # Test with default settings
    fig = pmv.ptable_heatmap(test_data)

    # Find the colorbar trace
    colorbar_trace = get_main_colorbar_trace(fig)
    assert colorbar_trace.colorbar.tickformat == "~s"
    assert colorbar_trace.colorbar.title.text is None

    # Test with log scale
    fig_log = pmv.ptable_heatmap(test_data, log=True)

    # Check that log scale colorbar has formatted tick labels
    colorbar_trace = get_main_colorbar_trace(fig_log)
    assert hasattr(colorbar_trace.colorbar, "ticktext"), (
        "Log scale colorbar should have ticktext"
    )

    expected = ("10k", "20k", "50k", "100k", "200k", "500k", "1M", "2M", "5M", "10M")
    assert colorbar_trace.colorbar.ticktext == expected


@pytest.mark.parametrize(
    ("colorbar", "expected_tickformat"),
    [(None, "~s"), ({"tickformat": ".1~s"}, ".1~s")],
)
def test_ptable_heatmap_splits_colorbar_formatting(
    colorbar: dict[str, str] | None,
    expected_tickformat: str,
) -> None:
    """ptable_heatmap_splits should use SI suffixes for colorbar ticks."""
    # Create test data with a wide range of values
    test_data = pd.DataFrame({"Dataset 1": {"Fe": 1_000, "O": 10_000, "C": 100_000}})
    test_data["Dataset 2"] = test_data["Dataset 1"] * 2

    fig = pmv.ptable_heatmap_splits(test_data, colorbar=colorbar)

    # Find traces with colorbars for each dataset
    dataset1_colorbars = get_colorbar_traces_by_title(fig, "Dataset 1")
    dataset2_colorbars = get_colorbar_traces_by_title(fig, "Dataset 2")

    # Verify we have colorbars for both datasets
    assert len(dataset1_colorbars) > 0
    assert len(dataset2_colorbars) > 0

    # Verify that all colorbars use SI suffixes
    for trace in dataset1_colorbars + dataset2_colorbars:
        assert isinstance(trace, go.Scatter)
        assert trace.marker.colorbar.tickformat == expected_tickformat, (
            "Colorbar not using SI suffixes"
        )


def test_ptable_hists_colorbar_formatting() -> None:
    """Test that ptable_hists uses SI suffixes for colorbar ticks."""
    # Create test data with a wide range of values
    test_data = {
        "Fe": np.logspace(2, 6, 100),  # Values from 100 to 1,000,000
        "O": np.logspace(2, 6, 100),
        "C": np.logspace(2, 6, 100),
    }

    # Test with default settings and colorbar
    fig = pmv.ptable_hists(test_data, colorbar={"title": "Test"})

    # Find the colorbar trace
    colorbar_trace = get_scatter_colorbar_trace(fig)

    # Verify colorbar settings
    assert colorbar_trace.marker.colorbar.tickformat == "~s", (
        "Colorbar not using SI suffixes"
    )
    assert colorbar_trace.marker.colorbar.title.text == "Test", (
        "Colorbar title not set correctly"
    )


def test_ptable_scatter_colorbar_si_formatting() -> None:
    """Test that ptable_scatter uses SI suffixes for colorbar ticks."""
    # Create test data with a wide range of values for color
    test_data = {
        "Fe": ([1, 2, 3], [4, 5, 6], [100, 1_000, 10_000]),  # x, y, color
        "O": ([1, 2, 3], [4, 5, 6], [100, 1_000, 10_000]),
        "C": ([1, 2, 3], [4, 5, 6], [100, 1_000, 10_000]),
    }

    # Test with SI suffixes for colorbar
    fig = pmv.ptable_scatter(test_data, colorbar={"tickformat": ".1~s"})

    # Find the colorbar trace
    colorbar_trace = get_scatter_colorbar_trace(fig)

    # Verify that the colorbar uses SI suffixes
    assert colorbar_trace.marker.colorbar.tickformat == ".1~s"


def count_si_formatted_axes(fig: Figure, axis_type: str = "xaxis") -> int:
    """Count the number of axes with SI suffixes.

    Args:
        fig (Figure): The figure to check
        axis_type (str): The type of axis to check ('xaxis' or 'yaxis')

    Returns:
        int: The number of axes with SI suffixes
    """
    count = 0
    layout_dict = fig.layout._props

    for attr_name in layout_dict:
        if attr_name.startswith(axis_type):
            axis = getattr(fig.layout, attr_name)
            # Check if tickformat uses D3 SI suffixes
            if getattr(axis, "tickformat", "").endswith("s"):
                count += 1

    return count


def test_ptable_hists_log_scale_formatting() -> None:
    """Test that ptable_hists with log=True correctly formats tick labels."""
    # Create test data with a wide range of values
    test_data = {
        "Fe": np.logspace(2, 6, 100),  # Values from 100 to 1,000,000
        "O": np.logspace(2, 6, 100),
        "C": np.logspace(2, 6, 100),
    }

    # Test with log scale
    fig = pmv.ptable_hists(test_data, log=True, colorbar={"title": "Test"})

    # Find the colorbar trace
    colorbar_trace = get_scatter_colorbar_trace(fig)

    # Verify that the colorbar uses SI suffixes
    assert colorbar_trace.marker.colorbar.tickformat == "~s"
    assert colorbar_trace.marker.colorbar.title.text == "Test", (
        "Colorbar title not set correctly"
    )

    # Check that at least one x-axis has the SI suffixes
    si_formatted_axes_count = count_si_formatted_axes(fig, "xaxis")
    assert si_formatted_axes_count > 0
