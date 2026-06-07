"""Tests for widget describe()/short_summary/check_inputs (pure Python, no browser)."""

from __future__ import annotations

from typing import Any

import pytest
from pymatgen.core import Lattice, Structure

import pymatviz as pmv
from pymatviz.widgets._describe import (
    _minmax,
    check_inputs,
    describe_widget,
    short_summary,
)


_STRUCT = Structure(Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.mark.parametrize(
    ("widget", "expected"),
    [
        (
            pmv.ScatterPlotWidget(
                series=[{"x": [0, 1, 2], "y": [1, 2, 3], "label": "a"}]
            ),
            {
                "n_series": 1,
                "n_points": 3,
                "x_range": [0.0, 2.0],
                "y_range": [1.0, 3.0],
            },
        ),
        (pmv.BarPlotWidget(series=[{"x": [0, 1], "y": [2, 3]}]), {"n_series": 1}),
        (
            pmv.ScatterPlot3DWidget(series=[{"x": [1, 2], "y": [2, 3], "z": [3, 4]}]),
            {"n_series": 1, "z_range": [3.0, 4.0]},
        ),
        (
            pmv.StructureWidget(structure=_STRUCT),
            {"n_sites": 2, "formula": "Si2", "elements": ["Si"]},
        ),
        (
            pmv.PeriodicTableWidget(heatmap_values={"Fe": 42, "O": 100}),
            {"n_elements": 2, "value_range": [42.0, 100.0]},
        ),
        (
            pmv.HeatmapMatrixWidget(
                x_items=["A", "B"], y_items=["A"], values=[[1.0, 0.5]]
            ),
            {"shape": [2, 1], "value_range": [0.5, 1.0]},
        ),
        (
            pmv.CompositionWidget(composition="Fe2O3"),
            {"formula": "Fe2O3", "elements": ["Fe", "O"]},
        ),
        (
            pmv.ConvexHullWidget(
                entries=[
                    {"composition": {"Li": 1}, "energy": -1.9},
                    {"composition": {"O": 1}, "energy": -3.0},
                ]
            ),
            {"n_entries": 2, "chemical_system": ["Li", "O"]},
        ),
        (
            pmv.SpacegroupBarPlotWidget(data=[225, 225, 166]),
            {"n_entries": 3, "n_unique": 2},
        ),
        (
            pmv.XrdWidget(patterns={"x": [10, 20, 30], "y": [100, 50, 75]}),
            {"n_peaks": 3, "two_theta_range": [10.0, 30.0]},
        ),
    ],
)
def test_describe_widget(widget: Any, expected: dict[str, Any]) -> None:
    """describe() reports the widget type plus the expected structured facts."""
    report = widget.describe()
    assert report["widget_type"] == widget.widget_type
    for key, value in expected.items():
        assert report[key] == value, f"{key}: {report.get(key)!r} != {value!r}"


@pytest.mark.parametrize(
    "widget",
    [pmv.ScatterPlotWidget(), pmv.PeriodicTableWidget(), pmv.HeatmapMatrixWidget()],
)
def test_describe_handles_empty_data(widget: Any) -> None:
    """describe() tolerates empty/None data without raising."""
    report = widget.describe()
    assert report["widget_type"] == widget.widget_type


def test_describe_unknown_widget_type() -> None:
    """Unknown widget types fall back to just the type."""
    assert describe_widget({"widget_type": "mystery"}) == {"widget_type": "mystery"}


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (
            {"widget_type": "structure", "formula": "Si2", "n_sites": 2},
            "structure: Si2 (2 sites)",
        ),
        ({"widget_type": "scatter_plot", "n_series": 3}, "scatter_plot: 3 series"),
        ({"widget_type": "convex_hull", "n_entries": 5}, "convex_hull: 5 entries"),
        ({"widget_type": "xrd"}, "xrd"),
    ],
)
def test_short_summary(report: dict[str, Any], expected: str) -> None:
    """short_summary builds a one-line label from a describe report."""
    assert short_summary(report) == expected


@pytest.mark.parametrize(
    ("widget", "expect_warning"),
    [
        (pmv.ScatterPlotWidget(), True),
        (pmv.ScatterPlotWidget(series=[{"x": [0, 1], "y": [1, 2]}]), False),
        (pmv.PeriodicTableWidget(), True),
        (pmv.PeriodicTableWidget(heatmap_values={"Fe": 1}), False),
        # rdf_plot warns when neither source given, incl. empty (not just None)
        (pmv.RdfPlotWidget(), True),
        (pmv.RdfPlotWidget(structures=[]), True),
        (pmv.RdfPlotWidget(patterns=[]), True),
        (pmv.RdfPlotWidget(patterns=[{"label": "a", "x": [1.0], "y": [2.0]}]), False),
    ],
)
def test_check_inputs(widget: Any, expect_warning: bool) -> None:
    """check_inputs flags empty primary data, stays quiet for populated widgets."""
    assert bool(check_inputs(widget.to_dict())) is expect_warning


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 2.0, 3.0], [1.0, 3.0]),
        ([float("nan"), 2.0, float("inf"), 1.0], [1.0, 2.0]),  # NaN/inf dropped
        ([float("nan"), float("inf"), float("-inf")], None),  # all non-finite
        ([[1, 2], {"a": 3}], [1.0, 3.0]),  # nested flattened
        ([True, False, 5], [5.0, 5.0]),  # bools ignored
        ([], None),
    ],
)
def test_minmax_filters_non_finite(values: Any, expected: list[float] | None) -> None:
    """_minmax flattens numbers, dropping NaN/inf and bools."""
    assert _minmax(values) == expected
