"""Tests for plot widgets: scatter, bar, histogram, 3D scatter, heatmap matrix and
spacegroup bar plots.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import traitlets as tl

from pymatviz.widgets.bar_plot import BarPlotWidget
from pymatviz.widgets.box_plot import BoxPlotWidget
from pymatviz.widgets.heatmap_matrix import HeatmapMatrixWidget
from pymatviz.widgets.histogram import HistogramWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from pymatviz.widgets.scatter_plot_3d import ScatterPlot3DWidget
from pymatviz.widgets.spacegroup_bar import SpacegroupBarPlotWidget


def test_to_dict_reflects_constructor_values_and_mutations() -> None:
    """to_dict values match the constructor args and pick up later mutations."""
    widget = HistogramWidget(
        series=[{"values": [2, 3], "label": "h"}],
        bins=25,
        mode="overlay",
        selected_series_idx=1,
        x_axis={"label": "Value"},
    )
    state = widget.to_dict()
    assert state["bins"] == 25
    assert state["mode"] == "overlay"
    assert state["selected_series_idx"] == 1
    assert state["series"] == [{"values": [2.0, 3.0], "label": "h"}]
    assert state["x_axis"] == {"label": "Value"}

    widget.x_axis = {"label": "updated"}
    assert widget.to_dict()["x_axis"] == {"label": "updated"}

    widget.style = "height: 999px;"
    assert widget.to_dict()["style"] == "height: 999px;"


@pytest.mark.parametrize(
    "samples", [[2, 1], (2, 1), np.array([2, 1]), pd.Series([2, 1])]
)
@pytest.mark.parametrize("hold_notifications", [False, True])
def test_histogram_series_updates_normalize(
    samples: Any, hold_notifications: bool
) -> None:
    """Updated samples match constructor normalization and produce JSON-safe state."""
    series: Any = ({"values": samples, "label": "updated", "opacity": np.float32(0.5)},)
    widget = HistogramWidget(series=[{"values": [0]}])
    with widget.hold_trait_notifications() if hold_notifications else nullcontext():
        widget.series = series

    expected = [{"values": [2.0, 1.0], "label": "updated", "opacity": 0.5}]
    assert widget.series == HistogramWidget(series=series).series == expected
    dict_json = json.dumps(widget.to_dict(), allow_nan=False)
    state_json = json.dumps(widget.get_state(), allow_nan=False)
    assert json.loads(dict_json)["series"] == expected
    assert json.loads(state_json)["series"] == expected


@pytest.mark.parametrize(
    ("series", "match"),
    [
        ([{"x": [0, 1], "y": [2, 3]}], "must include key 'values'"),
        ([{"values": [float("nan")]}], "non-finite"),
        ([{"values": [float("inf")]}], "non-finite"),
        ([{"values": ["bad"]}], "must be numeric"),
        ([{"values": 1}], "must be list-like"),
        ([1], "entries must be dicts"),
    ],
)
@pytest.mark.parametrize("hold_notifications", [False, True])
def test_histogram_series_updates_reject_invalid(
    series: Any, match: str, hold_notifications: bool
) -> None:
    """Invalid series fail on init; updates leave state intact and never notify."""
    with pytest.raises(tl.TraitError, match=match):
        HistogramWidget(series=series)
    widget = HistogramWidget(series=[{"values": [1, 2]}])
    old_state = widget.to_dict()
    observer = Mock()
    widget.observe(observer, names="series")
    with (
        pytest.raises(tl.TraitError, match=match),
        widget.hold_trait_notifications() if hold_notifications else nullcontext(),
    ):
        widget.series = series
    assert widget.to_dict() == old_state
    observer.assert_not_called()


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (
            ({"x": np.array([1, 2]), "y": pd.Series([3.0, 4.0]), "label": "b"},),
            [{"x": [1, 2], "y": [3.0, 4.0], "label": "b"}],
        ),
        ([{"x": [1, 2], "y": np.array([np.nan, 2.0])}], "non-finite"),
        ([{"x": [1, 2]}], "must include keys 'x' and 'y'"),
    ],
)
def test_bar_plot_series_updates_normalize(series: Any, expected: Any) -> None:
    """Reassigned BarPlotWidget series normalize like the constructor or raise."""
    widget = BarPlotWidget(series=[{"x": [0], "y": [1]}])
    if isinstance(expected, str):
        old_state = widget.to_dict()
        with pytest.raises(tl.TraitError, match=expected):
            widget.series = series
        assert widget.to_dict() == old_state
        return
    widget.series = series
    assert widget.series == BarPlotWidget(series=series).series == expected
    state_json = json.dumps(widget.get_state(), allow_nan=False)
    assert json.loads(state_json)["series"] == expected


@pytest.mark.parametrize("series", [None, [], [{"values": []}]])
def test_histogram_series_updates_clear(series: Any) -> None:
    """Clearing or providing an empty sample array stays valid after construction."""
    widget = HistogramWidget(series=[{"values": [1, 2]}])
    widget.series = series
    assert widget.series == HistogramWidget(series=series).series == series


@pytest.mark.parametrize(
    ("series", "match"),
    [
        ([{"x": [1], "y": [2], "label": "no-z"}], "missing required key"),
        (["not-a-dict"], "must be a dict"),
        ([{1: [0], "y": [1], "z": [2]}], "non-string keys"),
        ([{"x": [1, 2], "y": [2], "z": [3]}], "mismatched x/y/z lengths"),
        ([{"x": np.array([1, np.nan]), "y": [2, 3], "z": [3, 4]}], "non-finite"),
    ],
)
def test_scatter_plot_3d_rejects_invalid_series(series: list[Any], match: str) -> None:
    """ScatterPlot3DWidget rejects malformed series on init and reassignment, keeping
    prior data, and normalizes valid reassignments to JSON.
    """
    with pytest.raises(tl.TraitError, match=match):
        ScatterPlot3DWidget(series=series)
    widget = ScatterPlot3DWidget(series=[{"x": [1], "y": [2], "z": [3]}])
    with pytest.raises(tl.TraitError, match=match):
        widget.series = series
    assert widget.series == [{"x": [1], "y": [2], "z": [3]}]
    widget.series = [
        {"x": np.array([1.5]), "y": (2,), "z": np.array([3], dtype=np.float32)}
    ]
    assert widget.series == [{"x": [1.5], "y": [2], "z": [3.0]}]


@pytest.mark.parametrize(
    "widget_cls",
    [ScatterPlotWidget, BarPlotWidget, HistogramWidget],
)
def test_show_not_shadowed_by_display_traitlet(widget_cls: type) -> None:
    """show() remains callable on widgets that define a 'display' traitlet."""
    widget = widget_cls(display={"x_grid": True})
    with patch("IPython.display.display") as mock_display:
        assert widget.show() is None
    mock_display.assert_called_once_with(widget)


def test_spacegroup_bar_data_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """SpacegroupBarPlotWidget counts every input form per space group number."""
    import pymatviz.structures as pmv_structures

    widget = SpacegroupBarPlotWidget(data=pd.Series([225, 166, 62, 225]))
    assert widget.data == {"62": 1, "166": 1, "225": 2}
    # number, symbol and synced digit-string keys of one group merge; zeros drop
    widget.data = {225: np.int64(2), "Fm-3m": 1, "225": 1, 166: 0, "P2_1/c": 1}
    assert widget.data == {"14": 1, "225": 4}
    widget.data = widget.data  # re-validating the synced form is a no-op
    assert widget.data == {"14": 1, "225": 4}
    # mapping keys normalize in one batch per key type (numbers, symbols), not one
    # call per key
    n_calls = 0
    real_normalize_spacegroups = pmv_structures.normalize_spacegroups

    def counting_normalize(*args: Any) -> Any:
        """Count normalize_spacegroups calls."""
        nonlocal n_calls
        n_calls += 1
        return real_normalize_spacegroups(*args)

    monkeypatch.setattr(pmv_structures, "normalize_spacegroups", counting_normalize)
    widget.data = {"Fm-3m": 1, "P2_1/c": 2, "Pnma": 3}
    assert (widget.data, n_calls) == ({"14": 2, "62": 3, "225": 1}, 1)
    widget.data = {225: 1, np.int64(62): 2, "225": 3, "Fm-3m": 4, "P 1 21/c 1": 5}
    assert (widget.data, n_calls) == ({"14": 5, "62": 2, "225": 8}, 3)
    for bad_data, match in [
        ({225: -1}, "non-negative"),
        ({225: 1.5}, "integers"),
        ({"not-a-spg": 1}, "Invalid"),
    ]:
        with pytest.raises(tl.TraitError, match=match):
            SpacegroupBarPlotWidget(data=bad_data)


DF_XY = pd.DataFrame(
    {"a": [1.0, 2.0, 3.0, np.nan], "b": [4.0, 5.0, 6.0, 7.0], "g": list("ppqq")}
)


@pytest.mark.parametrize(
    ("widget_cls", "columns", "series_key", "x_label"),
    [
        (ScatterPlotWidget, {"x": "a", "y": "b"}, "x", "a"),
        (BarPlotWidget, {"x": "a", "y": "b"}, "x", "a"),
        (BoxPlotWidget, {"y": "a"}, "y", None),
        (HistogramWidget, {"x": "a"}, "values", "a"),
    ],
)
def test_cartesian_widgets_take_dataframe_columns(
    widget_cls: type, columns: dict[str, str], series_key: str, x_label: str | None
) -> None:
    """DataFrame columns become series (one per color group, NaN rows dropped) and
    default axis labels that user axis options extend.
    """
    widget = widget_cls(DF_XY, color="g", y_axis={"scale_type": "log"}, **columns)
    assert [series_dict["label"] for series_dict in widget.series] == ["p", "q"]
    assert [list(series_dict[series_key]) for series_dict in widget.series] == [
        [1.0, 2.0],
        [3.0],
    ]
    y_label = {"label": columns["y"]} if "y" in columns else {}
    assert widget.y_axis == {"scale_type": "log"} | y_label
    assert (widget.x_axis or {}).get("label") == x_label
    assert len(widget_cls(DF_XY, **columns).series) == 1  # no color: one series


@pytest.mark.parametrize(
    ("data", "kwargs", "err", "match"),
    [
        ([{"x": [1], "y": [2]}], {"x": "a"}, TypeError, r"columns \['a', None\] need"),
        (DF_XY, {"x": "a"}, ValueError, r"column\(s\) for \['y'\]"),
        (DF_XY, {"x": "a", "y": "nope"}, KeyError, "nope"),
    ],
)
def test_cartesian_dataframe_errors(
    data: Any, kwargs: dict[str, str], err: type[Exception], match: str
) -> None:
    """Column names need a DataFrame that has them all."""
    with pytest.raises(err, match=match):
        ScatterPlotWidget(data, **kwargs)


def test_heatmap_matrix_takes_dataframe() -> None:
    """A DataFrame's columns, index and cells (NaN as None) become x_items, y_items
    and values.
    """
    df_matrix = pd.DataFrame({"c1": [1, 3], "c2": [2, 4]}, index=pd.Index(["r1", "r2"]))
    widget = HeatmapMatrixWidget(df_matrix)
    assert [item["key"] for item in widget.x_items] == ["c1", "c2"]
    assert [item["key"] for item in widget.y_items] == ["r1", "r2"]
    assert widget.values == [[1, 2], [3, 4]]  # noqa: PD011
    with pytest.raises(ValueError, match="in the DataFrame, not both"):
        HeatmapMatrixWidget(df_matrix, values=[[0]])
    # NaN cells become None (empty cells) instead of failing as non-finite
    widget = HeatmapMatrixWidget(pd.DataFrame({"a": [1.0, float("nan")]}))
    assert widget.values == [[1.0], [None]]  # noqa: PD011


@pytest.mark.parametrize("orientation", ["horizontal", "HORIZONTAL"])
def test_bar_plot_dataframe_column_roles_and_orientation(orientation: str) -> None:
    """One column may be x and color, "" is a column name too, and horizontal bars
    (in any case) label y with the x column.
    """
    df_bars = pd.DataFrame({"c": ["a", "b"], "v": [1, 2], "": ["p", "q"]})
    for color, labels in (("c", ["a", "b"]), ("", ["p", "q"])):
        widget = BarPlotWidget(df_bars, x="c", y="v", color=color)
        assert [series_dict["label"] for series_dict in widget.series] == labels
    flipped = BarPlotWidget(df_bars, x="c", y="v", orientation=orientation)
    assert (flipped.y_axis, flipped.x_axis) == ({"label": "c"}, {"label": "v"})
