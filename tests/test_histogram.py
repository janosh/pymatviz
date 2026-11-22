from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pymatviz.histogram import elements_hist, histogram
from pymatviz.widgets.bar_plot import BarPlotWidget
from pymatviz.widgets.histogram import HistogramWidget
from tests.conftest import df_regr, y_true


@pytest.fixture
def python_bins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bin every histogram in Python, as for data too large for the browser."""
    # the pymatviz.histogram attribute is the histogram function, not its module
    histogram_mod = importlib.import_module("pymatviz.histogram")
    monkeypatch.setattr(histogram_mod, "MAX_BROWSER_HISTOGRAM_VALUES", 0)


@pytest.mark.parametrize(
    ("kwargs", "browser"),
    [
        ({}, True),
        ({"density": True}, False),
        ({"bins": "auto"}, False),
        ({"bin_width": 0.5}, False),
        ({"orientation": "horizontal"}, False),
    ],
)
def test_histogram_bins_in_browser_unless_python_needed(
    kwargs: dict[str, Any], browser: bool
) -> None:
    """Raw values go to the browser unless an option needs NumPy binning."""
    opts: dict[str, Any] = {"bins": np.int64(4)} | kwargs  # NumPy ints as bins
    widget = histogram({"a": [0.0, 1.0, 2.0], "b": [5.0, 6.0]}, **opts)
    assert isinstance(widget, HistogramWidget if browser else BarPlotWidget)
    if isinstance(widget, HistogramWidget):
        assert (widget.bins, widget.mode, widget.show_legend) == (4, "overlay", True)
        assert [list(series_dict["values"]) for series_dict in widget.series] == [
            [0, 1, 2],
            [5, 6],
        ]
        assert widget.y_axis == {"label": "Count", "scale_type": "linear"}


def test_histogram_browser_x_range_and_dataframe() -> None:
    """A DataFrame column splits by color; values outside x_range are dropped."""
    df_vals = pd.DataFrame({"v": [-1.0, 0.5, 2.0, 3.0], "g": list("ppqq")})
    widget = histogram(df_vals, x="v", color="g", x_range=(0, 2.5))
    assert [series_dict["label"] for series_dict in widget.series] == ["p", "q"]
    assert [list(series_dict["values"]) for series_dict in widget.series] == [
        [0.5],
        [2.0],
    ]
    assert (widget.x_range, widget.x_axis) == ([0, 2.5], {"label": "v"})
    with pytest.raises(ValueError, match="name the DataFrame column"):
        histogram(df_vals)
    with pytest.raises(TypeError, match="need a DataFrame"):
        histogram([1.0], x="v")


@pytest.mark.parametrize(
    ("kwargs", "expected_x", "expected_y", "expected_labels", "scale_type"),
    [
        (
            {},
            ["O", "Fe", "Li", "P"],
            [7, 3, 1, 1],
            ["58%", "25%", "8%", "8%"],
            "linear",
        ),
        (
            {"log_y": True},
            ["O", "Fe", "Li", "P"],
            [7, 3, 1, 1],
            ["58%", "25%", "8%", "8%"],
            "log",
        ),
        # percent labels stay shares of all elements, not just the kept ones
        ({"max_elements": 2}, ["O", "Fe"], [7, 3], ["58%", "25%"], "linear"),
        (
            {"show_values": "count"},
            ["O", "Fe", "Li", "P"],
            [7, 3, 1, 1],
            ["7", "3", "1", "1"],
            "linear",
        ),
        ({"show_values": None}, ["O", "Fe", "Li", "P"], [7, 3, 1, 1], None, "linear"),
    ],
)
def test_elements_hist(
    kwargs: dict[str, Any],
    expected_x: list[str],
    expected_y: list[int],
    expected_labels: list[str] | None,
    scale_type: str,
) -> None:
    """Element counts, bar labels, log axis and style options reach the widget."""
    widget = elements_hist(
        ["Fe2O3", "LiFePO4"],
        bar_width=0.5,
        opacity=0.9,
        y_axis={"format": "d"},
        bar={"stroke_width": 3},
        **kwargs,
    )
    assert isinstance(widget, BarPlotWidget)
    (series,) = widget.series
    assert series["x"] == expected_x
    assert series["y"] == expected_y
    assert series.get("labels") == expected_labels
    assert series["bar_width"] == 0.5
    assert widget.y_axis["scale_type"] == scale_type
    assert widget.y_axis["format"] == "d"  # user axis config merged over defaults
    assert widget.bar == {"opacity": 0.9, "stroke_width": 3, "stroke_color": "black"}
    assert widget.show_legend is False


def test_elements_hist_invalid_show_values() -> None:
    """Unknown show_values raise instead of silently showing count labels."""
    with pytest.raises(ValueError, match="Invalid show_values"):
        elements_hist(["Fe2O3"], show_values="fraction")  # ty: ignore[invalid-argument-type]


@pytest.mark.usefixtures("python_bins")
@pytest.mark.parametrize("log_y", [True, False])
@pytest.mark.parametrize("bins", [20, 100])
@pytest.mark.parametrize("values", [y_true, df_regr.y_true])
def test_python_histogram(values: Any, log_y: bool, bins: int) -> None:
    """Counts match np.histogram over the data range and axes are configured."""
    widget = histogram(values, log_y=log_y, bins=bins)
    (series,) = widget.series
    counts, edges = np.histogram(np.asarray(values), bins=bins)
    np.testing.assert_array_equal(series["y"], counts)
    np.testing.assert_allclose(series["x"], (edges[:-1] + edges[1:]) / 2, rtol=1e-15)
    assert widget.y_axis == {
        "label": "Count",
        "scale_type": "log" if log_y else "linear",
    }
    assert widget.x_axis == {"label": getattr(values, "name", None) or "Value"}
    assert widget.mode == "overlay"
    assert widget.show_legend is False


@pytest.mark.usefixtures("python_bins")
def test_python_histogram_dict_shares_bin_edges() -> None:
    """Dict input bins every series on edges spanning all values, with a legend."""
    widget = histogram(
        {"a": [0, 1], "b": [9, 10]}, bins=5, mode="grouped", bar={"opacity": 0.3}
    )
    assert widget.bar == {"border_radius": 0, "opacity": 0.3}
    assert [srs["label"] for srs in widget.series] == ["a", "b"]
    assert widget.series[0]["x"] == widget.series[1]["x"] == [1, 3, 5, 7, 9]
    # uncolored so MatterViz cycles its palette across series
    assert all("color" not in srs for srs in widget.series)
    assert widget.series[0]["y"] == [2, 0, 0, 0, 0]
    assert widget.series[1]["y"] == [0, 0, 0, 0, 2]
    assert widget.show_legend is True
    assert widget.mode == "grouped"  # kwargs override defaults
    density = histogram({"a": [0, 1], "b": [9, 10]}, bins=5, density=True)
    for srs, vals in zip(density.series, ([0, 1], [9, 10]), strict=True):
        expected = np.histogram(vals, bins=np.linspace(0, 10, 6), density=True)[0]
        np.testing.assert_allclose(srs["y"], expected, rtol=1e-15, atol=0)
    # a series without values in x_range has zero counts but no defined density
    in_out: dict[str, list[float]] = {"inside": [0.25], "outside": [10.0]}
    counts = histogram(in_out, bins=2, x_range=(0, 1))
    assert [srs["y"] for srs in counts.series] == [[1, 0], [0, 0]]
    with pytest.raises(ValueError, match=r"undefined for series \['outside'\]"):
        histogram(in_out, bins=2, x_range=(0, 1), density=True)


@pytest.mark.usefixtures("python_bins")
@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({}, "must not be empty"),
        ([], "non-empty 1D"),
        ([[1, 2], [3, 4]], "non-empty 1D"),
        ([1, np.nan, 3], r"'Value' contain 1 non-finite"),
        ({"ok": [1, 2], "bad": [np.inf, -np.inf]}, r"'bad' contain 2 non-finite"),
    ],
)
def test_python_histogram_rejects_invalid_values(values: Any, match: str) -> None:
    """Histogram rejects empty, non-1D and non-finite inputs, naming the series."""
    with pytest.raises(ValueError, match=match):
        histogram(values)


@pytest.mark.usefixtures("python_bins")
def test_python_histogram_preserves_zero_x_range_boundary() -> None:
    """Histogram treats zero x_range bounds as explicit values."""
    (series,) = histogram([1, 2, 3], x_range=(0, None), bins=3).series
    assert series["x"][0] - series["bar_width"][0] / 2 == 0


@pytest.mark.usefixtures("python_bins")
@pytest.mark.parametrize(
    ("values", "bins", "centers", "widths", "counts"),
    [
        ([0.5, 2, 4], [0, 1, 3, 6], [0.5, 2, 4.5], [1, 2, 3], [1, 1, 1]),
        ([5, 5, 5], 2, [4.75, 5.25], [0.5, 0.5], [0, 3]),
    ],
)
@pytest.mark.parametrize("density", [False, True])
def test_python_histogram_bin_geometry(
    values: list[float],
    bins: int | list[float],
    centers: list[float],
    widths: list[float],
    counts: list[int],
    density: bool,
) -> None:
    """Bin centers and widths preserve counts and density area for uneven bins."""
    (series,) = histogram(values, bins=bins, density=density, bin_width=0.5).series
    np.testing.assert_array_equal(series["x"], centers)
    np.testing.assert_array_equal(series["bar_width"], 0.5 * np.array(widths))
    expected = (
        np.array(counts) / (len(values) * np.array(widths)) if density else counts
    )
    np.testing.assert_allclose(series["y"], expected, rtol=1e-14, atol=0)


@pytest.mark.usefixtures("python_bins")
@pytest.mark.parametrize(
    ("series_name", "expected_title"),
    [(None, "Value"), ("my_col", "my_col"), (0, "0"), ("", "")],
)
def test_python_histogram_series_x_axis_label(
    series_name: str | int | None, expected_title: str
) -> None:
    """Histogram uses series name as x-axis title; only None falls back to 'Value'."""
    widget = histogram(pd.Series([1.0, 2.0, 3.0], name=series_name))
    assert widget.x_axis["label"] == expected_title
    assert widget.series[0]["label"] == expected_title


@pytest.mark.parametrize("orientation", ["horizontal", "HORIZONTAL"])
@pytest.mark.parametrize(
    ("func", "cat_label", "count_label"),
    [("histogram", "Value", "Count"), ("elements_hist", "Element", "Element Count")],
)
def test_horizontal_swaps_axis_defaults(
    func: str, cat_label: str, count_label: str, orientation: str
) -> None:
    """Horizontal bars (any case) put category defaults on y, count/log on x."""
    if func == "histogram":
        widget = histogram([1.0, 2.0, 3.0], log_y=True, orientation=orientation)
    else:
        widget = elements_hist(["Fe2O3"], log_y=True, orientation=orientation)
    assert widget.x_axis == {"label": count_label, "scale_type": "log"}
    assert widget.y_axis == {"label": cat_label}


def test_elements_hist_large_counts() -> None:
    """Integer counts keep all digits; fractional counts use 3 significant digits."""
    widget = elements_hist(pd.Series({"O": 1_234_567, "Fe": 2.5}), show_values="count")
    assert widget.series[0]["labels"] == ["1,234,567", "2.5"]


@pytest.mark.parametrize(
    ("df_vals", "color", "expected"),
    [
        # one column as values and groups
        (pd.DataFrame({"v": [1.0, 2.0]}), "v", {"1.0": [1.0], "2.0": [2.0]}),
        # unused categories make no empty distribution
        (
            pd.DataFrame(
                {"v": [1.0], "g": pd.Categorical(["a"], categories=["a", "b"])}
            ),
            "g",
            {"a": [1.0]},
        ),
    ],
)
def test_histogram_dataframe_groups(
    df_vals: pd.DataFrame, color: str, expected: dict[str, list[float]]
) -> None:
    """A column may both supply values and groups; only observed groups plot."""
    widget = histogram(df_vals, x="v", color=color)
    assert {
        series_dict["label"]: list(series_dict["values"])
        for series_dict in widget.series
    } == expected


def test_histogram_keeps_groups_equal_as_strings() -> None:
    """Groups 1 and "1" stay two distributions instead of one overwriting the other."""
    df_vals = pd.DataFrame({"v": [0.0, 10.0], "g": pd.Series([1, "1"], dtype=object)})
    widget = histogram(df_vals, x="v", color="g")
    assert sorted(list(series_dict["values"]) for series_dict in widget.series) == [
        [0.0],
        [10.0],
    ]
