from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from pymatviz import powerups
from pymatviz.widgets.binned_scatter_plot import BinnedScatterPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from tests.conftest import y_pred, y_true


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def make_series(n_series: int = 1) -> list[dict[str, Any]]:
    """n_series labeled, colored slices of the shared regression data."""
    chunks = np.array_split(np.arange(len(y_true)), n_series)
    return [
        {"x": y_true[idx], "y": y_pred[idx], "label": f"s{num}", "color": f"c{num}"}
        for num, idx in enumerate(chunks)
    ]


def make_widget(n_series: int = 1) -> ScatterPlotWidget:
    """Parity ScatterPlotWidget with n_series slices of the regression data."""
    return ScatterPlotWidget(series=make_series(n_series))


@pytest.mark.parametrize(
    ("metrics", "fmt", "prefix", "suffix", "expected"),
    [
        (("MAE", "R2"), ".3", "", "", "MAE = 0.121\nR² = 0.785"),
        ("RMSE", ".2f", "Model A\n", "", "Model A\nRMSE = 0.15"),
        (
            {"Mean err": "MAE", "Fixed": 1.5},
            ".1f",
            "",
            "\nn=100",
            "Mean err = 0.1\nFixed = 1.5\nn=100",
        ),
        (["max_error", "R2_adj"], ".2f", "", "", "max_error = 0.47\nR²_adj = 0.78"),
        # default ".3g" keeps 3-digit values out of scientific notation (".3": 1.23e+02)
        ({"Count": 123, "Big": 1234.5}, None, "", "", "Count = 123\nBig = 1.23e+03"),
    ],
)
@pytest.mark.parametrize("widget_cls", [ScatterPlotWidget, BinnedScatterPlotWidget])
def test_annotate_metrics(
    metrics: Mapping[str, float | str] | Sequence[str] | str,
    fmt: str | None,
    prefix: str,
    suffix: str,
    expected: str,
    widget_cls: type,
) -> None:
    """Metrics of the series data (or explicit xs/ys) go in the annotation badge."""
    widget = widget_cls(series=make_series(2))
    kwargs: dict[str, Any] = dict(metrics=metrics, prefix=prefix, suffix=suffix)
    if fmt is not None:
        kwargs["fmt"] = fmt
    assert powerups.annotate_metrics(widget, **kwargs) is widget
    assert widget.annotation == expected
    explicit = powerups.annotate_metrics(make_widget(), y_true, y_pred, **kwargs)
    assert explicit.annotation == expected
    # repeated calls stack blocks
    powerups.annotate_metrics(widget, series=0, metrics=["MAE"])
    assert widget.annotation.startswith(f"{expected}\nMAE = ")


@pytest.mark.parametrize(
    ("kwargs", "err_type", "match"),
    [
        ({"metrics": "foo"}, ValueError, "Unrecognized metric 'foo'"),
        ({"metrics": 42}, TypeError, "metrics must be dict|list|tuple|set"),
        ({"metrics": {"a": [1]}}, TypeError, "must be numbers or metric names"),
        ({"series": 5}, IndexError, r"series indices \[5\] out of range"),
        ({"series": [True]}, IndexError, r"series indices \[True\] out of range"),
        ({"series": True}, IndexError, r"series indices \[True\] out of range"),
        ({"xs": [1, 2], "ys": [1]}, ValueError, "equal length"),
        # metrics of no points would show MAE = nan (with RuntimeWarnings)
        (
            {"xs": [np.nan, 1], "ys": [1, np.inf]},
            ValueError,
            r"need finite \(x, y\) pairs, got none of 2",
        ),
    ],
)
def test_annotate_metrics_raises(
    kwargs: dict[str, Any], err_type: type[Exception], match: str
) -> None:
    """Invalid metrics, series indices and data shapes raise."""
    with pytest.raises(err_type, match=match):
        powerups.annotate_metrics(make_widget(), **kwargs)


@pytest.mark.parametrize(
    "func",
    [
        powerups.annotate_metrics,
        powerups.add_identity_line,
        powerups.add_best_fit_line,
        powerups.enhance_parity_plot,
        powerups.add_ecdf_line,
    ],
)
def test_powerups_reject_non_widgets(func: Any) -> None:
    """Powerups need a MatterViz scatter widget, and series data unless y = x."""
    with pytest.raises(TypeError, match="widget must be ScatterPlotWidget"):
        func({"data": []})
    if func is not powerups.add_identity_line:
        with pytest.raises(ValueError, match="has no series"):
            func(ScatterPlotWidget(series=[]))


@pytest.mark.parametrize("widget_cls", [ScatterPlotWidget, BinnedScatterPlotWidget])
@pytest.mark.parametrize("style", [None, {"color": "red", "dash": ""}])
def test_add_identity_line(widget_cls: type, style: dict[str, str] | None) -> None:
    """The y = x line is a diagonal ref line (BinnedScatterPlot: an overlay)."""
    widget = powerups.add_identity_line(widget_cls(series=make_series()), style=style)
    lines = (
        widget.overlays["ref_lines"]
        if isinstance(widget, BinnedScatterPlotWidget)
        else widget.ref_lines
    )
    (line,) = lines
    assert (line["type"], line["slope"], line["intercept"]) == ("diagonal", 1, 0)
    expected_style = {"color": "gray", "width": 1, "dash": "6 4"} | (style or {})
    assert line["style"] == expected_style


@pytest.mark.parametrize(
    ("series", "explicit", "expected_color"),
    [(None, False, "currentColor"), (1, False, "c1"), (None, True, "currentColor")],
)
def test_add_best_fit_line(
    series: int | None, explicit: bool, expected_color: str
) -> None:
    """The LS fit segment spans the fitted x range and is labeled with its fit."""
    widget = make_widget(2)
    xs, ys = (y_true[:20], y_pred[:20]) if explicit else ((), ())
    powerups.add_best_fit_line(widget, xs, ys, series=series)
    (line,) = widget.ref_lines
    if explicit:
        fit_x, fit_y = xs, ys
    else:
        picked = widget.series if series is None else [widget.series[series]]
        fit_x = np.concatenate([srs["x"] for srs in picked])
        fit_y = np.concatenate([srs["y"] for srs in picked])
    slope, intercept = np.polyfit(fit_x, fit_y, 1)
    assert line["metadata"] == {
        "slope": slope,
        "intercept": intercept,
        "log_x": False,
        "log_y": False,
    }
    assert line["p1"] == [min(fit_x), slope * min(fit_x) + intercept]
    assert line["p2"] == [max(fit_x), slope * max(fit_x) + intercept]
    assert line["style"]["color"] == line["annotation"]["color"] == expected_color
    sign = "+" if intercept >= 0 else "-"
    expected_text = f"LS fit: y = {slope:.2g}x {sign} {abs(intercept):.2g}"
    assert line["annotation"]["text"] == expected_text


@pytest.mark.parametrize(
    ("x_scale", "y_scale", "text"),
    [
        ("log", "log", "LS fit: log10(y) = 1.5·log10(x) + 0.48"),
        ("linear", {"type": "log"}, "LS fit: log10(y) = 0.2x + 0.48"),
        ("log", "linear", "LS fit: y = 2·log10(x) + 3"),
    ],
)
@pytest.mark.parametrize("widget_cls", [ScatterPlotWidget, BinnedScatterPlotWidget])
def test_add_best_fit_line_log_axes(
    x_scale: str | dict[str, str],
    y_scale: str | dict[str, str],
    text: str,
    widget_cls: type,
) -> None:
    """On log axes the fit is linear in log10 space and spans the positive data."""
    log_x = x_scale == "log"
    log_y = y_scale != "linear"
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 5.0, 10.0])  # x <= 0 not shown on log x
    pos = xs > 0
    if log_x and log_y:  # power law y = 3 x^1.5
        ys = 3 * np.abs(xs) ** 1.5
    elif log_y:  # exponential y = 3 * 10^(0.2 x)
        ys = 3 * 10 ** (0.2 * xs)
    else:  # logarithmic y = 2 log10(x) + 3, placeholder where x <= 0
        ys = np.where(pos, 2 * np.log10(np.where(pos, xs, 1)) + 3, -5)
    widget = widget_cls(
        series=[{"x": xs, "y": ys}],
        x_axis={"scale_type": x_scale},
        y_axis={"scale_type": y_scale},
    )
    powerups.add_best_fit_line(widget)
    lines = (
        widget.overlays["ref_lines"]
        if widget_cls is BinnedScatterPlotWidget
        else widget.ref_lines
    )
    (line,) = lines
    fit_xs = xs[pos] if log_x else xs
    true_y = dict(zip(xs.tolist(), ys.tolist(), strict=True))
    for point, x_val in ((line["p1"], fit_xs.min()), (line["p2"], fit_xs.max())):
        np.testing.assert_allclose(point, [x_val, true_y[x_val]], rtol=1e-12, atol=0)
    assert (line["metadata"]["log_x"], line["metadata"]["log_y"]) == (log_x, log_y)
    anno_text = (
        widget.annotation
        if widget_cls is BinnedScatterPlotWidget
        else line["annotation"]["text"]
    )
    assert anno_text == text


@pytest.mark.parametrize(
    ("axes", "match"),
    [
        ({"x_axis": {"scale_type": "arcsinh"}}, "x_axis scale_type='arcsinh'"),
        (
            {"y_axis": {"scale_type": {"type": "arcsinh", "threshold": 2}}},
            "y_axis scale_type=",
        ),
        (
            {"x_axis": {"scale_type": "log"}, "y_axis": {"scale_type": "log"}},
            r"needs >= 2 finite points \(positive on log axes\), got 1 of 3",
        ),
    ],
)
def test_add_best_fit_line_unsupported_axes(axes: dict[str, Any], match: str) -> None:
    """Non linear/log scales and too few positive points on log axes raise."""
    widget = ScatterPlotWidget(series=[{"x": [-1, 1, 2], "y": [1, 2, -3]}], **axes)
    with pytest.raises(ValueError, match=match):
        powerups.add_best_fit_line(widget)


def test_add_best_fit_line_options_and_errors() -> None:
    """Style overrides apply, annotations can be disabled, < 2 points raise."""
    widget = powerups.add_best_fit_line(
        make_widget(), style={"color": "red", "width": 3}, annotate_params=False
    )
    (line,) = widget.ref_lines
    assert line["style"] == {"color": "red", "width": 3, "dash": "8 4"}
    assert "annotation" not in line
    with pytest.raises(ValueError, match="needs >= 2 finite points, got 1 of 2"):
        powerups.add_best_fit_line(make_widget(), [1, np.nan], [1, 2])
    with pytest.raises(ValueError, match=r"needs varying x values, all are 1\.0"):
        powerups.add_best_fit_line(make_widget(), [1, 1, 1], [1, 2, 3])


@pytest.mark.parametrize(
    ("identity_line", "best_fit_line", "stats", "expected_lines", "has_stats"),
    [
        (True, None, True, ["diagonal", "segment"], True),  # R² = 0.79 > 0.3
        (False, False, True, [], True),
        (True, False, False, ["diagonal"], False),
        (
            {"style": {"color": "red"}},
            {"annotate_params": False},
            {"fmt": ".1"},
            ["diagonal", "segment"],
            True,
        ),
    ],
)
def test_enhance_parity_plot(
    identity_line: bool | dict[str, Any],
    best_fit_line: bool | dict[str, Any] | None,
    stats: bool | dict[str, Any],
    expected_lines: list[str],
    has_stats: bool,
) -> None:
    """Identity line, best-fit line and stats are each optional and configurable."""
    widget = powerups.enhance_parity_plot(
        make_widget(3),
        identity_line=identity_line,
        best_fit_line=best_fit_line,
        stats=stats,
    )
    assert [line["type"] for line in widget.ref_lines or []] == expected_lines
    assert bool(widget.annotation) == has_stats
    if isinstance(stats, dict):
        assert widget.annotation == "MAE = 0.1\nR² = 0.8"
    if isinstance(identity_line, dict):
        assert widget.ref_lines[0]["style"]["color"] == "red"
    if isinstance(best_fit_line, dict):
        assert "annotation" not in widget.ref_lines[1]


def test_enhance_parity_plot_forwards_data_kwargs() -> None:
    """Stats and best-fit dicts may select series or pass their own xs/ys."""
    widget = powerups.enhance_parity_plot(
        make_widget(2),
        identity_line=False,
        best_fit_line={"series": 1},
        stats={"series": 0, "metrics": ["MAE"]},
    )
    srs0, srs1 = widget.series
    expected_mae = np.abs(np.asarray(srs0["x"]) - np.asarray(srs0["y"])).mean()
    assert widget.annotation == f"MAE = {expected_mae:.3}"
    slope, _ = np.polyfit(srs1["x"], srs1["y"], 1)
    np.testing.assert_allclose(
        widget.ref_lines[0]["metadata"]["slope"], slope, rtol=1e-12, atol=0
    )
    explicit = powerups.enhance_parity_plot(
        make_widget(),
        identity_line=False,
        best_fit_line=False,
        stats={"xs": [1.0, 2.0], "ys": [1.5, 2.5], "metrics": ["MAE"]},
    )
    assert explicit.annotation == "MAE = 0.5"


def test_enhance_parity_plot_series_color() -> None:
    """A single selected series (numpy int too) colors the fit line."""
    widget = powerups.enhance_parity_plot(
        make_widget(2), series=np.int64(1), best_fit_line=True
    )
    assert widget.ref_lines[1]["style"]["color"] == "c1"


def test_enhance_parity_plot_auto_best_fit_line() -> None:
    """best_fit_line=None skips the fit for uncorrelated data."""
    rng = np.random.default_rng(seed=0)
    widget = ScatterPlotWidget(series=[{"x": rng.random(50), "y": rng.random(50)}])
    powerups.enhance_parity_plot(widget)
    assert [line["type"] for line in widget.ref_lines] == ["diagonal"]
    # R² = 1, but a log y axis leaves one positive point: skipped, not raised
    log_y = ScatterPlotWidget(
        series=[{"x": [-2, -1, 1], "y": [-2, -1, 1]}], y_axis={"scale_type": "log"}
    )
    powerups.enhance_parity_plot(log_y)
    assert [line["type"] for line in log_y.ref_lines] == ["diagonal"]
    with pytest.raises(ValueError, match="needs >= 2 finite points"):
        powerups.enhance_parity_plot(log_y, best_fit_line=True)


@pytest.mark.parametrize("values", [(), [3, 1, 2, 2]])
@pytest.mark.parametrize("n_series", [1, 2])
def test_add_ecdf_line(values: Sequence[float], n_series: int) -> None:
    """ECDF step series go on a y2 axis spanning [0, 1]."""
    widget = make_widget(n_series)
    n_before = len(widget.series)
    powerups.add_ecdf_line(widget, values, line_style={"dash": "4 2"})
    new_series = widget.series[n_before:]
    sources = (
        [values] if len(values) else [srs["x"] for srs in widget.series[:n_before]]
    )
    assert len(new_series) == len(sources)
    for srs, source in zip(new_series, sources, strict=True):
        sorted_vals = np.sort(source)
        fractions = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        # post-step vertices: each value repeated, fraction rising after it
        np.testing.assert_array_equal(srs["x"][::2], sorted_vals)
        np.testing.assert_array_equal(srs["y"][1::2], fractions[:-1])
        assert srs["y"][-1] == 1
        assert srs["y_axis"] == "y2"
        assert srs["line_style"]["dash"] == "4 2"
    labels = [srs["label"] for srs in new_series]
    if len(values) or n_series == 1:
        assert labels == ["Cumulative"]
    else:
        assert labels == ["Cumulative (s0)", "Cumulative (s1)"]
        assert [srs["line_style"]["color"] for srs in new_series] == ["c0", "c1"]
    assert widget.y2_axis == {"label": "Cumulative"}
    assert widget.y2_range == [0, 1]


def test_add_ecdf_line_ignores_own_ecdf_series() -> None:
    """Repeated ECDF calls and parity powerups skip earlier ECDF (y2) series."""
    widget = make_widget()
    powerups.add_ecdf_line(widget)
    powerups.add_ecdf_line(widget)
    assert [srs["label"] for srs in widget.series] == ["s0", *["Cumulative"] * 2]
    # a differently labeled ECDF shares the [0, 1] y2 axis with earlier ones
    powerups.add_ecdf_line(widget, [3, 1, 2], label="Reference")
    assert widget.series[-1]["label"] == "Reference"
    assert widget.y2_axis == {"label": "Cumulative"}
    powerups.annotate_metrics(widget)
    assert widget.annotation == "MAE = 0.121\nR² = 0.785"


def test_add_ecdf_line_keeps_y2_range() -> None:
    """A user-set y2_range survives adding ECDF lines."""
    widget = ScatterPlotWidget(series=make_series(), y2_range=[0, 0.5])
    powerups.add_ecdf_line(widget)
    assert widget.y2_range == [0, 0.5]


@pytest.mark.parametrize("y2_axis", [None, {"label": "Temperature"}])
def test_add_ecdf_line_rejects_foreign_y2_series(
    y2_axis: dict[str, str] | None,
) -> None:
    """An ECDF can't share the y2 axis with other data, whose range it would clip."""
    temps = {"x": [0, 1], "y": [20, 90], "y_axis": "y2", "label": "T"}
    widget = ScatterPlotWidget(series=[*make_series(), temps], y2_axis=y2_axis)
    with pytest.raises(ValueError, match="y2 axis already plots 'T' with values"):
        powerups.add_ecdf_line(widget)
    assert widget.y2_range is None


def test_add_ecdf_line_raises() -> None:
    """ECDFs need finite values and a ScatterPlotWidget."""
    with pytest.raises(ValueError, match="at least one finite value"):
        powerups.add_ecdf_line(make_widget(), [np.nan])
    with pytest.raises(TypeError, match="widget must be ScatterPlotWidget, got Bin"):
        powerups.add_ecdf_line(BinnedScatterPlotWidget(series=make_series()))  # ty: ignore[invalid-argument-type]


def test_annotate_metrics_r2_adj_two_points() -> None:
    """Adjusted R² is NaN for n <= 2 samples instead of dividing by zero."""
    widget = powerups.annotate_metrics(
        make_widget(), [1.0, 2.0], [1.1, 1.9], metrics=["R2_adj"]
    )
    assert widget.annotation == "R²_adj = nan"


@pytest.mark.parametrize(
    ("name", "sklearn_name"),
    [("R2", "r2_score"), ("MAPE", "mean_absolute_percentage_error")],
)
def test_builtin_metrics_match_sklearn(name: str, sklearn_name: str) -> None:
    """Built-in R2/MAPE match sklearn bit for bit, constant targets warn-free."""
    import sklearn.metrics

    xs, ys = np.append(y_true, 0.0), np.append(y_pred, 0.1)
    expected = getattr(sklearn.metrics, sklearn_name)(xs, ys)
    assert powerups.BUILTIN_METRICS[name](xs, ys) == expected
    with warnings.catch_warnings():  # constant targets: nan/inf, no RuntimeWarning
        warnings.simplefilter("error")
        powerups.BUILTIN_METRICS[name](np.ones(3), np.array([1.0, 2.0, 1.0]))
        powerups.BUILTIN_METRICS[name](np.ones(3), np.ones(3))


@pytest.mark.parametrize("n_points", [50, 20_000])
@pytest.mark.parametrize(
    "func",
    [
        powerups.annotate_metrics,
        powerups.add_best_fit_line,
        powerups.add_ecdf_line,
        powerups.enhance_parity_plot,
    ],
)
def test_powerups_read_json_and_binary_series(func: Any, n_points: int) -> None:
    """Powerups read small JSON and large binary series; large ECDFs sync binary."""
    from pymatviz.widgets._binary import BINARY_FORMAT, series_column

    xs = np.random.default_rng(seed=0).normal(size=n_points)
    widget = ScatterPlotWidget(series=[{"x": xs, "y": 1.1 * xs + 0.1, "label": "a"}])
    is_binary = n_points >= 10_000
    assert (widget.series[0].get("format") == BINARY_FORMAT) is is_binary
    func(widget)
    if func is powerups.add_ecdf_line:
        ecdf = widget.series[-1]
        assert (ecdf.get("format") == BINARY_FORMAT) is is_binary
        assert ecdf["label"] == "Cumulative"
        np.testing.assert_array_equal(series_column(ecdf, "x")[::2], np.sort(xs))
        assert series_column(ecdf, "y")[-1] == 1
    if func in (powerups.add_best_fit_line, powerups.enhance_parity_plot):
        fit = widget.ref_lines[-1]["metadata"]
        np.testing.assert_allclose(  # exact line y = 1.1x + 0.1, f64 lstsq round-off
            [fit["slope"], fit["intercept"]], [1.1, 0.1], rtol=0, atol=1e-12
        )
    if func in (powerups.annotate_metrics, powerups.enhance_parity_plot):
        mae = float(np.abs(xs - (1.1 * xs + 0.1)).mean())
        assert f"MAE = {mae:.3}" in widget.annotation
