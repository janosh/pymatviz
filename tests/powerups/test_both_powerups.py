from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import r2_score

import pymatviz as pmv
from pymatviz.typing import MATPLOTLIB, PLOTLY, Backend
from pymatviz.utils import pretty_label
from tests.conftest import _extract_anno_from_fig, y_pred, y_true


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


@pytest.mark.parametrize("annotate_params", [True, False, {"color": "green"}])
def test_add_best_fit_line(
    plotly_scatter: go.Figure,
    matplotlib_scatter: plt.Figure,
    annotate_params: bool | dict[str, Any],
) -> None:
    # test plotly
    fig_plotly = pmv.powerups.add_best_fit_line(
        plotly_scatter, annotate_params=annotate_params
    )
    assert isinstance(fig_plotly, go.Figure)
    best_fit_line = fig_plotly.layout.shapes[-1]  # retrieve best fit line
    assert best_fit_line.type == "line"
    expected_color = (
        annotate_params.get("color") if isinstance(annotate_params, dict) else "navy"
    )
    assert best_fit_line.line.color == expected_color

    # reconstruct slope and intercept from best fit line endpoints
    x0, x1 = best_fit_line.x0, best_fit_line.x1
    y0, y1 = best_fit_line.y0, best_fit_line.y1
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0

    if annotate_params:
        assert fig_plotly.layout.annotations[-1].text == (
            f"LS fit: y = {slope:.2g}x + {intercept:.2g}"
        )
        assert fig_plotly.layout.annotations[-1].font.color == expected_color
    else:
        assert len(fig_plotly.layout.annotations) == 0

    # test matplotlib
    fig_mpl = pmv.powerups.add_best_fit_line(
        matplotlib_scatter, annotate_params=annotate_params
    )
    assert isinstance(fig_mpl, plt.Figure)

    with pytest.raises(IndexError):
        fig_mpl.axes[1]
    best_fit_line = (ax := fig_mpl.axes[0]).lines[-1]  # retrieve best fit line
    assert best_fit_line.get_linestyle() == "--"
    assert best_fit_line.get_color() == expected_color

    anno: AnchoredText = next(
        (child for child in ax.get_children() if isinstance(child, AnchoredText)), None
    )

    x0, y0 = best_fit_line._xy1
    x1, y1 = best_fit_line._xy2
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0

    if annotate_params:
        assert anno.txt.get_text() == f"LS fit: y = {slope:.2g}x + {intercept:.2g}"
    else:
        assert anno is None


def test_add_best_fit_line_invalid_fig() -> None:
    with pytest.raises(TypeError, match="must be instance of"):
        pmv.powerups.add_best_fit_line("not a valid fig")


def test_add_best_fit_line_custom_line_kwargs(plotly_scatter: go.Figure) -> None:
    line_kwargs = {"width": 3, "dash": "dot"}
    result = pmv.powerups.add_best_fit_line(plotly_scatter, line_kwargs=line_kwargs)

    best_fit_line = result.layout.shapes[-1]
    assert best_fit_line.line.width == 2
    assert best_fit_line.line.dash == "dash"


@pytest.mark.parametrize("trace_idx", [0, 1])
def test_add_best_fit_line_trace_idx(trace_idx: int) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[3, 2, 1]))

    result = pmv.powerups.add_best_fit_line(fig, trace_idx=trace_idx)

    best_fit_line = result.layout.shapes[-1]
    expected_slope = 1 if trace_idx == 0 else -1
    actual_slope = (best_fit_line.y1 - best_fit_line.y0) / (
        best_fit_line.x1 - best_fit_line.x0
    )
    assert np.isclose(actual_slope, expected_slope, atol=1e-6)


def test_add_best_fit_line_faceted_plot(plotly_faceted_scatter: go.Figure) -> None:
    result = pmv.powerups.add_best_fit_line(plotly_faceted_scatter)

    assert len(result.layout.shapes) == 2
    assert len(result.layout.annotations) == 2
    assert (  # check that both annotations are for best fit lines
        sum(anno.text.startswith("LS fit: y =") for anno in result.layout.annotations)
        == 2
    )


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_add_best_fit_line_custom_xs_ys(
    backend: str, plotly_scatter: go.Figure, matplotlib_scatter: plt.Figure
) -> None:
    fig = plotly_scatter if backend == "plotly" else matplotlib_scatter
    custom_x = np.array([1, 2, 3, 4, 5])
    custom_y = np.array([2, 3, 4, 5, 6])

    fig_with_fit = pmv.powerups.add_best_fit_line(fig, xs=custom_x, ys=custom_y)

    if backend == "plotly":
        best_fit_line = fig_with_fit.layout.shapes[-1]
        slope = (best_fit_line.y1 - best_fit_line.y0) / (
            best_fit_line.x1 - best_fit_line.x0
        )
    else:
        ax = fig_with_fit.axes[0]
        best_fit_line = next(line for line in ax.lines if line.get_linestyle() == "--")
        x_data, y_data = best_fit_line.get_data()
        slope = (y_data[1] - y_data[0]) / (x_data[1] - x_data[0])

    expected_slope = 1.0
    assert slope == pytest.approx(expected_slope)


@pytest.mark.parametrize(
    ("xaxis_type", "yaxis_type", "trace_idx", "line_kwargs", "retain_xy_limits"),
    [
        ("linear", "log", 0, None, True),
        ("log", "linear", 1, {"color": "red"}, False),
        ("log", "log", 0, {"color": "green"}, True),
        ("linear", "linear", 1, None, False),
    ],
)
def test_add_identity_line(
    plotly_scatter: go.Figure,
    xaxis_type: str,
    yaxis_type: str,
    trace_idx: int,
    line_kwargs: dict[str, str] | None,
    retain_xy_limits: bool,
) -> None:
    # Set axis types
    plotly_scatter.layout.xaxis.type = xaxis_type
    plotly_scatter.layout.yaxis.type = yaxis_type

    # record initial axis limits
    dev_fig_pre = plotly_scatter.full_figure_for_development(warn=False)
    x_range_pre = dev_fig_pre.layout.xaxis.range
    y_range_pre = dev_fig_pre.layout.yaxis.range

    fig = pmv.powerups.add_identity_line(
        plotly_scatter,
        line_kwargs=line_kwargs,
        trace_idx=trace_idx,
        retain_xy_limits=retain_xy_limits,
    )
    assert isinstance(fig, go.Figure)

    # retrieve identity line
    line = next((shape for shape in fig.layout.shapes if shape.type == "line"), None)
    assert line is not None

    assert line.layer == "below"
    assert line.line.color == (line_kwargs["color"] if line_kwargs else "gray")
    # check line coordinates
    assert line.x0 == line.y0
    assert line.x1 == line.y1
    # check fig axis types
    assert fig.layout.xaxis.type == xaxis_type
    assert fig.layout.yaxis.type == yaxis_type

    if retain_xy_limits:
        assert dev_fig_pre.layout.xaxis.range == x_range_pre
        assert dev_fig_pre.layout.yaxis.range == y_range_pre
    else:
        dev_fig_post = fig.full_figure_for_development(warn=False)
        x_range_post = dev_fig_post.layout.xaxis.range
        y_range_post = dev_fig_post.layout.yaxis.range
        # this assumes that the x and y axis had different ranges initially which became
        # equalized by adding an identity line (which is the case for plotly_scatter)
        assert x_range_post != x_range_pre
        assert y_range_post != y_range_pre


@pytest.mark.parametrize("line_kwargs", [None, {"color": "blue"}])
def test_add_identity_matplotlib(
    matplotlib_scatter: plt.Figure, line_kwargs: dict[str, str] | None
) -> None:
    expected_line_color = (line_kwargs or {}).get("color", "black")
    # test Figure
    fig = pmv.powerups.add_identity_line(matplotlib_scatter, line_kwargs=line_kwargs)
    assert isinstance(fig, plt.Figure)

    # test Axes
    ax = pmv.powerups.add_identity_line(
        matplotlib_scatter.axes[0], line_kwargs=line_kwargs
    )
    assert isinstance(ax, plt.Axes)

    line = fig.axes[0].lines[-1]  # retrieve identity line
    assert line.get_color() == expected_line_color

    # test with new log scale axes
    _fig_log, ax_log = plt.subplots()
    ax_log.plot([1, 10, 100], [10, 100, 1000])
    ax_log.set(xscale="log", yscale="log")
    ax_log = pmv.powerups.add_identity_line(ax, line_kwargs=line_kwargs)

    line = fig.axes[0].lines[-1]
    assert line.get_color() == expected_line_color


def test_add_identity_raises() -> None:
    for fig in (None, "foo", 42.0):
        with pytest.raises(
            TypeError,
            match=f"{fig=} must be instance of plotly.graph_objs._figure.Figure | "
            f"matplotlib.figure.Figure | matplotlib.axes._axes.Axes",
        ):
            pmv.powerups.add_identity_line(fig)


@pytest.mark.parametrize(
    ("metrics", "fmt"),
    [
        ("MSE", ".1"),
        (["RMSE"], ".1"),
        (("MAPE", "MSE"), ".2"),
        ({"MAE", "R2", "RMSE"}, ".3"),
        ({"MAE": 1.4, "R2": 0.2, "RMSE": 1.9}, ".0"),
    ],
)
@pytest.mark.parametrize("backend", [PLOTLY, MATPLOTLIB])
def test_annotate_metrics(
    metrics: dict[str, float] | Sequence[str],
    fmt: str,
    plotly_scatter: go.Figure,
    matplotlib_scatter: plt.Figure,
    backend: Backend,
) -> None:
    # randomly switch between plotly and matplotlib
    fig = plotly_scatter if backend == PLOTLY else matplotlib_scatter

    out_fig = pmv.powerups.annotate_metrics(
        y_pred, y_true, metrics=metrics, fmt=fmt, fig=fig
    )
    assert out_fig is fig

    expected = dict(MAE=0.121, R2=0.784, RMSE=0.146, MAPE=0.52, MSE=0.021)

    expected_text = ""
    newline = "<br>" if isinstance(out_fig, go.Figure) else "\n"
    if isinstance(metrics, dict):
        for key, val in metrics.items():
            label = pretty_label(key, backend)
            expected_text += f"{label} = {val:{fmt}}{newline}"
    else:
        for key in [metrics] if isinstance(metrics, str) else metrics:
            label = pretty_label(key, backend)
            expected_text += f"{label} = {expected[key]:{fmt}}{newline}"

    anno_text = _extract_anno_from_fig(out_fig)
    assert anno_text == expected_text, f"{anno_text=}"

    prefix, suffix = f"Metrics:{newline}", f"{newline}the end"
    out_fig = pmv.powerups.annotate_metrics(
        y_pred, y_true, metrics=metrics, fmt=fmt, prefix=prefix, suffix=suffix, fig=fig
    )
    anno_text_with_fixes = _extract_anno_from_fig(out_fig)
    assert anno_text_with_fixes == prefix + expected_text + suffix, (
        f"{anno_text_with_fixes=}"
    )


def test_annotate_metrics_faceted_plotly(plotly_faceted_scatter: go.Figure) -> None:
    out_fig = pmv.powerups.annotate_metrics(y_true, y_pred, fig=plotly_faceted_scatter)

    assert len(out_fig.layout.annotations) == 2
    for anno in out_fig.layout.annotations:
        assert "MAE" in anno.text
        assert "R<sup>2</sup>" in anno.text


def test_annotate_metrics_prefix_suffix(plotly_scatter: go.Figure) -> None:
    prefix, suffix = "Metrics:", "End"
    out_fig = pmv.powerups.annotate_metrics(
        y_true, y_pred, fig=plotly_scatter, prefix=prefix, suffix=suffix
    )

    anno_text = _extract_anno_from_fig(out_fig)
    assert anno_text.startswith(prefix)
    assert anno_text.endswith(suffix)


@pytest.mark.parametrize("metrics", [42, datetime.now(tz=timezone.utc)])
def test_annotate_metrics_bad_metrics(metrics: Any) -> None:
    with pytest.raises(TypeError, match="metrics must be dict|list|tuple|set"):
        pmv.powerups.annotate_metrics(y_true, y_pred, metrics=metrics)


def test_annotate_metrics_bad_fig() -> None:
    with pytest.raises(TypeError, match="Unexpected type for fig: str"):
        pmv.powerups.annotate_metrics(y_true, y_pred, fig="not a figure")


@pytest.mark.parametrize("backend", [PLOTLY, MATPLOTLIB])
def test_annotate_metrics_different_lengths(backend: Backend) -> None:
    fig = go.Figure() if backend == PLOTLY else plt.figure()
    xs, ys = y_true, y_pred[:-1]

    err_msg = f"xs and ys must have the same shape. Got {xs.shape} and {ys.shape}"
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        pmv.powerups.annotate_metrics(xs, ys, fig=fig)


@pytest.mark.parametrize(
    ("identity_line", "best_fit_line", "stats"),
    [
        (True, True, True),  # all features enabled
        (False, False, False),  # all features disabled
        ({}, {}, {}),  # custom styling
        (True, None, True),  # auto best-fit line based on R²
    ],
)
def test_enhance_parity_plot(
    plotly_scatter: go.Figure,
    matplotlib_scatter: plt.Figure,
    identity_line: bool | dict[str, Any],
    best_fit_line: bool | dict[str, Any] | None,
    stats: bool | dict[str, Any],
) -> None:
    """Test enhance_parity_plot with different combinations of features and styling."""
    # Test with plotly backend
    fig_plotly = pmv.powerups.enhance_parity_plot(
        plotly_scatter,
        identity_line=identity_line,
        best_fit_line=best_fit_line,
        stats=stats,
    )
    assert isinstance(fig_plotly, go.Figure)

    # Check identity line
    identity_lines = [
        shape
        for shape in fig_plotly.layout.shapes
        if shape.type == "line" and shape.x0 == shape.y0 and shape.x1 == shape.y1
    ]
    if identity_line:
        assert len(identity_lines) == 1
        if isinstance(identity_line, dict) and "color" in identity_line:
            assert identity_lines[0].line.color == identity_line["color"]
    else:
        assert not identity_lines

    # Check best fit line
    best_fit_lines = [
        shape
        for shape in fig_plotly.layout.shapes
        if shape.type == "line" and shape.x0 != shape.y0
    ]
    if best_fit_line or (
        best_fit_line is None
        and r2_score(fig_plotly.data[0].x, fig_plotly.data[0].y) > 0.3
    ):
        assert len(best_fit_lines) == 1
        if isinstance(best_fit_line, dict) and "color" in best_fit_line:
            assert best_fit_lines[0].line.color == best_fit_line["color"]
    else:
        assert not best_fit_lines

    # Check stats
    if stats:
        assert fig_plotly.layout.annotations
        anno_text = fig_plotly.layout.annotations[-1].text
        assert anno_text == "MAE = 3.34e+02<br>R<sup>2</sup> = -1.36e+02<br>"
        if isinstance(stats, dict) and stats.get("loc") == "upper left":
            assert fig_plotly.layout.annotations[-1].x < 0.5
            assert fig_plotly.layout.annotations[-1].y > 0.5

    # Test with matplotlib backend
    # Create test data since we can't extract it from matplotlib figure
    xs = np.array([1, 2, 3, 4, 5])
    ys = np.array([2, 3, 4, 5, 6])

    fig_mpl = pmv.powerups.enhance_parity_plot(
        matplotlib_scatter,
        xs=xs,
        ys=ys,
        identity_line=identity_line,
        best_fit_line=best_fit_line,
        stats=stats,
    )
    assert isinstance(fig_mpl, plt.Figure)

    ax = fig_mpl.axes[0]
    lines = ax.lines

    # Check identity line (should be first line if present)
    if identity_line:
        plt_identity_line = next(
            (line for line in lines if np.allclose(line.get_xdata(), line.get_ydata())),
            None,
        )
        assert isinstance(plt_identity_line, plt.Line2D)
        if isinstance(identity_line, dict) and "color" in identity_line:
            assert plt_identity_line.get_color() == identity_line["color"]

    # Check best fit line (should be last line if present)
    if best_fit_line or (best_fit_line is None and r2_score(xs, ys) > 0.3):
        plt_best_fit_line = next(
            (ln for ln in lines if not np.allclose(ln.get_xdata(), ln.get_ydata())),
            None,
        )
        assert isinstance(plt_best_fit_line, plt.Line2D)
        if isinstance(best_fit_line, dict) and "color" in best_fit_line:
            assert plt_best_fit_line.get_color() == best_fit_line["color"]

    # Check stats
    if stats:
        anno = next(
            (child for child in ax.get_children() if isinstance(child, AnchoredText)),
            None,
        )
        assert anno is not None
        anno_text = anno.txt.get_text()
        assert anno_text == "LS fit: y = 1x + 1"


def test_enhance_parity_plot_no_data_matplotlib() -> None:
    """Test enhance_parity_plot raises error when no data provided for matplotlib."""
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="this powerup can only get x/y data from"):
        pmv.powerups.enhance_parity_plot(ax)


def test_enhance_parity_plot_faceted_plotly(plotly_faceted_scatter: go.Figure) -> None:
    """Test enhance_parity_plot with faceted plotly figure."""
    fig = pmv.powerups.enhance_parity_plot(plotly_faceted_scatter)
    n_facets = len(plotly_faceted_scatter.data)

    # Should add identity line and best fit line to each subplot
    assert sum(shape.type == "line" for shape in fig.layout.shapes) == n_facets

    # Should add stats to each subplot
    assert len(fig.layout.annotations) == n_facets
    assert [anno.text for anno in fig.layout.annotations] == [
        "MAE = 3.0<br>R<sup>2</sup> = -12.5<br>"
    ] * n_facets


def test_enhance_parity_plot_custom_data() -> None:
    """Test enhance_parity_plot with custom x/y data."""
    fig = go.Figure()
    xs = np.array([1, 2, 3, 4, 5])
    ys = np.array([2, 3, 4, 5, 6])

    fig.add_scatter(x=[10, 20, 30], y=[10, 20, 30])  # dummy data

    enhanced_fig = pmv.powerups.enhance_parity_plot(fig, xs=xs, ys=ys)

    # Should use provided data for best fit line and stats
    assert len(enhanced_fig.layout.shapes) == 2  # at least identity line
    assert len(enhanced_fig.layout.annotations) == 2  # stats and least fit annotation

    # Check that stats are computed from provided data, not figure data
    # (which has perfect x/y agreement)
    anno_text = enhanced_fig.layout.annotations[-1].text
    assert anno_text == "MAE = 1.0<br>R<sup>2</sup> = 0.5<br>"
    enhanced_fig = pmv.powerups.enhance_parity_plot(fig)
    anno_text = enhanced_fig.layout.annotations[-1].text
    assert anno_text == "MAE = 0.0<br>R<sup>2</sup> = 1.0<br>"
