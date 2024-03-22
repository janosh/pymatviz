from __future__ import annotations

from datetime import datetime, timezone
from random import random
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest
from matplotlib.text import Annotation

from pymatviz.powerups import (
    add_best_fit_line,
    add_ecdf_line,
    add_identity_line,
    annotate_bars,
    annotate_metrics,
    get_fig_xy_range,
    with_marginal_hist,
)
from pymatviz.utils import Backend, pretty_label
from tests.conftest import y_pred, y_true


if TYPE_CHECKING:
    from collections.abc import Sequence


def _extract_anno_from_fig(fig: go.Figure | plt.Figure, idx: int = -1) -> str:
    # get plotly or matplotlib annotation text. idx=-1 gets the most recently added
    # annotation
    if not isinstance(fig, (go.Figure, plt.Figure)):
        raise TypeError(f"Unexpected {type(fig)=}")

    if isinstance(fig, go.Figure):
        anno_text = fig.layout.annotations[idx].text
    else:
        text_box = fig.axes[0].artists[idx]
        anno_text = text_box.txt.get_text()

    return anno_text


@pytest.mark.parametrize(
    "metrics, fmt",
    [
        ("MSE", ".1"),
        (["RMSE"], ".1"),
        (("MAPE", "MSE"), ".2"),
        ({"MAE", "R2", "RMSE"}, ".3"),
        ({"MAE": 1.4, "R2": 0.2, "RMSE": 1.9}, ".0"),
    ],
)
def test_annotate_metrics(
    metrics: dict[str, float] | Sequence[str],
    fmt: str,
    plotly_scatter: go.Figure,
    matplotlib_scatter: plt.Figure,
) -> None:
    # randomly switch between plotly and matplotlib
    fig = plotly_scatter if random() > 0.5 else matplotlib_scatter

    out_fig = annotate_metrics(y_pred, y_true, metrics=metrics, fmt=fmt, fig=fig)

    assert out_fig is fig
    backend: Backend = "plotly" if isinstance(out_fig, go.Figure) else "matplotlib"

    expected = dict(MAE=0.113, R2=0.765, RMSE=0.144, MAPE=0.5900, MSE=0.0206)

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
    out_fig = annotate_metrics(
        y_pred, y_true, metrics=metrics, fmt=fmt, prefix=prefix, suffix=suffix, fig=fig
    )
    anno_text_with_fixes = _extract_anno_from_fig(out_fig)
    assert (
        anno_text_with_fixes == prefix + expected_text + suffix
    ), f"{anno_text_with_fixes=}"


@pytest.mark.parametrize("metrics", [42, datetime.now(tz=timezone.utc)])
def test_annotate_metrics_bad_metrics(metrics: int | datetime) -> None:
    err_msg = f"metrics must be dict|list|tuple|set, not {type(metrics).__name__}"
    with pytest.raises(TypeError, match=err_msg):
        annotate_metrics(y_pred, y_true, metrics=metrics)  # type: ignore[arg-type]


def test_annote_metrics_bad_fig() -> None:
    err_msg = "Unexpected type for fig: str, must be one of"
    with pytest.raises(TypeError, match=err_msg):
        annotate_metrics(y_pred, y_true, fig="invalid")


@pytest.mark.parametrize("xaxis_type", ["linear", "log"])
@pytest.mark.parametrize("yaxis_type", ["linear", "log"])
@pytest.mark.parametrize("trace_idx", [0, 1])
@pytest.mark.parametrize("line_kwds", [None, {"color": "blue"}])
def test_add_identity_line(
    plotly_scatter: go.Figure,
    xaxis_type: str,
    yaxis_type: str,
    trace_idx: int,
    line_kwds: dict[str, str] | None,
) -> None:
    # Set axis types
    plotly_scatter.layout.xaxis.type = xaxis_type
    plotly_scatter.layout.yaxis.type = yaxis_type

    fig = add_identity_line(plotly_scatter, line_kwds=line_kwds, trace_idx=trace_idx)
    assert isinstance(fig, go.Figure)

    # retrieve identity line
    line = next((shape for shape in fig.layout.shapes if shape.type == "line"), None)
    assert line is not None

    assert line.layer == "below"
    assert line.line.color == (line_kwds["color"] if line_kwds else "gray")
    # check line coordinates
    assert line.x0 == line.y0
    assert line.x1 == line.y1
    # check fig axis types
    assert fig.layout.xaxis.type == xaxis_type
    assert fig.layout.yaxis.type == yaxis_type


@pytest.mark.parametrize("line_kwds", [None, {"color": "blue"}])
def test_add_identity_matplotlib(
    matplotlib_scatter: plt.Figure, line_kwds: dict[str, str] | None
) -> None:
    expected_line_color = (line_kwds or {}).get("color", "black")
    # test Figure
    fig = add_identity_line(matplotlib_scatter, line_kwds=line_kwds)
    assert isinstance(fig, plt.Figure)

    # test Axes
    ax = add_identity_line(matplotlib_scatter.axes[0], line_kwds=line_kwds)
    assert isinstance(ax, plt.Axes)

    line = fig.axes[0].lines[-1]  # retrieve identity line
    assert line.get_color() == expected_line_color

    # test with new log scale axes
    _fig_log, ax_log = plt.subplots()
    ax_log.plot([1, 10, 100], [10, 100, 1000])
    ax_log.set(xscale="log", yscale="log")
    ax_log = add_identity_line(ax, line_kwds=line_kwds)

    line = fig.axes[0].lines[-1]
    assert line.get_color() == expected_line_color


def test_add_identity_raises() -> None:
    for fig in (None, "foo", 42.0):
        with pytest.raises(
            TypeError,
            match=f"{fig=} must be instance of plotly.graph_objs._figure.Figure | "
            f"matplotlib.figure.Figure | matplotlib.axes._axes.Axes",
        ):
            add_identity_line(fig)


@pytest.mark.parametrize(
    "v_offset,h_offset,labels,fontsize,y_max_headroom,adjust_test_pos",
    [
        (10, 0, None, 14, 1.2, False),
        (20, 0, ["label1", "label2", "label3"], 10, 1.5, True),
        (5, 5, [100, 200, 300], 16, 1.0, False),
    ],
)
def test_annotate_bars(
    v_offset: int,
    h_offset: int,
    labels: Sequence[str] | None,
    fontsize: int,
    y_max_headroom: float,
    adjust_test_pos: bool,
) -> None:
    bars = plt.bar(["A", "B", "C"], [1, 3, 2])
    ax = plt.gca()
    annotate_bars(
        ax,
        v_offset=v_offset,
        h_offset=h_offset,
        labels=labels,
        fontsize=fontsize,
        y_max_headroom=y_max_headroom,
        adjust_test_pos=adjust_test_pos,
    )

    assert len(ax.texts) == len(bars)

    if labels is None:
        labels = [str(bar.get_height()) for bar in bars]

    # test that labels have expected text and fontsize
    for text, label in zip(ax.texts, labels):
        assert text.get_text() == str(label)
        assert text.get_fontsize() == fontsize

    # test that y_max_headroom is respected
    ylim_max = ax.get_ylim()[1]
    assert ylim_max >= max(bar.get_height() for bar in bars) * y_max_headroom

    # test error when passing wrong number of labels
    bad_labels = ("label1", "label2")
    with pytest.raises(
        ValueError,
        match=f"Got {len(bad_labels)} labels but {len(bars)} bars to annotate",
    ):
        annotate_bars(ax, labels=bad_labels)

    # test error message if adjustText not installed
    err_msg = (
        "adjustText not installed, falling back to default matplotlib label "
        "placement. Use pip install adjustText."
    )
    with (
        patch.dict("sys.modules", {"adjustText": None}),
        pytest.raises(ImportError, match=err_msg),
    ):
        annotate_bars(ax, adjust_test_pos=True)


@pytest.mark.parametrize(
    "trace_kwargs",
    [None, {}, {"name": "foo", "line_color": "red"}],
)
def test_add_ecdf_line(
    plotly_scatter: go.Figure,
    trace_kwargs: dict[str, str] | None,
) -> None:
    fig = add_ecdf_line(plotly_scatter, trace_kwargs=trace_kwargs)
    assert isinstance(fig, go.Figure)

    trace_kwargs = trace_kwargs or {}

    ecdf = fig.data[-1]  # retrieve ecdf line
    expected_name = trace_kwargs.get("name", "Cumulative")
    expected_color = trace_kwargs.get("line_color", "gray")
    assert ecdf.name == expected_name
    assert ecdf.line.color == expected_color
    assert ecdf.yaxis == "y2"
    assert fig.layout.yaxis2.range == (0, 1)
    assert fig.layout.yaxis2.title.text == expected_name
    assert fig.layout.yaxis2.color == expected_color


def test_add_ecdf_line_raises() -> None:
    for fig in (None, "foo", 42.0):
        with pytest.raises(
            TypeError,
            match=f"{fig=} must be instance of plotly.graph_objs._figure.Figure",
        ):
            add_ecdf_line(fig)


def test_with_marginal_hist() -> None:
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6])
    ax_main = with_marginal_hist([1, 2, 3], [4, 5, 6], fig=fig)
    assert isinstance(ax_main, plt.Axes)
    assert len(fig.axes) == 4

    gs = fig.add_gridspec(2, 2)
    ax_main = with_marginal_hist([1, 2, 3], [4, 5, 6], cell=gs[1, 0])
    assert isinstance(ax_main, plt.Axes)
    assert len(fig.axes) == 7


@pytest.mark.parametrize("annotate_params", [True, False, {"color": "green"}])
def test_add_best_fit_line(
    plotly_scatter: go.Figure,
    matplotlib_scatter: plt.Figure,
    annotate_params: bool | dict[str, Any],
) -> None:
    fig_plotly = add_best_fit_line(plotly_scatter, annotate_params=annotate_params)
    assert isinstance(fig_plotly, go.Figure)
    assert fig_plotly.layout.shapes[-1].type == "line"

    if annotate_params:
        assert fig_plotly.layout.annotations[-1].text.startswith("LS fit: ")
    else:
        assert len(fig_plotly.layout.annotations) == 0

    fig_mpl = add_best_fit_line(matplotlib_scatter, annotate_params=annotate_params)
    assert isinstance(fig_mpl, plt.Figure)
    with pytest.raises(IndexError):
        fig_mpl.axes[1]
    ax = fig_mpl.axes[0]
    assert ax.lines[-1].get_linestyle() == "--"

    anno = next(  # TODO figure out why this always gives None
        (child for child in ax.get_children() if isinstance(child, Annotation)),
        None,
    )

    # if annotate_params:
    #     assert anno.get_text().startswith("LS fit: ")
    # else:
    assert anno is None


def test_add_best_fit_line_invalid_fig() -> None:
    with pytest.raises(TypeError, match="must be instance of"):
        add_best_fit_line("invalid")


def test_get_fig_xy_range(
    plotly_scatter: go.Figure, matplotlib_scatter: plt.Figure
) -> None:
    for fig in (plotly_scatter, matplotlib_scatter, matplotlib_scatter.axes[0]):
        x_range, y_range = get_fig_xy_range(fig)
        assert isinstance(x_range, tuple)
        assert isinstance(y_range, tuple)
        assert len(x_range) == 2
        assert len(y_range) == 2
        assert x_range[0] < x_range[1]
        assert y_range[0] < y_range[1]
        for val in (*x_range, *y_range):
            assert isinstance(val, float)

    # test invalid input
    # currently suboptimal behavior: fig must be passed as kwarg to trigger helpful
    # error message
    with pytest.raises(
        TypeError, match="Unexpected type for fig: str, must be one of None"
    ):
        get_fig_xy_range(fig="invalid")
