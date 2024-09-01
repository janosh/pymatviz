from __future__ import annotations

from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pytest
from plotly.graph_objs.layout import Updatemenu
from plotly.subplots import make_subplots

from pymatviz.powerups.plotly import (
    add_ecdf_line,
    select_colorscale,
    select_marker_mode,
    toggle_grid,
    toggle_log_linear_x_axis,
    toggle_log_linear_y_axis,
)


@pytest.mark.parametrize(
    "trace_kwargs, expected_name, expected_color, expected_dash",
    [
        (None, "Cumulative", "#636efa", "solid"),
        ({}, "Cumulative", "#636efa", "solid"),
        ({"name": "foo", "line": {"color": "red"}}, "foo", "red", "solid"),
        ({"line": {"dash": "dash"}}, "Cumulative", "#636efa", "dash"),
    ],
)
def test_add_ecdf_line(
    plotly_scatter: go.Figure,
    trace_kwargs: dict[str, Any] | None,
    expected_name: str,
    expected_color: str,
    expected_dash: str,
) -> None:
    fig = add_ecdf_line(plotly_scatter, trace_kwargs=trace_kwargs)
    assert isinstance(fig, go.Figure)

    ecdf_trace = fig.data[-1]
    assert ecdf_trace.name == expected_name
    dev_fig = fig.full_figure_for_development(warn=False)
    assert dev_fig.data[-1].line.color == expected_color
    assert ecdf_trace.line.dash == expected_dash
    assert ecdf_trace.yaxis == "y2"
    assert fig.layout.yaxis2.range == (0, 1)
    assert fig.layout.yaxis2.title.text == expected_name
    assert dev_fig.layout.yaxis2.color in (expected_color, "#444")

    assert ecdf_trace.legendgroup == fig.data[0].name


def test_add_ecdf_line_stacked() -> None:
    x = ["A", "B", "C"]
    y1 = [1, 2, 3]
    y2 = [2, 3, 4]

    fig = go.Figure()
    fig.add_bar(x=x, y=y1, name="Group 1")
    fig.add_bar(x=x, y=y2, name="Group 2")
    fig.update_layout(barmode="stack")

    fig = add_ecdf_line(fig, values=np.concatenate([y1, y2]))

    assert len(fig.data) == 3
    ecdf_trace = fig.data[-1]
    assert ecdf_trace.name == "Cumulative"
    assert ecdf_trace.yaxis == "y2"
    assert fig.layout.yaxis2.range == (0, 1)


def test_add_ecdf_line_faceted() -> None:
    fig = make_subplots(rows=2, cols=2)
    for row in range(1, 3):
        for col in range(1, 3):
            fig.add_scatter(
                x=[1, 2, 3], y=[4, 5, 6], name=f"Trace {row}{col}", row=row, col=col
            )

    fig = add_ecdf_line(fig)

    assert len(fig.data) == 8  # 4 original traces + 1 ECDF trace
    ecdf_trace = fig.data[-1]
    assert ecdf_trace.name == "Cumulative"
    assert ecdf_trace.yaxis == "y2"


def test_add_ecdf_line_histogram() -> None:
    fig = go.Figure(go.Histogram(x=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]))
    fig = add_ecdf_line(fig)

    assert len(fig.data) == 2
    ecdf_trace = fig.data[-1]
    assert ecdf_trace.name == "Cumulative"
    assert ecdf_trace.yaxis == "y2"


def test_add_ecdf_line_bar() -> None:
    fig = go.Figure(go.Bar(x=[1, 2, 3, 4], y=[1, 2, 3, 4]))
    fig = add_ecdf_line(fig)

    assert len(fig.data) == 2
    ecdf_trace = fig.data[-1]
    assert ecdf_trace.name == "Cumulative"
    assert ecdf_trace.yaxis == "y2"


def test_add_ecdf_line_raises() -> None:
    # check TypeError when passing invalid fig
    for fig in (None, "foo", 42.0):
        with pytest.raises(
            TypeError,
            match=f"{fig=} must be instance of plotly.graph_objs._figure.Figure",
        ):
            add_ecdf_line(fig)

    # check ValueError when x-values cannot be auto-determined
    fig_violin = px.violin(x=[1, 2, 3], y=[4, 5, 6])
    violin_trace = type(fig_violin.data[0])
    qual_name = f"{violin_trace.__module__}.{violin_trace.__qualname__}"
    with pytest.raises(
        ValueError, match=f"Cannot auto-determine x-values for ECDF from {qual_name}"
    ):
        add_ecdf_line(fig_violin)

    # check ValueError disappears when passing x-values explicitly
    add_ecdf_line(fig_violin, values=[1, 2, 3])


def test_toggle_log_linear_y_axis(plotly_scatter: go.Figure) -> None:
    fig = plotly_scatter
    assert isinstance(toggle_log_linear_y_axis, dict)
    assert fig.layout.updatemenus == ()
    fig.layout.updatemenus = [toggle_log_linear_y_axis]

    # check that figure now has "Log Y"/"Linear Y" toggle buttons
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 2
    assert fig.layout.yaxis.type is None
    assert buttons[0].args[0]["yaxis.type"] == "linear"
    assert buttons[1].args[0]["yaxis.type"] == "log"

    # simulate clicking "Log Y" button
    fig.update_layout(buttons[0].args[0])
    assert fig.layout.yaxis.type == "linear"
    fig.update_layout(buttons[1].args[0])
    assert fig.layout.yaxis.type == "log"


def test_toggle_log_linear_x_axis(plotly_scatter: go.Figure) -> None:
    fig = plotly_scatter
    assert isinstance(toggle_log_linear_x_axis, dict)
    assert fig.layout.updatemenus == ()
    fig.layout.updatemenus = [toggle_log_linear_x_axis]

    # check that figure now has "Log X"/"Linear X" toggle buttons
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 2
    assert fig.layout.xaxis.type is None
    assert buttons[0].args[0]["xaxis.type"] == "linear"
    assert buttons[1].args[0]["xaxis.type"] == "log"

    # simulate clicking buttons
    fig.update_layout(buttons[0].args[0])
    assert fig.layout.xaxis.type == "linear"
    fig.update_layout(buttons[1].args[0])
    assert fig.layout.xaxis.type == "log"


def test_toggle_grid(plotly_scatter: go.Figure) -> None:
    fig = plotly_scatter
    assert isinstance(toggle_grid, dict)
    assert fig.layout.updatemenus == ()
    fig.layout.updatemenus = [toggle_grid]

    # check that figure now has "Show Grid"/"Hide Grid" toggle buttons
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 2
    assert fig.layout.xaxis.showgrid is None
    assert fig.layout.yaxis.showgrid is None
    assert buttons[0].args[0]["xaxis.showgrid"] is True
    assert buttons[0].args[0]["yaxis.showgrid"] is True
    assert buttons[1].args[0]["xaxis.showgrid"] is False
    assert buttons[1].args[0]["yaxis.showgrid"] is False

    # simulate clicking buttons
    fig.update_layout(buttons[0].args[0])
    assert fig.layout.xaxis.showgrid is True
    assert fig.layout.yaxis.showgrid is True
    fig.update_layout(buttons[1].args[0])
    assert fig.layout.xaxis.showgrid is False
    assert fig.layout.yaxis.showgrid is False


def test_select_colorscale() -> None:
    # make a dummy heatmap
    fig = go.Figure(go.Heatmap(z=[[1, 2], [3, 4]]))
    assert isinstance(select_colorscale, dict)
    assert fig.layout.updatemenus == ()
    fig.layout.updatemenus = [select_colorscale]

    # check that figure now has colorscale toggle buttons
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 4
    colorscales = ["Viridis", "Plasma", "Inferno", "Magma"]
    for button, colorscale in zip(buttons, colorscales, strict=True):
        assert button.args[0]["colorscale"] == colorscale


def test_select_marker_mode(plotly_scatter: go.Figure) -> None:
    fig = plotly_scatter
    assert isinstance(select_marker_mode, dict)
    assert fig.layout.updatemenus == ()
    fig.layout.updatemenus = [select_marker_mode]

    # check that figure now has plot type toggle buttons
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 3
    plot_types = ["markers", "lines", "lines+markers"]
    for button, plot_type in zip(buttons, plot_types, strict=True):
        assert button.args[0]["mode"] == plot_type

    # simulate clicking each plot type button
    for button in buttons:
        fig.update_traces(button.args[0])
        assert fig.data[0].mode == button.args[0]["mode"]


@pytest.mark.parametrize(
    "powerup, expected_buttons",
    [
        (toggle_log_linear_x_axis, 2),
        (toggle_grid, 2),
        (select_colorscale, 4),
        (select_marker_mode, 3),
    ],
)
def test_powerup_structure(powerup: dict[str, Any], expected_buttons: int) -> None:
    assert isinstance(powerup, dict)
    assert powerup["type"] == "buttons"
    assert isinstance(powerup["buttons"], list)
    assert len(powerup["buttons"]) == expected_buttons
    for button in powerup["buttons"]:
        assert isinstance(button, dict)
        assert "args" in button
        assert "label" in button
        assert "method" in button


def test_multiple_powerups(plotly_scatter: go.Figure) -> None:
    fig = plotly_scatter
    assert fig.layout.updatemenus == ()

    powerups = [
        toggle_log_linear_x_axis,
        toggle_grid,
        select_colorscale,
        select_marker_mode,
    ]
    fig.layout.updatemenus = powerups

    assert len(fig.layout.updatemenus) == len(powerups)
    for idx, powerup in enumerate(powerups):
        assert fig.layout.updatemenus[idx] == Updatemenu(powerup), f"{idx=}"
