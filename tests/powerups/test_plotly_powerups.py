from __future__ import annotations

from typing import TYPE_CHECKING

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


if TYPE_CHECKING:
    from typing import Any

np_rng = np.random.default_rng(seed=0)


@pytest.mark.parametrize(
    ("trace_kwargs", "expected_name", "expected_color", "expected_dash"),
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
    fig = add_ecdf_line(plotly_scatter, trace_kwargs=trace_kwargs, traces=0)
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

    # Check that legendgroup is set, ensuring ECDF line toggles with its trace when
    # clicking legend handle
    assert ecdf_trace.legendgroup == "0"


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
    assert fig.layout.yaxis2.title.text == "Cumulative"
    # Verify ecdf trace data
    assert len(ecdf_trace.x) == len(np.concatenate([y1, y2]))
    assert ecdf_trace.y[-1] == 1.0  # Last ECDF value should be 1.0


def test_add_ecdf_line_faceted() -> None:
    fig = make_subplots(rows=2, cols=2)
    for row in range(1, 3):
        for col in range(1, 3):
            fig.add_scatter(
                x=[1, 2, 3], y=[4, 5, 6], name=f"Trace {row}{col}", row=row, col=col
            )

    n_orig_traces = len(fig.data)
    assert n_orig_traces == 4

    # Use default behavior (all traces)
    fig = add_ecdf_line(fig)

    # We should have exactly 8 traces: 4 original + 4 ECDF traces (1 per original trace)
    assert len(fig.data) == 2 * n_orig_traces
    assert len(fig.data) == 8

    # Check that the original trace names are included in the ECDF trace names
    found_names = set()
    for i, trace in enumerate(fig.data[4:]):
        # Find traces with "Cumulative" name pattern
        assert "Cumulative" in trace.name
        assert f"Trace {(i // 2) + 1}{(i % 2) + 1}" in trace.name
        found_names.add(trace.name)

    # Should have Cumulative trace for each of the original traces
    assert len(found_names) == 4
    assert "Cumulative (Trace 11)" in found_names
    assert "Cumulative (Trace 12)" in found_names
    assert "Cumulative (Trace 21)" in found_names
    assert "Cumulative (Trace 22)" in found_names


def test_add_ecdf_line_all_traces() -> None:
    """Test that add_ecdf_line defaults to adding one ECDF line per trace."""
    fig = go.Figure()
    fig.add_scatter(x=[1, 2, 3], y=[1, 2, 3], name="Trace 1")
    fig.add_scatter(x=[4, 5, 6], y=[4, 5, 6], name="Trace 2")

    # Default behavior should now be all traces
    fig = add_ecdf_line(fig)

    # Should have 2 original traces + 2 ECDF traces
    assert len(fig.data) == 4

    # Check that ECDF trace names include original trace names
    assert fig.data[2].name == "Cumulative (Trace 1)"
    assert fig.data[3].name == "Cumulative (Trace 2)"

    # Check that ECDF traces use secondary y-axis
    assert fig.data[2].yaxis == "y2"
    assert fig.data[3].yaxis == "y2"

    # Check legendgroup matching
    assert fig.data[2].legendgroup == "0"
    assert fig.data[3].legendgroup == "1"


def test_add_ecdf_line_histogram() -> None:
    fig = go.Figure(go.Histogram(x=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]))
    fig = add_ecdf_line(fig)

    assert len(fig.data) == 2
    ecdf_trace = fig.data[-1]
    # Since it's a single trace, no name appending happens
    assert ecdf_trace.name == "Cumulative"
    assert ecdf_trace.yaxis == "y2"
    assert ecdf_trace.legendgroup == "0"

    # Check y values are in ascending order from 0 to 1
    assert ecdf_trace.y[0] > 0
    assert ecdf_trace.y[-1] == 1.0
    assert all(
        ecdf_trace.y[i] <= ecdf_trace.y[i + 1] for i in range(len(ecdf_trace.y) - 1)
    )


def test_add_ecdf_line_bar() -> None:
    fig = go.Figure(go.Bar(x=[1, 2, 3, 4], y=[1, 2, 3, 4]))
    fig = add_ecdf_line(fig)

    assert len(fig.data) == 2
    ecdf_trace = fig.data[-1]
    # Since it's a single trace, no name appending happens
    assert ecdf_trace.name == "Cumulative"
    assert ecdf_trace.yaxis == "y2"
    assert ecdf_trace.legendgroup == "0"

    # Check y values start at 0.25 (1/4) and end at 1.0
    assert ecdf_trace.y[0] == 0.1
    assert ecdf_trace.y[-1] == 1.0
    assert len(ecdf_trace.y) == 10


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
        TypeError, match=f"Cannot auto-determine x-values for ECDF from {qual_name}"
    ):
        add_ecdf_line(fig_violin)

    # check ValueError disappears when passing x-values explicitly
    fig_with_ecdf = add_ecdf_line(fig_violin, values=[1, 2, 3])
    assert len(fig_with_ecdf.data) == 2
    assert fig_with_ecdf.data[1].name == "Cumulative"


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
    assert buttons[0].label == "Linear Y"
    assert buttons[1].label == "Log Y"
    assert buttons[0].method == "relayout"
    assert buttons[1].method == "relayout"

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
    assert buttons[0].label == "Linear X"
    assert buttons[1].label == "Log X"
    assert buttons[0].method == "relayout"
    assert buttons[1].method == "relayout"

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
    assert buttons[0].label == "Show Grid"
    assert buttons[1].label == "Hide Grid"
    assert buttons[0].method == "relayout"
    assert buttons[1].method == "relayout"

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
        assert button.label == colorscale
        assert button.method == "restyle"


def test_select_marker_mode(plotly_scatter: go.Figure) -> None:
    fig = plotly_scatter
    assert isinstance(select_marker_mode, dict)
    assert fig.layout.updatemenus == ()
    fig.layout.updatemenus = [select_marker_mode]

    # check that figure now has plot type toggle buttons
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 3
    plot_types = ["markers", "lines", "lines+markers"]
    labels = ["Scatter", "Line", "Line+Markers"]
    for button, plot_type, label in zip(buttons, plot_types, labels, strict=True):
        assert button.args[0]["mode"] == plot_type
        assert button.label == label
        assert button.method == "restyle"

    # simulate clicking each plot type button
    for button in buttons:
        fig.update_traces(button.args[0])
        assert fig.data[0].mode == button.args[0]["mode"]


@pytest.mark.parametrize(
    ("powerup", "expected_buttons", "expected_type"),
    [
        (toggle_log_linear_x_axis, 2, "buttons"),
        (toggle_grid, 2, "buttons"),
        (select_colorscale, 4, "buttons"),
        (select_marker_mode, 3, "buttons"),
    ],
)
def test_powerup_structure(
    powerup: dict[str, Any], expected_buttons: int, expected_type: str
) -> None:
    assert isinstance(powerup, dict)
    assert powerup["type"] == expected_type
    assert isinstance(powerup["buttons"], list)
    assert len(powerup["buttons"]) == expected_buttons
    for button in powerup["buttons"]:
        assert isinstance(button, dict)
        assert "args" in button
        assert isinstance(button["args"], list)
        assert len(button["args"]) > 0
        assert "label" in button
        assert isinstance(button["label"], str)
        assert "method" in button
        assert button["method"] in ("restyle", "relayout", "update")


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
        muni_i: Updatemenu = fig.layout.updatemenus[idx]
        assert muni_i == Updatemenu(powerup), f"{idx=}"
        assert isinstance(muni_i, Updatemenu)
        assert muni_i.type == powerup["type"]
        assert len(muni_i.buttons) == len(powerup["buttons"])  # type: ignore[arg-type]


def test_add_ecdf_line_color_matching() -> None:
    """Test that ECDF line colors match their corresponding trace colors.

    Covers explicit colors, default colorway, and mixed cases.
    """
    # --- Case 1: Explicit Colors ---
    fig_explicit = go.Figure()
    explicit_colors = ("red", "blue", "green", "purple")
    for idx, color in enumerate(explicit_colors):
        fig_explicit.add_bar(
            x=[idx, idx + 1, idx + 2],
            y=[3, 5, 2],
            name=f"Trace {idx}",
            marker=dict(color=color),
        )

    # Add ECDF lines one by one (mimics adding to specific traces)
    for idx in range(len(fig_explicit.data)):
        fig_explicit = add_ecdf_line(fig_explicit, traces=idx)

    assert len(fig_explicit.data) == 2 * len(explicit_colors)

    # Check that ECDF lines have matching explicit colors
    dev_fig_explicit = fig_explicit.full_figure_for_development(warn=False)
    for idx, color in enumerate(explicit_colors):
        ecdf_trace_dev = dev_fig_explicit.data[idx + len(explicit_colors)]
        assert ecdf_trace_dev.line.color == color
        assert fig_explicit.data[idx + len(explicit_colors)].legendgroup == str(idx)

    # --- Case 2: Default Colorway ---
    fig_colorway = go.Figure()
    n_traces_colorway = 3
    for idx in range(n_traces_colorway):
        # No explicit color set, should use default colorway
        fig_colorway.add_histogram(x=np_rng.normal(idx * 5, 1, 100), name=f"Hist {idx}")

    # Get the default colorway for comparison
    default_colorway = px.colors.qualitative.Plotly

    # Add ECDF lines using the default (all traces)
    fig_colorway = add_ecdf_line(fig_colorway)

    assert len(fig_colorway.data) == 2 * n_traces_colorway

    dev_fig_colorway = fig_colorway.full_figure_for_development(warn=False)
    for idx in range(n_traces_colorway):
        expected_color = default_colorway[idx % len(default_colorway)]
        ecdf_trace_dev = dev_fig_colorway.data[idx + n_traces_colorway]
        original_trace_dev = dev_fig_colorway.data[idx]

        # Check original trace color matches colorway
        assert original_trace_dev.marker.color.casefold() == expected_color.casefold()
        # Check ECDF line color matches original trace color
        assert ecdf_trace_dev.line.color.casefold() == expected_color.casefold()
        assert fig_colorway.data[idx + n_traces_colorway].legendgroup == str(idx)
        assert f"Cumulative (Hist {idx})" == fig_colorway.data[idx + 3].name

    # --- Case 3: Mixed Explicit and Colorway ---
    fig_mixed = go.Figure()
    mixed_colors = ("orange", None, "cyan", None)  # None should use colorway
    expected_final_colors = [
        "orange",
        default_colorway[1],  # First None uses 2nd colorway color
        "cyan",
        default_colorway[3],  # Second None uses 4th colorway color
    ]

    for idx, mixed_color in enumerate(mixed_colors):
        marker_dict = dict(color=mixed_color) if mixed_color else {}
        fig_mixed.add_scatter(
            x=[idx, idx + 1], y=[1, 2], name=f"Mixed {idx}", marker=marker_dict
        )

    # Add ECDF lines using the default (all traces)
    fig_mixed = add_ecdf_line(fig_mixed)

    assert len(fig_mixed.data) == 2 * len(mixed_colors)

    dev_fig_mixed = fig_mixed.full_figure_for_development(warn=False)
    for idx, expected_color in enumerate(expected_final_colors):
        ecdf_trace_dev = dev_fig_mixed.data[idx + len(mixed_colors)]
        assert ecdf_trace_dev.line.color.casefold() == expected_color.casefold()
        assert fig_mixed.data[idx + len(mixed_colors)].legendgroup == str(idx)
        assert (
            f"Cumulative (Mixed {idx})" == fig_mixed.data[idx + len(mixed_colors)].name
        )


def test_add_ecdf_line_annotation_positioning() -> None:
    """Test that multiple ECDF lines have vertically offset annotations."""
    # Create a figure with multiple traces
    fig = go.Figure()
    fig.add_scatter(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 4, 5], name="Trace 1")
    fig.add_scatter(x=[1, 2, 3, 4, 5], y=[5, 4, 3, 2, 1], name="Trace 2")

    # Add ECDF line with text annotation
    trace_kwargs = {"text": "Annotation 1"}
    fig = add_ecdf_line(fig, traces=0, trace_kwargs=trace_kwargs)

    # Add second ECDF line with text annotation
    trace_kwargs = {"text": "Annotation 2"}
    fig = add_ecdf_line(fig, traces=1, trace_kwargs=trace_kwargs)

    # Check that we have the right number of traces
    assert len(fig.data) == 4  # 2 original traces + 2 ECDF lines

    # Check ECDF trace properties
    assert fig.data[2].name == "Cumulative"
    assert fig.data[3].name == "Cumulative"
    assert fig.data[2].legendgroup == "0"
    assert fig.data[3].legendgroup == "1"
    assert fig.data[2].text == "Annotation 1"
    assert fig.data[3].text == "Annotation 2"

    # For the last test, create a new trace and add an ECDF line for it
    fig.add_scatter(x=[1, 2, 3, 4, 5], y=[3, 3, 3, 3, 3], name="Trace 3")

    # Now add a third ECDF line
    fig = add_ecdf_line(fig, traces=4, trace_kwargs={"text": "Annotation 3"})

    # Check that the operation was successful
    assert len(fig.data) == 6  # 3 original traces + 3 ECDF lines
    assert fig.data[5].name == "Cumulative"
    assert fig.data[5].legendgroup == "4"  # Should match the index of the trace (4)
    assert fig.data[5].text == "Annotation 3"
