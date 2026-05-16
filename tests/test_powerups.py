from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pytest
from plotly.graph_objs.layout import Updatemenu
from plotly.graph_objs.layout.annotation import Font
from plotly.subplots import make_subplots

from pymatviz import powerups


if TYPE_CHECKING:
    from typing import Any

np_rng = np.random.default_rng(seed=0)


def test_powerups_custom_metrics_and_font_preservation() -> None:
    metrics_fig = powerups.annotate_metrics(
        [1, 2, 3], [1, 2, 3], fig=go.Figure(), metrics={"Custom": 1.23}
    )
    assert metrics_fig.layout.annotations[0].text == "Custom = 1.23<br>"

    fit_fig = powerups.add_best_fit_line(
        go.Figure(),
        xs=[1, 2, 3],
        ys=[1, 2, 3],
        annotate_params={"color": "red", "font": Font(family="Arial", size=13)},
    )
    anno_font = fit_fig.layout.annotations[0].font
    assert anno_font.color == "red"
    assert anno_font.family == "Arial"
    assert anno_font.size == 13


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
    fig = powerups.add_ecdf_line(plotly_scatter, trace_kwargs=trace_kwargs, traces=0)
    assert isinstance(fig, go.Figure)

    ecdf_trace = fig.data[-1]
    assert ecdf_trace.name == expected_name
    dev_fig = fig.full_figure_for_development(warn=False)
    assert dev_fig.data[-1].line.color.casefold() == expected_color.casefold()
    assert ecdf_trace.line.dash == expected_dash
    assert ecdf_trace.yaxis == "y2"
    assert fig.layout.yaxis2.range == (0, 1)
    assert fig.layout.yaxis2.title.text == expected_name
    assert dev_fig.layout.yaxis2.color in (expected_color, "#444")

    assert ecdf_trace.legendgroup == "0"


def test_add_ecdf_line_stacked() -> None:
    x = ["A", "B", "C"]
    y1 = [1, 2, 3]
    y2 = [2, 3, 4]

    fig = go.Figure()
    fig.add_bar(x=x, y=y1, name="Group 1")
    fig.add_bar(x=x, y=y2, name="Group 2")
    fig.update_layout(barmode="stack")

    fig = powerups.add_ecdf_line(fig, values=np.concatenate([y1, y2]))

    assert len(fig.data) == 3
    ecdf_trace = fig.data[-1]
    assert ecdf_trace.name == "Cumulative"
    assert ecdf_trace.yaxis == "y2"
    assert fig.layout.yaxis2.range == (0, 1)
    assert fig.layout.yaxis2.title.text == "Cumulative"
    assert len(ecdf_trace.x) == len(y1) + len(y2)
    assert ecdf_trace.y[-1] == 1.0


def test_add_ecdf_line_faceted() -> None:
    fig = make_subplots(rows=2, cols=2)
    for row in range(1, 3):
        for col in range(1, 3):
            fig.add_scatter(
                x=[1, 2, 3], y=[4, 5, 6], name=f"Trace {row}{col}", row=row, col=col
            )

    n_orig_traces = len(fig.data)
    assert n_orig_traces == 4

    fig = powerups.add_ecdf_line(fig)

    assert len(fig.data) == 2 * n_orig_traces
    assert [trace.name for trace in fig.data[4:]] == [
        "Cumulative (Trace 11)",
        "Cumulative (Trace 12)",
        "Cumulative (Trace 21)",
        "Cumulative (Trace 22)",
    ]


def test_add_ecdf_line_all_traces() -> None:
    fig = go.Figure()
    fig.add_scatter(x=[1, 2, 3], y=[1, 2, 3], name="Trace 1")
    fig.add_scatter(x=[4, 5, 6], y=[4, 5, 6], name="Trace 2")

    fig = powerups.add_ecdf_line(fig)

    assert len(fig.data) == 4
    assert [(trace.name, trace.yaxis, trace.legendgroup) for trace in fig.data[2:]] == [
        ("Cumulative (Trace 1)", "y2", "0"),
        ("Cumulative (Trace 2)", "y2", "1"),
    ]


@pytest.mark.parametrize(
    ("trace", "expected_first_y", "expected_len"),
    [
        (go.Histogram(x=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]), 0.1, 10),
        (go.Bar(x=[1, 2, 3, 4], y=[1, 2, 3, 4]), 0.1, 10),
    ],
)
def test_add_ecdf_line_single_trace(
    trace: go.Histogram | go.Bar, expected_first_y: float, expected_len: int
) -> None:
    fig = go.Figure(trace)
    fig = powerups.add_ecdf_line(fig)

    assert len(fig.data) == 2
    ecdf_trace = fig.data[-1]
    assert ecdf_trace.name == "Cumulative"
    assert ecdf_trace.yaxis == "y2"
    assert ecdf_trace.legendgroup == "0"
    assert ecdf_trace.y[0] == expected_first_y
    assert ecdf_trace.y[-1] == 1.0
    assert len(ecdf_trace.y) == expected_len
    assert list(ecdf_trace.y) == sorted(ecdf_trace.y)


def test_add_ecdf_line_raises() -> None:
    invalid_figs: tuple[Any, ...] = (None, "foo", 42.0)
    for fig in invalid_figs:
        with pytest.raises(
            TypeError,
            match=f"{fig=} must be instance of go.Figure",
        ):
            powerups.add_ecdf_line(fig)

    fig_violin = px.violin(x=[1, 2, 3], y=[4, 5, 6])
    violin_trace = type(fig_violin.data[0])
    qual_name = f"{violin_trace.__module__}.{violin_trace.__qualname__}"
    with pytest.raises(
        TypeError, match=f"Cannot auto-determine x-values for ECDF from {qual_name}"
    ):
        powerups.add_ecdf_line(fig_violin)

    fig_with_ecdf = powerups.add_ecdf_line(fig_violin, values=[1, 2, 3])
    assert len(fig_with_ecdf.data) == 2
    assert fig_with_ecdf.data[1].name == "Cumulative"


@pytest.mark.parametrize(
    ("powerup", "button_specs"),
    [
        (
            powerups.toggle_log_linear_y_axis,
            [
                ("Linear Y", "relayout", {"yaxis.type": "linear"}),
                ("Log Y", "relayout", {"yaxis.type": "log"}),
            ],
        ),
        (
            powerups.toggle_log_linear_x_axis,
            [
                ("Linear X", "relayout", {"xaxis.type": "linear"}),
                ("Log X", "relayout", {"xaxis.type": "log"}),
            ],
        ),
        (
            powerups.toggle_grid,
            [
                (
                    "Show Grid",
                    "relayout",
                    {"xaxis.showgrid": True, "yaxis.showgrid": True},
                ),
                (
                    "Hide Grid",
                    "relayout",
                    {"xaxis.showgrid": False, "yaxis.showgrid": False},
                ),
            ],
        ),
    ],
)
def test_relayout_powerups(
    plotly_scatter: go.Figure,
    powerup: dict[str, Any],
    button_specs: list[tuple[str, str, dict[str, Any]]],
) -> None:
    assert isinstance(powerup, dict)
    assert plotly_scatter.layout.updatemenus == ()
    for _label, _method, args in button_specs:
        for key in args:
            parent, attr = key.split(".")
            assert getattr(getattr(plotly_scatter.layout, parent), attr) is None
    plotly_scatter.layout.updatemenus = [powerup]

    menu = plotly_scatter.layout.updatemenus[0]
    assert isinstance(menu, Updatemenu)
    buttons = menu.buttons
    assert len(buttons) == len(button_specs)
    for button, (label, method, args) in zip(buttons, button_specs, strict=True):
        assert (button.label, button.method, button.args[0]) == (label, method, args)
        plotly_scatter.update_layout(args)
        for key, val in args.items():
            parent, attr = key.split(".")
            assert getattr(getattr(plotly_scatter.layout, parent), attr) == val


@pytest.mark.parametrize(
    ("fig", "powerup", "button_specs", "apply"),
    [
        (
            go.Figure(go.Heatmap(z=[[1, 2], [3, 4]])),
            powerups.select_colorscale,
            [
                (scale, "restyle", {"colorscale": scale})
                for scale in ["Viridis", "Plasma", "Inferno", "Magma"]
            ],
            False,
        ),
        (
            go.Figure(go.Scatter(x=[1, 2], y=[3, 4])),
            powerups.select_marker_mode,
            [
                ("Scatter", "restyle", {"type": "scatter", "mode": "markers"}),
                ("Line", "restyle", {"type": "scatter", "mode": "lines"}),
                (
                    "Line+Markers",
                    "restyle",
                    {"type": "scatter", "mode": "lines+markers"},
                ),
            ],
            True,
        ),
    ],
)
def test_restyle_powerups(
    fig: go.Figure,
    powerup: dict[str, Any],
    button_specs: list[tuple[str, str, dict[str, Any]]],
    apply: bool,
) -> None:
    assert isinstance(powerup, dict)
    assert fig.layout.updatemenus == ()
    fig.layout.updatemenus = [powerup]

    menu = fig.layout.updatemenus[0]
    assert isinstance(menu, Updatemenu)
    buttons = menu.buttons
    assert len(buttons) == len(button_specs)
    for button, (label, method, args) in zip(buttons, button_specs, strict=True):
        assert (button.label, button.method, button.args[0]) == (label, method, args)
        if apply:
            fig.update_traces(args)
            assert fig.data[0].mode == args["mode"]


@pytest.mark.parametrize(
    ("powerup", "expected_buttons", "expected_type"),
    [
        (powerups.toggle_log_linear_x_axis, 2, "buttons"),
        (powerups.toggle_grid, 2, "buttons"),
        (powerups.select_colorscale, 4, "buttons"),
        (powerups.select_marker_mode, 3, "buttons"),
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

    powerup_list = [
        powerups.toggle_log_linear_x_axis,
        powerups.toggle_grid,
        powerups.select_colorscale,
        powerups.select_marker_mode,
    ]
    fig.layout.updatemenus = powerup_list

    assert fig.layout.updatemenus == tuple(
        Updatemenu(powerup) for powerup in powerup_list
    )


def test_add_ecdf_line_color_matching() -> None:
    fig_explicit = go.Figure()
    explicit_colors = ("red", "blue", "green", "purple")
    for idx, color in enumerate(explicit_colors):
        fig_explicit.add_bar(
            x=[idx, idx + 1, idx + 2],
            y=[3, 5, 2],
            name=f"Trace {idx}",
            marker=dict(color=color),
        )

    for idx in range(len(explicit_colors)):
        fig_explicit = powerups.add_ecdf_line(fig_explicit, traces=idx)

    assert len(fig_explicit.data) == 2 * len(explicit_colors)
    dev_fig_explicit = fig_explicit.full_figure_for_development(warn=False)
    for idx, color in enumerate(explicit_colors):
        ecdf_trace_dev = dev_fig_explicit.data[idx + len(explicit_colors)]
        assert ecdf_trace_dev.line.color == color
        assert fig_explicit.data[idx + len(explicit_colors)].legendgroup == str(idx)

    fig_colorway = go.Figure()
    n_traces_colorway = 3
    for idx in range(n_traces_colorway):
        fig_colorway.add_histogram(x=np_rng.normal(idx * 5, 1, 100), name=f"Hist {idx}")

    default_colorway = px.colors.qualitative.Plotly
    fig_colorway = powerups.add_ecdf_line(fig_colorway)

    assert len(fig_colorway.data) == 2 * n_traces_colorway
    dev_fig_colorway = fig_colorway.full_figure_for_development(warn=False)
    for idx in range(n_traces_colorway):
        expected_color = default_colorway[idx % len(default_colorway)]
        ecdf_trace_dev = dev_fig_colorway.data[idx + n_traces_colorway]
        original_trace_dev = dev_fig_colorway.data[idx]
        assert original_trace_dev.marker.color.casefold() == expected_color.casefold()
        assert ecdf_trace_dev.line.color.casefold() == expected_color.casefold()
        assert fig_colorway.data[idx + n_traces_colorway].legendgroup == str(idx)
        assert f"Cumulative (Hist {idx})" == fig_colorway.data[idx + 3].name

    fig_mixed = go.Figure()
    mixed_colors = ("orange", None, "cyan", None)
    for idx, mixed_color in enumerate(mixed_colors):
        marker_dict = dict(color=mixed_color) if mixed_color else {}
        fig_mixed.add_scatter(
            x=[idx, idx + 1], y=[1, 2], name=f"Mixed {idx}", marker=marker_dict
        )
    fig_mixed = powerups.add_ecdf_line(fig_mixed)

    assert len(fig_mixed.data) == 2 * len(mixed_colors)
    dev_fig_mixed = fig_mixed.full_figure_for_development(warn=False)
    for idx, expected_color in enumerate(
        ["orange", default_colorway[1], "cyan", default_colorway[3]]
    ):
        ecdf_trace_dev = dev_fig_mixed.data[idx + len(mixed_colors)]
        assert ecdf_trace_dev.line.color.casefold() == expected_color.casefold()
        assert fig_mixed.data[idx + len(mixed_colors)].legendgroup == str(idx)
        assert (
            f"Cumulative (Mixed {idx})" == fig_mixed.data[idx + len(mixed_colors)].name
        )


def test_add_ecdf_line_annotation_positioning() -> None:
    fig = go.Figure()
    for y_vals, name in [
        ([1, 2, 3, 4, 5], "Trace 1"),
        ([5, 4, 3, 2, 1], "Trace 2"),
    ]:
        fig.add_scatter(x=[1, 2, 3, 4, 5], y=y_vals, name=name)
    for idx, text in enumerate(["Annotation 1", "Annotation 2"]):
        fig = powerups.add_ecdf_line(fig, traces=idx, trace_kwargs={"text": text})

    assert len(fig.data) == 4
    for trace, legendgroup, text in zip(
        fig.data[2:], ["0", "1"], ["Annotation 1", "Annotation 2"], strict=True
    ):
        assert trace.name == "Cumulative"
        assert trace.legendgroup == legendgroup
        assert trace.text == text

    fig.add_scatter(x=[1, 2, 3, 4, 5], y=[3, 3, 3, 3, 3], name="Trace 3")
    fig = powerups.add_ecdf_line(fig, traces=4, trace_kwargs={"text": "Annotation 3"})

    assert len(fig.data) == 6
    assert fig.data[5].name == "Cumulative"
    assert fig.data[5].legendgroup == "4"
    assert fig.data[5].text == "Annotation 3"
