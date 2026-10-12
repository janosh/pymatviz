from __future__ import annotations

import re
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
    """Metric dicts map labels to numbers or computed metric names; fonts persist."""
    metrics = {"Custom": 1.23, "Mean abs. error": "MAE", "R2": "r2_score"}
    metrics_fig = powerups.annotate_metrics(
        [1, 2, 3], [1, 2, 4], fig=go.Figure(), metrics=metrics
    )
    assert metrics_fig.layout.annotations[0].text == (
        "Custom = 1.23<br>Mean abs. error = 0.333<br>R<sup>2</sup> = 0.5<br>"
    )
    bad_metrics: Any = {"MAE": None}
    with pytest.raises(TypeError, match="must be numbers or metric names"):
        powerups.annotate_metrics([1], [1], fig=go.Figure(), metrics=bad_metrics)
    with pytest.raises(ValueError, match=re.escape("Unrecognized metrics: {'bogus'}")):
        powerups.annotate_metrics([1], [1], fig=go.Figure(), metrics={"B": "bogus"})

    # faceted figures take one array per subplot, a flat array can't be attributed
    facet_fig = px.scatter(x=[1, 2, 3, 4], y=[1, 2, 4, 3], facet_col=[0, 0, 1, 1])
    powerups.annotate_metrics([[1, 2], [3, 4]], [[1, 2], [4, 3]], fig=facet_fig)
    assert [anno.xref for anno in facet_fig.layout.annotations[-2:]] == [
        "x domain",
        "x2 domain",
    ]
    assert "MAE = 0.0" in facet_fig.layout.annotations[-2].text
    with pytest.raises(ValueError, match="one xs/ys array per subplot"):
        powerups.annotate_metrics([1, 2, 3, 4], [1, 2, 4, 3], fig=facet_fig)

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


@pytest.mark.parametrize("explicit_data", [False, True])
@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("annotation_mode", ["combined", "per_trace"])
def test_enhance_parity_plot_keeps_dict_best_fit_line(
    explicit_data: bool, stats: bool, annotation_mode: powerups.AnnotationMode
) -> None:
    """Direct and trace-derived fits preserve styling with optional metrics."""
    fig = go.Figure()
    fig.add_scatter(x=[1, 2, 3], y=[1, 2, 4])

    powerups.enhance_parity_plot(
        fig,
        xs=[1, 2, 3] if explicit_data else (),
        ys=[1, 2, 4] if explicit_data else (),
        identity_line=False,
        best_fit_line={"color": "red"},
        stats=stats,
        annotation_mode=annotation_mode,
    )

    assert fig.layout.shapes[-1].line.color == "red"


@pytest.mark.parametrize("axis_type", ["linear", "log"])
def test_add_identity_line(axis_type: str) -> None:
    """Identity line spans the overlap of the padded x and y ranges, retained limits
    use axis units.
    """
    fig = go.Figure().add_scatter(x=[1, 100], y=[10, 1000])
    fig.update_xaxes(type=axis_type).update_yaxes(type=axis_type)
    powerups.add_identity_line(fig, retain_xy_limits=True)

    (x_min, x_max), (y_min, y_max) = (
        ((10**-0.1, 10**2.1), (10**0.9, 10**3.1))
        if axis_type == "log"
        else ((1 - 4.95, 100 + 4.95), (10 - 49.5, 1000 + 49.5))
    )
    shape = fig.layout.shapes[0]
    np.testing.assert_allclose(
        (shape.x0, shape.y0, shape.x1, shape.y1),
        (max(x_min, y_min),) * 2 + (min(x_max, y_max),) * 2,
        rtol=1e-12,
        atol=0,
    )
    to_axis_units = np.log10 if axis_type == "log" else np.asarray
    for axis_range, expected in (
        (fig.layout.xaxis.range, (x_min, x_max)),
        (fig.layout.yaxis.range, (y_min, y_max)),
    ):
        np.testing.assert_allclose(
            axis_range, to_axis_units(expected), rtol=1e-12, atol=1e-15
        )

    # disjoint x and y ranges: y=x would lie outside the plot, so no line
    disjoint = go.Figure().add_scatter(x=[0, 1], y=[10, 11])
    with pytest.warns(UserWarning, match="Skipping identity line"):
        powerups.add_identity_line(disjoint)
    assert disjoint.layout.shapes == ()


@pytest.mark.parametrize(
    ("annotation_mode", "expected_lines", "expected_texts"),
    [
        (
            "per_trace",
            [("red", 1, 3, 3, 7), ("navy", 1, 3, 3, 1)],
            ["LS fit: y = 2x + 1", "LS fit: y = -1x + 4"],
        ),
        ("combined", [("red", 1, 3, 3, 4)], ["LS fit: y = 0.5x + 2.5"]),
        ("none", [("red", 1, 3, 3, 7), ("navy", 1, 3, 3, 1)], []),
    ],
)
def test_add_best_fit_line_from_traces(
    annotation_mode: powerups.AnnotationMode,
    expected_lines: list[tuple[str, float, float, float, float]],
    expected_texts: list[str],
) -> None:
    """Trace-derived fits skip NaNs, honor annotation_mode and forward shape kwargs."""
    fig = go.Figure()
    fig.add_scatter(x=[1, 2, 3, np.nan], y=[3, 5, 7, 0], marker_color="red")
    fig.add_scatter(x=[1, 2, 3], y=[3, 2, 1])
    for _ in range(2):  # repeat calls stack per-trace labels, replace combined ones
        powerups.add_best_fit_line(fig, annotation_mode=annotation_mode, opacity=0.5)

    shapes = fig.layout.shapes[-len(expected_lines) :]
    assert len(fig.layout.shapes) == 2 * len(expected_lines)
    np.testing.assert_allclose(
        [(shape.x0, shape.y0, shape.x1, shape.y1) for shape in shapes],
        [line[1:] for line in expected_lines],
        rtol=0,
        atol=1e-12,
    )
    assert [shape.line.color for shape in shapes] == [
        line[0] for line in expected_lines
    ]
    assert {(shape.xref, shape.yref, shape.opacity) for shape in shapes} == {
        ("x", "y", 0.5)
    }
    n_repeats = 1 if annotation_mode == "combined" else 2
    annos = fig.layout.annotations
    assert [anno.text for anno in annos] == expected_texts * n_repeats
    assert [anno.y for anno in annos] == pytest.approx(
        [0.02 + 0.05 * idx for idx in range(len(annos))]
    )
    assert {(anno.xref, anno.yref) for anno in annos} <= {("x domain", "y domain")}


@pytest.mark.parametrize("annotation_mode", ["per_trace", "combined"])
def test_add_best_fit_line_faceted(annotation_mode: powerups.AnnotationMode) -> None:
    """Faceted figures get fits and stacked labels inside each trace's own subplot."""
    xs = np.tile([1.0, 2.0, 3.0], 4)
    slopes = np.repeat([1.0, 2.0, 3.0, 4.0], 3)
    df_facets = {
        "x": xs,
        "y": slopes * xs + 1,
        "facet": np.repeat(["A", "B"], 6),
        "group": np.tile(np.repeat(["p", "q"], 3), 2),
    }
    fig = px.scatter(df_facets, x="x", y="y", facet_col="facet", color="group")
    powerups.add_best_fit_line(fig, annotation_mode=annotation_mode)

    fit_annos = [anno for anno in fig.layout.annotations if "LS fit" in anno.text]
    if annotation_mode == "combined":
        expected_refs = [("x", "y"), ("x2", "y2")]
        expected_texts = ["LS fit: y = 1.5x + 1", "LS fit: y = 3.5x + 1"]
        expected_ys = [0.02, 0.02]
    else:  # px orders traces by color group, then facet
        expected_refs = [("x", "y"), ("x", "y"), ("x2", "y2"), ("x2", "y2")]
        expected_texts = [f"LS fit: y = {slope}x + 1" for slope in (1, 2, 3, 4)]
        expected_ys = [0.02, 0.07, 0.02, 0.07]
    assert [(shape.xref, shape.yref) for shape in fig.layout.shapes] == expected_refs
    assert [(anno.xref, anno.yref) for anno in fit_annos] == [
        (f"{xref} domain", f"{yref} domain") for xref, yref in expected_refs
    ]
    assert [anno.text for anno in fit_annos] == expected_texts
    assert [anno.y for anno in fit_annos] == pytest.approx(expected_ys)


@pytest.mark.parametrize(
    ("trace_kwargs", "expected_name", "expected_color", "expected_dash"),
    [
        (None, "Cumulative", "#636efa", "solid"),
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
    """Explicit values add one ECDF line, merging line kwargs into the defaults."""
    x = ["A", "B", "C"]
    y1 = [1, 2, 3]
    y2 = [2, 3, 4]

    fig = go.Figure()
    fig.add_bar(x=x, y=y1, name="Group 1")
    fig.add_bar(x=x, y=y2, name="Group 2")
    fig.update_layout(barmode="stack")

    fig = powerups.add_ecdf_line(
        fig,
        values=np.concatenate([y1, y2]),
        trace_kwargs={"line": {"color": "orange", "width": 4}},
    )

    assert len(fig.data) == 3
    ecdf_trace = fig.data[-1]
    assert ecdf_trace.name == "Cumulative"
    assert ecdf_trace.yaxis == "y2"
    line = ecdf_trace.line
    assert (line.color, line.dash, line.width) == ("orange", "solid", 4)
    assert fig.layout.yaxis2.range == (0, 1)
    assert fig.layout.yaxis2.title.text == "Cumulative"
    assert fig.layout.yaxis2.color == "orange"
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


@pytest.mark.parametrize(
    ("trace", "expected_x", "expected_y"),
    [
        (
            go.Histogram(x=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4]),
            [1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
            np.arange(1, 11) / 10,
        ),
        (go.Bar(x=[1, 2, 3, 4], y=[1, 2, 3, 4]), [1, 2, 3, 4], [0.1, 0.3, 0.6, 1]),
        # float bar heights are weights, not repeat counts
        (go.Bar(x=[3, 1, 2], y=[0.5, 1.5, 2.0]), [1, 2, 3], [0.375, 0.875, 1]),
        # bin edges: one more x than y, bars sit at the left edges
        (go.Bar(x=[0, 1, 2, 3], y=[2, 1, 1]), [0, 1, 2], [0.5, 0.75, 1]),
        # horizontal: values on the y-axis, ECDF along a top x2 axis
        (go.Histogram(y=[1, 2, 2, 3]), [1, 2, 2, 3], [0.25, 0.5, 0.75, 1]),
        (
            go.Bar(x=[0.5, 1.5, 2.0], y=[3, 1, 2], orientation="h"),
            [1, 2, 3],
            [0.375, 0.875, 1],
        ),
    ],
    ids=[
        "histogram",
        "bar-int",
        "bar-float",
        "bar-bin-edges",
        "histogram-horizontal",
        "bar-horizontal",
    ],
)
def test_add_ecdf_line_single_trace(
    trace: go.Histogram | go.Bar, expected_x: list[float], expected_y: list[float]
) -> None:
    """ECDFs of histograms use raw samples, of bars use lengths as weights."""
    fig = powerups.add_ecdf_line(go.Figure(trace))

    assert len(fig.data) == 2
    ecdf_trace = fig.data[-1]
    assert ecdf_trace.name == "Cumulative"
    assert ecdf_trace.legendgroup == "0"
    # values_axis holds the data values, prob_axis the cumulative fraction
    horizontal = trace.orientation == "h" or trace.x is None
    values_axis, prob_axis = ("y", "x") if horizontal else ("x", "y")
    assert getattr(ecdf_trace, f"{prob_axis}axis") == f"{prob_axis}2"
    prob_layout_axis = fig.layout[f"{prob_axis}axis2"]
    assert (prob_layout_axis.overlaying, prob_layout_axis.range) == (prob_axis, (0, 1))
    np.testing.assert_array_equal(getattr(ecdf_trace, values_axis), expected_x)
    np.testing.assert_allclose(
        getattr(ecdf_trace, prob_axis), expected_y, rtol=1e-12, atol=0
    )


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
    assert plotly_scatter.layout.updatemenus == ()
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
        ecdf_trace = fig_mixed.data[idx + len(mixed_colors)]
        assert (ecdf_trace.name, ecdf_trace.yaxis, ecdf_trace.legendgroup) == (
            f"Cumulative (Mixed {idx})",
            "y2",
            str(idx),
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

    # text applies to every ECDF trace when selecting several at once
    fig = powerups.add_ecdf_line(fig, traces=[0, 1], trace_kwargs={"text": "Both"})
    assert len(fig.data) == 8
    assert [trace.text for trace in fig.data[6:]] == ["Both", "Both"]
    assert [list(trace.y) for trace in fig.data[6:]] == [[0.2, 0.4, 0.6, 0.8, 1]] * 2
    assert fig.layout.annotations == ()

    with pytest.raises(ValueError, match="offset_size"):
        powerups.add_ecdf_line(fig, traces=[0, 1], trace_kwargs={"offset_size": 0.1})


def test_annotate_metrics_r2_adj_two_points() -> None:
    """R2_adj is undefined for n <= 2 samples, must not raise ZeroDivisionError."""
    fig = powerups.annotate_metrics(
        np.array([1.0, 2.0]), np.array([1.1, 2.2]), fig=go.Figure(), metrics=["R2_adj"]
    )
    assert fig.layout.annotations[-1].text == "R<sup>2</sup><sub>adj</sub> = nan<br>"
