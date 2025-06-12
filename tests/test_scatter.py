from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest

import pymatviz as pmv
from tests.conftest import df_regr, np_rng


if TYPE_CHECKING:
    from typing import Any, Literal

    from tests.conftest import DfOrArrays


X_COL, Y_COL, *_ = df_regr
DF_TIPS = px.data.tips()


@pytest.mark.parametrize("log_density", [True, False, None])
@pytest.mark.parametrize("n_bins", [None, False, 50])
@pytest.mark.parametrize(
    "stats",
    [False, True, dict(prefix="test", font=dict(size=10))],
)
@pytest.mark.parametrize("density", ["kde", "empirical", None])
def test_density_scatter(
    df_or_arrays: DfOrArrays,
    log_density: bool | None,
    n_bins: int | None | bool,
    stats: bool | dict[str, Any],
    density: Literal["kde", "empirical"] | None,
) -> None:
    """Test density_scatter function with Plotly backend."""
    df, x, y = df_or_arrays
    fig = pmv.density_scatter(
        df=df,
        x=x,
        y=y,
        log_density=log_density,
        n_bins=n_bins,
        stats=stats,
        density=density,
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == (x if isinstance(x, str) else "Actual")
    assert fig.layout.yaxis.title.text == (y if isinstance(y, str) else "Predicted")

    # Check that we have scatter data
    assert len(fig.data) > 0
    assert all(isinstance(trace, go.Scatter) for trace in fig.data)

    # Check powerup elements
    if stats:
        # Check stats annotation exists and contains expected metrics
        stats_annotations = [
            ann
            for ann in fig.layout.annotations
            if any(metric in ann.text for metric in ("MAE", "RMSE", "R<sup>2</sup>"))
        ]
        assert len(stats_annotations) == 1, "Stats annotation not found"


@pytest.mark.parametrize("stats", [1, (1,), "foo"])
def test_density_scatter_raises_on_bad_stats_type(stats: Any) -> None:
    """Test that density_scatter raises TypeError for invalid stats type."""
    match = f"stats must be bool or dict, got {type(stats)} instead."

    vals = [1, 2, 3]
    with pytest.raises(TypeError, match=match):
        pmv.density_scatter(x=vals, y=vals, stats=stats)


def test_density_scatter_uses_series_name_as_label() -> None:
    """Test that density_scatter uses pandas Series names as axis labels."""
    x = pd.Series(np_rng.random(5), name="x")
    y = pd.Series(np_rng.random(5), name="y")
    fig = pmv.density_scatter(x=x, y=y, log_density=False)

    assert fig.layout.xaxis.title.text == "x"
    assert fig.layout.yaxis.title.text == "y"


def test_density_scatter_with_hist(df_or_arrays: DfOrArrays) -> None:
    """Test density_scatter_with_hist function."""
    df, x, y = df_or_arrays
    fig = pmv.density_scatter_with_hist(df=df, x=x, y=y)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("gridsize", [50, 100])
@pytest.mark.parametrize("weights", [None, "weights"])
def test_density_hexbin(
    df_or_arrays: DfOrArrays,
    gridsize: int,
    weights: str | None,
) -> None:
    """Test density_hexbin function with Plotly backend."""
    df_test, x, y = df_or_arrays

    # Add weights column if needed
    if weights and isinstance(df_test, pd.DataFrame):
        df_test = df_test.copy()
        df_test[weights] = np_rng.random(len(df_test))
        weights_array = df_test[weights].to_numpy()
    else:
        weights_array = None

    fig = pmv.density_hexbin(
        df=df_test, x=x, y=y, weights=weights_array, gridsize=gridsize
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

    # Check that we have scatter data with hexagon markers
    scatter_traces = [trace for trace in fig.data if isinstance(trace, go.Scatter)]
    assert len(scatter_traces) > 0


def test_density_hexbin_with_hist(df_or_arrays: DfOrArrays) -> None:
    """Test density_hexbin_with_hist function."""
    df, x, y = df_or_arrays
    fig = pmv.density_hexbin_with_hist(df=df, x=x, y=y)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    ("log_density", "stats", "bin_counts_col", "n_bins", "kwargs"),
    [
        (True, True, "custom count col", 1, {"color_continuous_scale": "Viridis"}),
        (
            True,
            dict(prefix="test", x=1, y=1, font_size=10),
            None,
            10,
            {"color_continuous_scale": None},
        ),
        (False, False, None, 100, {"template": "plotly_dark"}),
    ],
)
def test_density_scatter_plotly(
    df_or_arrays: DfOrArrays,
    log_density: bool,
    stats: bool | dict[str, Any],
    bin_counts_col: str | None,
    n_bins: int,
    kwargs: dict[str, Any],
) -> None:
    df, x, y = df_or_arrays
    if df is None:
        return
    fig = pmv.density_scatter_plotly(
        df=df,
        x=x,
        y=y,
        log_density=log_density,
        stats=stats,
        bin_counts_col=bin_counts_col,
        n_bins=n_bins,
        **kwargs,
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == (x if isinstance(x, str) else "Actual")
    assert fig.layout.yaxis.title.text == (y if isinstance(y, str) else "Predicted")
    bin_counts_col = bin_counts_col or "Point Density"
    colorbar = fig.layout.coloraxis.colorbar
    assert colorbar.title.text.replace("<br>", " ") == bin_counts_col

    if log_density:
        assert all(isinstance(val, float) for val in colorbar.tickvals)
        assert all(isinstance(text, str) for text in colorbar.ticktext)
    else:
        assert colorbar.tickvals is None
        assert colorbar.ticktext is None

    # Check powerup elements
    if stats:
        # Check stats annotation exists and contains expected metrics
        stats_annotations = [
            ann
            for ann in fig.layout.annotations
            if any(metric in ann.text for metric in ("MAE", "RMSE", "R<sup>2</sup>"))
        ]
        assert len(stats_annotations) == 1, "Stats annotation not found"
        assert all(
            metric in stats_annotations[0].text for metric in ("MAE", "R<sup>2</sup>")
        ), f"{stats_annotations[0].text=}"
        if isinstance(stats, dict):
            if "prefix" in stats:
                assert stats_annotations[0].text.startswith(stats["prefix"])
            if "x" in stats:
                assert stats_annotations[0].x == stats["x"]
            if "y" in stats:
                assert stats_annotations[0].y == stats["y"]


def test_density_scatter_plotly_hover_template() -> None:
    fig = pmv.density_scatter_plotly(df=df_regr, x=X_COL, y=Y_COL, log_density=True)

    # Check that hover template includes point density
    for trace in fig.data:
        assert "Point<br>Density" in trace.hovertemplate


@pytest.mark.parametrize("stats", [1, (1,), "foo"])
def test_density_scatter_plotly_raises_on_bad_stats_type(stats: Any) -> None:
    match = f"stats must be bool or dict, got {type(stats)} instead."
    with pytest.raises(TypeError, match=match):
        pmv.density_scatter_plotly(df=df_regr, x=X_COL, y=Y_COL, stats=stats)


def test_density_scatter_plotly_empty_dataframe() -> None:
    empty_df = pd.DataFrame({X_COL: [], Y_COL: []})
    with pytest.raises(ValueError, match="No valid traces with required data found"):
        pmv.density_scatter_plotly(df=empty_df, x=X_COL, y=Y_COL)


def test_density_scatter_plotly_facet() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS,
        x="total_bill",
        y="tip",
        facet_col="time",
        n_bins=10,
    )
    assert isinstance(fig, go.Figure)

    # Check that we have multiple subplots
    assert "xaxis" in fig.layout
    assert "xaxis2" in fig.layout

    # Check that each subplot has data
    assert len(fig.data) >= 2


def test_density_scatter_plotly_facet_log_density() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS,
        x="total_bill",
        y="tip",
        facet_col="time",
        log_density=True,
        n_bins=10,
    )
    assert isinstance(fig, go.Figure)

    # Check colorbar configuration for log density
    colorbar = fig.layout.coloraxis.colorbar
    assert colorbar.tickvals is not None
    assert colorbar.ticktext is not None
    assert len(colorbar.tickvals) == len(colorbar.ticktext)

    # Check that tick values are properly formatted
    for tick_text in colorbar.ticktext:
        assert isinstance(tick_text, str)
        # Should not have trailing .0
        assert not tick_text.endswith(".0") or tick_text == "1.0"


def test_density_scatter_plotly_facet_stats() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS,
        x="total_bill",
        y="tip",
        facet_col="time",
        stats=True,
        n_bins=10,
    )
    assert isinstance(fig, go.Figure)

    # Check that stats annotations exist for each facet
    stats_annotations = [
        ann
        for ann in fig.layout.annotations
        if any(metric in ann.text for metric in ("MAE", "RMSE", "R<sup>2</sup>"))
    ]
    assert len(stats_annotations) >= 1


def test_density_scatter_plotly_facet_best_fit_line() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS,
        x="total_bill",
        y="tip",
        facet_col="time",
        best_fit_line=True,
        n_bins=10,
    )
    assert isinstance(fig, go.Figure)

    # Check that best fit lines exist (may be in shapes or traces)
    line_traces = [
        trace for trace in fig.data if hasattr(trace, "mode") and trace.mode == "lines"
    ]
    line_shapes = [shape for shape in fig.layout.shapes if shape.type == "line"]
    assert len(line_traces) >= 1 or len(line_shapes) >= 1


def test_density_scatter_plotly_facet_custom_bins() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="time", n_bins=5
    )
    assert isinstance(fig, go.Figure)


def test_density_scatter_plotly_facet_custom_color() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS,
        x="total_bill",
        y="tip",
        facet_col="time",
        color_continuous_scale="Plasma",
        n_bins=10,
    )
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("density", ["kde", "empirical"])
def test_density_scatter_plotly_facet_density_methods(
    density: Literal["kde", "empirical"],
) -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="time", density=density, n_bins=5
    )
    assert isinstance(fig, go.Figure)


def test_density_scatter_plotly_facet_size() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="time", size="size", n_bins=10
    )
    assert isinstance(fig, go.Figure)


def test_density_scatter_plotly_facet_multiple_categories() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="day", n_bins=5
    )
    assert isinstance(fig, go.Figure)


def test_density_scatter_plotly_facet_identity_line() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="time", identity_line=True
    )
    assert isinstance(fig, go.Figure)


def test_density_scatter_plotly_facet_hover_template() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="time", n_bins=10
    )
    assert isinstance(fig, go.Figure)

    # Check hover template
    for trace in fig.data:
        if hasattr(trace, "hovertemplate"):
            assert "Point<br>Density" in trace.hovertemplate


def test_density_scatter_plotly_colorbar_kwargs() -> None:
    colorbar_kwargs = {"thickness": 20, "len": 0.8}
    fig = pmv.density_scatter_plotly(
        df=df_regr, x=X_COL, y=Y_COL, colorbar_kwargs=colorbar_kwargs
    )
    assert isinstance(fig, go.Figure)
    colorbar = fig.layout.coloraxis.colorbar
    assert colorbar.thickness == 20
    assert colorbar.len == 0.8


def test_colorbar_tick_labels_no_trailing_zeros() -> None:
    # Create data with a wide range to trigger log density
    x_vals = np.concatenate([np_rng.normal(0, 1, 1000), np_rng.normal(10, 1, 10)])
    y_vals = np.concatenate([np_rng.normal(0, 1, 1000), np_rng.normal(10, 1, 10)])
    df_wide_range = pd.DataFrame({"x": x_vals, "y": y_vals})

    fig = pmv.density_scatter_plotly(
        df=df_wide_range, x="x", y="y", log_density=True, n_bins=50
    )

    colorbar = fig.layout.coloraxis.colorbar
    assert colorbar.ticktext is not None

    # Check that tick labels don't have unnecessary trailing .0
    for tick_text in colorbar.ticktext:
        # Allow "1.0" but not other trailing .0
        if tick_text.endswith(".0") and tick_text != "1.0":
            pytest.fail(f"Found trailing .0 in tick label: {tick_text}")


def test_colorbar_density_range_and_formatting() -> None:
    # Create data with known density distribution
    n_points = 1000
    x_vals = np.concatenate(
        [
            np_rng.normal(0, 0.1, n_points // 2),  # High density cluster
            np_rng.normal(5, 2, n_points // 2),  # Lower density cluster
        ]
    )
    y_vals = np.concatenate(
        [
            np_rng.normal(0, 0.1, n_points // 2),  # High density cluster
            np_rng.normal(5, 2, n_points // 2),  # Lower density cluster
        ]
    )
    df_clusters = pd.DataFrame({"x": x_vals, "y": y_vals})

    fig = pmv.density_scatter_plotly(
        df=df_clusters, x="x", y="y", log_density=True, n_bins=20
    )

    colorbar = fig.layout.coloraxis.colorbar
    assert colorbar.tickvals is not None
    assert colorbar.ticktext is not None
    assert len(colorbar.tickvals) == len(colorbar.ticktext)

    # Check that tick values are in ascending order
    tick_vals = list(colorbar.tickvals)
    assert tick_vals == sorted(tick_vals)

    # Check that tick labels are properly formatted
    for tick_text in colorbar.ticktext:
        assert isinstance(tick_text, str)
        assert len(tick_text) > 0


def test_density_scatter_plotly_hover_template_with_custom_template() -> None:
    custom_template = "X: %{x}<br>Y: %{y}<br>Custom: %{marker.color}<extra></extra>"

    fig = pmv.density_scatter_plotly(
        df=df_regr,
        x=X_COL,
        y=Y_COL,
        log_density=True,
        hovertemplate=custom_template,
    )

    # Check that custom template is preserved and density info is added
    for trace in fig.data:
        assert "Point<br>Density" in trace.hovertemplate
        assert "Custom" in trace.hovertemplate


def test_density_scatter_plotly_hover_template_without_extra_tag() -> None:
    custom_template = "X: %{x}<br>Y: %{y}<br>Custom: %{marker.color}"

    fig = pmv.density_scatter_plotly(
        df=df_regr,
        x=X_COL,
        y=Y_COL,
        log_density=True,
        hovertemplate=custom_template,
    )

    # Check that template is properly formatted with extra tag
    for trace in fig.data:
        assert "<extra></extra>" in trace.hovertemplate
        assert "Point<br>Density" in trace.hovertemplate


def test_density_scatter_plotly_hover_format() -> None:
    # Test integer formatting
    fig_int = pmv.density_scatter_plotly(
        df=df_regr, x=X_COL, y=Y_COL, hover_format=".0f"
    )

    # Test decimal formatting
    fig_decimal = pmv.density_scatter_plotly(
        df=df_regr, x=X_COL, y=Y_COL, hover_format=".2f"
    )

    # Check that formatting is applied
    for trace in fig_int.data:
        assert ".0f" in trace.hovertemplate

    for trace in fig_decimal.data:
        assert ".2f" in trace.hovertemplate
