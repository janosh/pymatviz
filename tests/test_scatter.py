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
            5,
            {"color_continuous_scale": None},
        ),
        (False, False, None, 10, {"template": "plotly_dark"}),
    ],
)
def test_density_scatter_advanced(
    df_or_arrays: DfOrArrays,
    log_density: bool,
    stats: bool | dict[str, Any],
    bin_counts_col: str | None,
    n_bins: int,
    kwargs: dict[str, Any],
) -> None:
    df, x, y = df_or_arrays
    if df is None or not isinstance(x, str) or not isinstance(y, str):
        pytest.skip("Requires DataFrame with named x/y columns")
    fig = pmv.density_scatter(
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
    assert fig.layout.xaxis.title.text == x
    assert fig.layout.yaxis.title.text == y
    colorbar = fig.layout.coloraxis.colorbar
    expected_title = colorbar.title.text.replace("<br>", " ")
    assert expected_title == (bin_counts_col or expected_title)

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


@pytest.mark.parametrize(
    ("hover_template", "expected_content"),
    [
        (None, ["Point<br>Density"]),
        (
            "X: %{x}<br>Y: %{y}<br>Custom: %{marker.color}<extra></extra>",
            ["Point<br>Density", "Custom"],
        ),
        (
            "X: %{x}<br>Y: %{y}<br>Custom: %{marker.color}",
            ["Point<br>Density", "<extra></extra>"],
        ),
    ],
)
def test_density_scatter_hover_templates(
    hover_template: str | None, expected_content: list[str]
) -> None:
    """Test various hover template configurations."""
    kwargs = {"log_density": True}
    if hover_template:
        kwargs["hovertemplate"] = hover_template

    fig = pmv.density_scatter(df=df_regr, x=X_COL, y=Y_COL, **kwargs)

    for trace in fig.data:
        for content in expected_content:
            assert content in trace.hovertemplate


def test_density_scatter_empty_dataframe() -> None:
    empty_df = pd.DataFrame({X_COL: [], Y_COL: []})
    with pytest.raises(ValueError, match="No valid traces with required data found"):
        pmv.density_scatter(df=empty_df, x=X_COL, y=Y_COL)


@pytest.mark.parametrize(
    ("facet_col", "kwargs", "expected_checks"),
    [
        # Basic facet test
        ("time", {"n_bins": 5}, ["subplots", "data_count"]),
        # Log density test
        ("time", {"log_density": True, "n_bins": 5}, ["subplots", "log_colorbar"]),
        # Stats test
        ("time", {"stats": True, "n_bins": 5}, ["subplots", "stats_annotations"]),
        # Best fit line test
        ("time", {"best_fit_line": True, "n_bins": 5}, ["subplots", "fit_lines"]),
        # Multiple categories test
        ("day", {"n_bins": 3}, ["subplots", "data_count"]),
    ],
)
@pytest.mark.parametrize("density", ["kde", None])
def test_density_scatter_facet_variations(
    facet_col: str,
    kwargs: dict[str, Any],
    expected_checks: list[str],
    density: str | None,
) -> None:
    """Test various facet configurations."""
    if density:
        kwargs["density"] = density

    fig = pmv.density_scatter(
        df=DF_TIPS, x="total_bill", y="tip", facet_col=facet_col, **kwargs
    )
    assert isinstance(fig, go.Figure)

    # Check subplots exist
    if "subplots" in expected_checks:
        assert "xaxis" in fig.layout
        assert "xaxis2" in fig.layout
        assert len(fig.data) >= 2

    # Check data count
    if "data_count" in expected_checks:
        assert len(fig.data) >= 2

    # Check log density colorbar
    if "log_colorbar" in expected_checks:
        colorbar = fig.layout.coloraxis.colorbar
        assert colorbar.tickvals is not None
        assert colorbar.ticktext is not None
        assert len(colorbar.tickvals) == len(colorbar.ticktext)
        for tick_text in colorbar.ticktext:
            assert isinstance(tick_text, str)
            assert not tick_text.endswith(".0") or tick_text == "1.0"

    # Check stats annotations
    if "stats_annotations" in expected_checks:
        stats_annotations = [
            ann
            for ann in fig.layout.annotations
            if any(metric in ann.text for metric in ("MAE", "RMSE", "R<sup>2</sup>"))
        ]
        assert len(stats_annotations) >= 1

    # Check fit lines
    if "fit_lines" in expected_checks:
        line_traces = [
            trace
            for trace in fig.data
            if hasattr(trace, "mode") and trace.mode == "lines"
        ]
        line_shapes = [shape for shape in fig.layout.shapes if shape.type == "line"]
        assert len(line_traces) >= 1 or len(line_shapes) >= 1

    # Check hover template
    if "hover_template" in expected_checks:
        for trace in fig.data:
            if hasattr(trace, "hovertemplate"):
                assert "Point<br>Density" in trace.hovertemplate


@pytest.mark.parametrize(
    ("test_type", "kwargs", "expected_checks"),
    [
        # Colorbar kwargs test
        (
            "colorbar_kwargs",
            {"colorbar_kwargs": {"thickness": 20, "len": 0.8}},
            ["thickness", "len"],
        ),
        # Tick labels test
        ("tick_labels", {"log_density": True, "n_bins": 10}, ["tick_formatting"]),
        # Density range test
        (
            "density_range",
            {"log_density": True, "n_bins": 10},
            ["tick_ordering", "tick_formatting"],
        ),
    ],
)
def test_density_scatter_colorbar_variations(
    test_type: str, kwargs: dict[str, Any], expected_checks: list[str]
) -> None:
    """Test various colorbar configurations."""
    if test_type == "tick_labels":
        # Create data with wide range to trigger log density (smaller dataset)
        x_vals = np.concatenate([np_rng.normal(0, 1, 50), np_rng.normal(10, 1, 5)])
        y_vals = np.concatenate([np_rng.normal(0, 1, 50), np_rng.normal(10, 1, 5)])
        df = pd.DataFrame({"x": x_vals, "y": y_vals})
        x_col, y_col = "x", "y"
    elif test_type == "density_range":
        # Create data with known density distribution (smaller dataset)
        n_points = 100
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
        df = pd.DataFrame({"x": x_vals, "y": y_vals})
        x_col, y_col = "x", "y"
    else:
        df, x_col, y_col = df_regr, X_COL, Y_COL

    fig = pmv.density_scatter(df=df, x=x_col, y=y_col, **kwargs)
    assert isinstance(fig, go.Figure)

    colorbar = fig.layout.coloraxis.colorbar

    # Check thickness and len
    if "thickness" in expected_checks:
        assert colorbar.thickness == 20
    if "len" in expected_checks:
        assert colorbar.len == 0.8

    # Check tick formatting
    if "tick_formatting" in expected_checks:
        assert colorbar.ticktext is not None
        for tick_text in colorbar.ticktext:
            assert isinstance(tick_text, str)
            assert len(tick_text) > 0
            # Check no unnecessary trailing .0 (except for "1.0")
            if tick_text.endswith(".0") and tick_text != "1.0":
                pytest.fail(f"Found trailing .0 in tick label: {tick_text}")

    # Check tick ordering
    if "tick_ordering" in expected_checks:
        assert colorbar.tickvals is not None
        assert colorbar.ticktext is not None
        assert len(colorbar.tickvals) == len(colorbar.ticktext)
        tick_vals = list(colorbar.tickvals)
        assert tick_vals == sorted(tick_vals)


@pytest.mark.parametrize(
    ("hover_format", "expected_format"),
    [(".0f", ".0f"), (".2f", ".2f"), (".3f", ".3f")],
)
def test_density_scatter_hover_format(hover_format: str, expected_format: str) -> None:
    """Test hover format configuration."""
    fig = pmv.density_scatter(df=df_regr, x=X_COL, y=Y_COL, hover_format=hover_format)

    for trace in fig.data:
        assert expected_format in trace.hovertemplate


@pytest.mark.parametrize(
    "stats_config",
    [
        True,
        False,
        {"prefix": "Model:", "x": 0.1, "y": 0.9},
        {"metrics": ["MAE", "R2"], "fmt": ".2f"},
        {"font": {"size": 14, "color": "red"}},
    ],
)
def test_density_scatter_stats_annotation_configs(
    stats_config: bool | dict[str, Any],
) -> None:
    """Test stats annotation with various configuration options."""
    fig = pmv.density_scatter(df=df_regr, x=X_COL, y=Y_COL, stats=stats_config)
    assert isinstance(fig, go.Figure)

    if not stats_config:
        return

    stats_annotations = [
        ann
        for ann in fig.layout.annotations
        if any(
            metric in ann.text
            for metric in ("MAE", "RMSE", "R<sup>2</sup>", "MSE", "MAPE")
        )
    ]
    assert len(stats_annotations) >= 1

    if isinstance(stats_config, dict):
        ann = stats_annotations[0]
        if "prefix" in stats_config:
            assert ann.text.startswith(stats_config["prefix"])
        if "x" in stats_config:
            assert ann.x == stats_config["x"]
        if "y" in stats_config:
            assert ann.y == stats_config["y"]
        if "font" in stats_config:
            font_config = stats_config["font"]
            if "size" in font_config:
                assert ann.font.size == font_config["size"]
            if "color" in font_config:
                assert ann.font.color == font_config["color"]


@pytest.mark.parametrize("metrics", [["MAE"], ["R2"], ["MAE", "RMSE", "R2"]])
def test_density_scatter_stats_annotation_metrics(metrics: list[str]) -> None:
    """Test stats annotation with different metric combinations."""
    fig = pmv.density_scatter(df=df_regr, x=X_COL, y=Y_COL, stats={"metrics": metrics})
    stats_annotations = [
        ann
        for ann in fig.layout.annotations
        if any(
            metric in ann.text
            for metric in ("MAE", "RMSE", "R<sup>2</sup>", "MSE", "MAPE")
        )
    ]
    assert len(stats_annotations) == 1

    ann_text = stats_annotations[0].text
    for metric in metrics:
        if metric == "R2":
            assert "R<sup>2</sup>" in ann_text
        else:
            assert metric in ann_text


def test_density_scatter_stats_annotation_edge_cases() -> None:
    """Test stats annotation with edge cases."""
    # Test with single data point (should handle gracefully)
    single_point_df = pd.DataFrame({X_COL: [1.0], Y_COL: [1.0]})
    fig = pmv.density_scatter(df=single_point_df, x=X_COL, y=Y_COL, stats=True)
    assert isinstance(fig, go.Figure)

    # Test with normal data that should work
    normal_df = pd.DataFrame({X_COL: [1, 2, 3, 4, 5], Y_COL: [1.1, 2.2, 2.9, 4.1, 5.0]})
    fig = pmv.density_scatter(df=normal_df, x=X_COL, y=Y_COL, stats=True)

    stats_annotations = [
        ann
        for ann in fig.layout.annotations
        if any(metric in ann.text for metric in ("MAE", "R<sup>2</sup>"))
    ]
    assert len(stats_annotations) >= 1


def test_density_scatter_stats_annotation_faceted() -> None:
    """Test stats annotation with faceted plots."""
    fig = pmv.density_scatter(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="time", stats=True, n_bins=10
    )

    # Verify faceting and stats annotations
    assert "xaxis2" in fig.layout  # Multiple subplots
    assert len(fig.data) >= 2  # Multiple data traces

    stats_annotations = [
        ann
        for ann in fig.layout.annotations
        if any(metric in ann.text for metric in ("MAE", "R<sup>2</sup>"))
    ]
    assert len(stats_annotations) >= 1

    # Verify annotation quality
    for ann in stats_annotations:
        assert 0 <= ann.x <= 1  # Positioned in plot area
        assert 0 <= ann.y <= 1
        assert isinstance(ann.text, str)
        assert len(ann.text) > 0  # Proper formatting


def test_density_scatter_zero_handling_log_density() -> None:
    """Test log density auto-detection handles zeros without warnings."""
    df_zero = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]})
    fig = pmv.density_scatter(df=df_zero, x="x", y="y", log_density=None, n_bins=3)
    assert isinstance(fig, go.Figure)
    assert fig.layout.coloraxis.colorbar.tickvals is None  # No log scaling


def test_density_scatter_empty_dataframe_handling() -> None:
    """Test empty DataFrames raise appropriate error."""
    empty_df = pd.DataFrame({"x": [], "y": []})
    with pytest.raises(ValueError, match="No valid traces with required data found"):
        pmv.density_scatter(df=empty_df, x="x", y="y", n_bins=5)


def test_density_scatter_density_column_handling() -> None:
    """Test density column handling with None values."""
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.1, 2.2, 2.9, 4.1]})
    fig = pmv.density_scatter(df=df, x="x", y="y", density="empirical", n_bins=3)
    assert isinstance(fig, go.Figure)
