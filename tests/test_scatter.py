from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest

import pymatviz as pmv
from tests.conftest import df_regr, np_rng


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    from tests.conftest import DfOrArrays


X_COL, Y_COL, *_ = df_regr
DF_TIPS = px.data.tips()


@pytest.mark.parametrize(
    ("log_density", "stats"),
    [(True, False), (False, True), (None, dict(prefix="test", font=dict(size=10)))],
)
@pytest.mark.parametrize("n_bins", [None, False, 50])
@pytest.mark.parametrize("density", ["kde", "empirical", None])
def test_density_scatter(
    df_or_arrays: DfOrArrays,
    log_density: bool | None,
    n_bins: int | bool | None,
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


@pytest.mark.parametrize("plot_func", [pmv.density_scatter, pmv.density_hexbin])
@pytest.mark.parametrize("stats", [1, (1,), "foo"])
def test_density_plots_raise_on_bad_stats_type(
    plot_func: Callable[..., go.Figure], stats: Any
) -> None:
    """Density plots raise TypeError for invalid stats type."""
    err_msg = f"stats must be bool or dict, got {type(stats)} instead."

    vals = [1, 2, 3]
    with pytest.raises(TypeError, match=err_msg):
        plot_func(x=vals, y=vals, stats=stats)


def test_density_scatter_uses_series_name_as_label() -> None:
    """Test that density_scatter uses pandas Series names as axis labels."""
    x = pd.Series(np_rng.random(5), name="x")
    y = pd.Series(np_rng.random(5), name="y")
    fig = pmv.density_scatter(x=x, y=y, log_density=False)

    assert fig.layout.xaxis.title.text == "x"
    assert fig.layout.yaxis.title.text == "y"


@pytest.mark.parametrize(
    ("plot_func", "marker_symbol"),
    [
        (pmv.density_scatter_with_hist, "circle"),
        (pmv.density_hexbin_with_hist, "hexagon"),
    ],
)
def test_density_with_hist(
    df_or_arrays: DfOrArrays,
    plot_func: Callable[..., go.Figure],
    marker_symbol: str,
) -> None:
    """Marginal histograms flank the main panel holding parity lines and stats."""
    df, x, y = df_or_arrays
    fig = plot_func(df=df, x=x, y=y, identity_line=True, best_fit_line=True, stats=True)
    assert [type(trace) for trace in fig.data] == [go.Scatter, go.Bar, go.Bar]
    assert fig.data[0].marker.symbol == marker_symbol
    assert fig.layout.xaxis.title.text == (x if isinstance(x, str) else "Actual")
    assert fig.layout.yaxis2.title.text == (y if isinstance(y, str) else "Predicted")
    assert len(fig.layout.shapes) == 2
    assert len(fig.layout.annotations) == 2
    assert all(shape.xref == "x2" and shape.yref == "y2" for shape in fig.layout.shapes)
    assert all(
        anno.xref == "x2 domain" and anno.yref == "y2 domain"
        for anno in fig.layout.annotations
    )
    for marginal_kwarg in ("marginal_x", "marginal_y"):
        with pytest.raises(TypeError, match=rf"\['{marginal_kwarg}'\] not supported"):
            plot_func(df=df, x=x, y=y, **{marginal_kwarg: "box"})


@pytest.mark.parametrize("weighted", [False, True])
def test_density_hexbin_cell_membership(weighted: bool) -> None:
    """Bin by hexagonal cells, keeping zero and negative weighted totals."""
    fig = pmv.density_hexbin(
        [0, 2, 0.49, 0.51, 0.9],
        [0, 2, 0.49, 0.51, 0.1],
        weights=[0, 1, 2, -2, -3] if weighted else None,
        gridsize=2,
        identity_line=False,
        best_fit_line=False,
        stats=False,
    )
    trace = fig.data[0]
    bins = dict(
        zip(zip(trace.x, trace.y, strict=True), trace.marker.color, strict=True)
    )
    assert bins == {
        (0, 0): 0 if weighted else 1,
        (2, 2): 1,
        (0.5, 0.5): 0 if weighted else 2,
        (1, 0): -3 if weighted else 1,
    }


@pytest.mark.parametrize("constant", [False, True])
def test_density_hexbin_weight_alignment(constant: bool) -> None:
    """Preserve weights after filtering DataFrame rows and on constant coordinates."""
    df = pd.DataFrame({"x": [1, np.nan, 1 if constant else 2], "y": [1, 2, 1]})
    fig = pmv.density_hexbin(
        "x",
        "y",
        df=df,
        weights=[2, 100, 3],
        gridsize=2,
        identity_line=False,
        best_fit_line=False,
        stats=False,
    )
    assert sum(fig.data[0].marker.color) == 5
    assert np.isfinite(fig.data[0].x).all()
    assert np.isfinite(fig.data[0].y).all()
    if constant:
        assert tuple(fig.data[0].marker.color) == (5,)
    with pytest.raises(ValueError, match="one weight per DataFrame row"):
        pmv.density_hexbin("x", "y", df=df, weights=[1])


@pytest.mark.parametrize("gridsize", [1, 3, 15])
def test_density_hexbin_nearest_centers(gridsize: int) -> None:
    """Match exhaustive nearest-center assignment across the hexagonal grid."""
    points = np.vstack(([0, 0], [1, 1], np.random.default_rng(seed=0).random((100, 2))))
    grid_x, grid_y = np.meshgrid(
        np.arange(-1, gridsize + 2), np.arange(-1, gridsize + 2)
    )
    lattice = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    candidates = np.vstack((lattice, lattice + 0.5)) / gridsize
    distances = ((points[:, None] - candidates) ** 2 * [1, 3]).sum(axis=2)
    nearest = distances.argmin(axis=1)
    occupied, counts = np.unique(nearest, return_counts=True)
    expected = candidates[occupied]
    order = np.lexsort((expected[:, 1], expected[:, 0]))
    fig = pmv.density_hexbin(
        *points.T,
        gridsize=gridsize,
        identity_line=False,
        best_fit_line=False,
        stats=False,
    )
    actual = np.column_stack((fig.data[0].x, fig.data[0].y))
    # Coordinates in [0, 1] require only grid scaling; allow eight f64 epsilons.
    np.testing.assert_allclose(
        actual, expected[order], rtol=0, atol=8 * np.finfo(float).eps
    )
    np.testing.assert_array_equal(fig.data[0].marker.color, counts[order])


@pytest.mark.parametrize(
    ("coords", "weights", "gridsize", "message"),
    [
        ([], None, 2, "non-empty 1D"),
        ([1, np.inf], None, 2, "must be finite"),
        ([1, 2], [1], 2, "finite weights"),
        ([1, 2], [1, np.nan], 2, "finite weights"),
        ([1, 2], None, 0, "must be positive"),
    ],
)
def test_density_hexbin_invalid_inputs(
    coords: list[float], weights: list[float] | None, gridsize: int, message: str
) -> None:
    """Reject inputs that cannot define a finite hexagonal grid."""
    with pytest.raises(ValueError, match=message):
        pmv.density_hexbin(coords, coords, weights=weights, gridsize=gridsize)


@pytest.mark.parametrize(
    ("log_density", "bin_counts_col", "n_bins", "kwargs", "expected_title"),
    [
        (
            True,
            "custom count col",
            1,
            {"color_continuous_scale": "Viridis"},
            "custom<br>count<br>col",
        ),
        (True, None, 5, {"color_continuous_scale": None}, "Point<br>Density"),
        (False, None, 10, {"template": "plotly_dark"}, "Point<br>Density"),
    ],
)
def test_density_scatter_advanced(
    log_density: bool,
    bin_counts_col: str | None,
    n_bins: int,
    kwargs: dict[str, Any],
    expected_title: str,
) -> None:
    """Binned density colorbar uses bin_counts_col title and log-scale ticks."""
    fig = pmv.density_scatter(
        df=df_regr,
        x=X_COL,
        y=Y_COL,
        log_density=log_density,
        bin_counts_col=bin_counts_col,
        n_bins=n_bins,
        **kwargs,
    )
    assert fig.layout.xaxis.title.text == X_COL
    assert fig.layout.yaxis.title.text == Y_COL
    assert fig.data[0].marker.color is not None
    colorbar = fig.layout.coloraxis.colorbar
    assert colorbar.title.text == expected_title

    if log_density:
        assert all(isinstance(val, float) for val in colorbar.tickvals)
        assert all(isinstance(text, str) for text in colorbar.ticktext)
    else:
        assert colorbar.tickvals is None
        assert colorbar.ticktext is None


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
    fig = pmv.density_scatter(
        df=df_regr,
        x=X_COL,
        y=Y_COL,
        log_density=True,
        hovertemplate=hover_template,
    )

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
        ("time", {"n_bins": 5}, []),
        ("time", {"log_density": True, "n_bins": 5}, ["log_colorbar"]),
        ("time", {"stats": True, "n_bins": 10}, ["stats_annotations"]),
        ("time", {"best_fit_line": True, "n_bins": 5}, ["fit_lines"]),
        # facet order differs from groupby order
        (
            "time",
            {"n_bins": 5, "category_orders": {"time": ["Lunch", "Dinner"]}},
            ["fit_lines", "stats_annotations"],
        ),
        # several traces per facet still get one fit and one stats box per facet
        ("time", {"n_bins": 5, "symbol": "sex"}, ["fit_lines", "stats_annotations"]),
        ("day", {"n_bins": 3}, []),  # more than two categories
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
    # one subplot per facet value
    assert len({trace.xaxis for trace in fig.data}) == DF_TIPS[facet_col].nunique()
    assert "xaxis2" in fig.layout

    # Check log density colorbar
    if "log_colorbar" in expected_checks:
        colorbar = fig.layout.coloraxis.colorbar
        assert colorbar.tickvals is not None
        assert colorbar.ticktext is not None
        assert len(colorbar.tickvals) == len(colorbar.ticktext)
        for tick_text in colorbar.ticktext:
            assert isinstance(tick_text, str)
            assert not tick_text.endswith(".0") or tick_text == "1.0"

    # Check stats annotations: one per facet, positioned inside its subplot
    if "stats_annotations" in expected_checks:
        stats_annotations = [
            ann
            for ann in fig.layout.annotations
            if any(metric in ann.text for metric in ("MAE", "RMSE", "R<sup>2</sup>"))
        ]
        assert sorted(ann.xref for ann in stats_annotations) == [
            "x domain",
            "x2 domain",
        ]
        assert all(0 <= ann.x <= 1 and 0 <= ann.y <= 1 for ann in stats_annotations)

    # Check fit lines
    if "fit_lines" in expected_checks:
        line_traces = [
            trace
            for trace in fig.data
            if hasattr(trace, "mode") and trace.mode == "lines"
        ]
        assert line_traces == []
        # one least-squares fit per facet on its own axes, fitted to raw facet data
        fit_shapes = [shape for shape in fig.layout.shapes if shape.line.width == 2]
        facet_of_axis = {trace.xaxis: trace.customdata[0][1] for trace in fig.data}
        assert sorted(shape.xref for shape in fit_shapes) == sorted(facet_of_axis)
        for shape in fit_shapes:
            facet_df = DF_TIPS[DF_TIPS[facet_col] == facet_of_axis[shape.xref]]
            slope = np.polyfit(facet_df["total_bill"], facet_df["tip"], 1)[0]
            fit_slope = (shape.y1 - shape.y0) / (shape.x1 - shape.x0)
            np.testing.assert_allclose(fit_slope, slope, rtol=1e-9, atol=0)


def test_density_scatter_colorbar_kwargs() -> None:
    """colorbar_kwargs override the default colorbar settings."""
    fig = pmv.density_scatter(
        df=df_regr, x=X_COL, y=Y_COL, colorbar_kwargs={"thickness": 20, "len": 0.8}
    )
    colorbar = fig.layout.coloraxis.colorbar
    assert (colorbar.thickness, colorbar.len) == (20, 0.8)


@pytest.mark.parametrize(
    ("centers", "scales"),
    [((0, 10), (1, 1)), ((0, 5), (0.1, 2))],
    ids=["two-clusters", "dense-and-sparse-clusters"],
)
def test_density_scatter_log_ticks(
    centers: tuple[float, float], scales: tuple[float, float]
) -> None:
    """Log-density colorbar ticks are sorted and labeled without trailing .0."""
    rng = np.random.default_rng(seed=0)
    vals = np.concatenate(
        [
            rng.normal(center, scale, 50)
            for center, scale in zip(centers, scales, strict=True)
        ]
    )
    df = pd.DataFrame({"x": vals, "y": vals + rng.normal(0, 0.1, len(vals))})
    fig = pmv.density_scatter(df=df, x="x", y="y", log_density=True, n_bins=10)

    colorbar = fig.layout.coloraxis.colorbar
    assert len(colorbar.tickvals) == len(colorbar.ticktext) > 0
    assert list(colorbar.tickvals) == sorted(colorbar.tickvals)
    for tick_text in colorbar.ticktext:
        assert tick_text
        assert not tick_text.endswith(".0")


@pytest.mark.parametrize(
    ("counts", "expected_ticks"),
    [
        (
            [1, 10, 3000],
            ["1", "2", "5", "10", "20", "50", "100", "200", "500", "1k", "2k"],
        ),
        ([5, 400], ["5", "15", "45", "134", "400"]),
    ],
    ids=["wide-range-decades", "narrow-range-logspace"],
)
def test_update_colorbar_for_log_density_ticks(
    counts: list[int], expected_ticks: list[str]
) -> None:
    """Wide count ranges get 1-2-5 decade ticks, narrow ones log-spaced ticks."""
    fig = go.Figure(go.Scatter(x=[1], y=[1]))
    color_vals = np.log10(np.array(counts, dtype=float) + 1)
    pmv.scatter._update_colorbar_for_log_density(fig, color_vals, "Point Density")
    colorbar = fig.layout.coloraxis.colorbar
    assert list(colorbar.ticktext) == expected_ticks
    assert colorbar.title.text == "Point<br>Density"


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
    ("stats_config", "expected_labels", "anno_props"),
    [
        (
            {"prefix": "Model:", "x": 0.1, "y": 0.9},
            ["MAE", "R<sup>2</sup>"],
            {"x": 0.1, "y": 0.9},
        ),
        ({"metrics": ["MAE", "R2"], "fmt": ".2f"}, ["MAE", "R<sup>2</sup>"], {}),
        (
            {"font": {"size": 14, "color": "red"}},
            ["MAE", "R<sup>2</sup>"],
            {"font.size": 14, "font.color": "red"},
        ),
        ({"metrics": ["MAE"]}, ["MAE"], {}),
        ({"metrics": ["R2"]}, ["R<sup>2</sup>"], {}),
        ({"metrics": ["MAE", "RMSE", "R2"]}, ["MAE", "RMSE", "R<sup>2</sup>"], {}),
    ],
)
def test_density_scatter_stats_annotation(
    stats_config: dict[str, Any],
    expected_labels: list[str],
    anno_props: dict[str, Any],
) -> None:
    """Stats dicts control annotation metrics, prefix, number format and style."""
    fig = pmv.density_scatter(df=df_regr, x=X_COL, y=Y_COL, stats=stats_config)
    (ann,) = [ann for ann in fig.layout.annotations if "LS fit" not in ann.text]

    prefix = stats_config.get("prefix", "")
    assert ann.text.startswith(prefix)
    lines = ann.text.removeprefix(prefix).removesuffix("<br>").split("<br>")
    assert [line.split(" = ")[0] for line in lines] == expected_labels
    if stats_config.get("fmt") == ".2f":
        assert all(
            re.fullmatch(r"-?\d+\.\d{2}", line.split(" = ")[1]) for line in lines
        )
    for key, expected in anno_props.items():
        obj = ann
        for attr in key.split("."):
            obj = getattr(obj, attr)
        assert obj == expected


def test_density_scatter_single_point() -> None:
    """A single data point falls back to empirical density with undefined R^2."""
    single_point_df = pd.DataFrame({X_COL: [1.0], Y_COL: [1.0]})
    with (
        pytest.warns(UserWarning, match="Not enough data points for KDE"),
        pytest.warns(UserWarning, match="Skipping best fit line: needs >= 2 finite"),
    ):
        fig = pmv.density_scatter(df=single_point_df, x=X_COL, y=Y_COL, stats=True)
    assert fig.layout.annotations[-1].text == "MAE = 0.0<br>R<sup>2</sup> = nan<br>"
    assert len(fig.layout.shapes) == 1  # identity line only, no degenerate fit


def test_density_scatter_zero_handling_log_density() -> None:
    """Test log density auto-detection handles zeros without warnings."""
    df_zero = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]})
    fig = pmv.density_scatter(df=df_zero, x="x", y="y", log_density=None, n_bins=3)
    assert isinstance(fig, go.Figure)
    assert fig.layout.coloraxis.colorbar.tickvals is None  # No log scaling
