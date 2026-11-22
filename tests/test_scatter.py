from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import sklearn.metrics as skm

import pymatviz as pmv
from pymatviz.scatter import MAX_KDE_POINTS
from pymatviz.widgets.binned_scatter_plot import BinnedScatterPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from tests.conftest import df_regr, np_rng


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from tests.conftest import DfOrArrays


X_COL, Y_COL, *_ = df_regr


@pytest.mark.parametrize("log", [True, False, None])
def test_density_scatter(df_or_arrays: DfOrArrays, log: bool | None) -> None:
    """Density scatter colors points by their KDE and adds parity powerups."""
    df, x, y = df_or_arrays
    widget = pmv.density_scatter(x, y, df=df, log=log)
    assert isinstance(widget, ScatterPlotWidget)
    assert widget.x_axis == {"label": x if isinstance(x, str) else "Actual"}
    assert widget.y_axis == {"label": y if isinstance(y, str) else "Predicted"}
    (srs,) = widget.series
    colors = np.asarray(srs["color_values"])
    assert len(colors) == len(srs["x"]) == len(srs["y"])
    assert np.all(np.diff(colors) >= 0)  # densest points drawn last
    # the input points reordered by density, colored by scipy's KDE
    np.testing.assert_array_equal(np.sort(srs["x"]), np.sort(df_regr[X_COL]))
    xy_in = np.vstack([df_regr[X_COL], df_regr[Y_COL]])
    kde = scipy.stats.gaussian_kde(xy_in)(np.vstack([srs["x"], srs["y"]]))
    np.testing.assert_allclose(colors, kde, rtol=1e-12, atol=0)
    assert widget.color_scale["type"] == ("log" if log else "linear")
    assert widget.color_bar == {"title": "Point density"}
    assert [line["type"] for line in widget.ref_lines] == ["diagonal", "segment"]
    true_vals, pred_vals = df_regr[X_COL], df_regr[Y_COL]
    mae = skm.mean_absolute_error(true_vals, pred_vals)
    r2_val = skm.r2_score(true_vals, pred_vals)
    assert widget.annotation == f"MAE = {mae:.3g}\nR² = {r2_val:.3g}"


@pytest.mark.parametrize("plot_func", [pmv.density_scatter, pmv.density_hexbin])
@pytest.mark.parametrize("stats", [1, (1,), "foo"])
def test_density_plots_raise_on_bad_stats_type(
    plot_func: Callable[..., Any], stats: Any
) -> None:
    """Density plots raise TypeError for invalid stats type."""
    with pytest.raises(TypeError, match="stats must be bool or dict"):
        plot_func(x=[1, 2, 3], y=[1, 3, 2], stats=stats)


@pytest.mark.parametrize("plot_func", [pmv.density_scatter, pmv.density_hexbin])
def test_density_plots_use_series_names_and_drop_nan(
    plot_func: Callable[..., Any],
) -> None:
    """Series names become axis labels and NaN rows are dropped."""
    x_vals = pd.Series([*np_rng.random(5), np.nan], name="x")
    y_vals = pd.Series([*np_rng.random(5), 1.0], name="y")
    widget = plot_func(x_vals, y_vals, stats=False, xlabel=None, ylabel="custom")
    assert widget.x_axis == {"label": "x"}
    assert widget.y_axis == {"label": "custom"}
    assert len(widget.series[0]["x"]) == 5
    unnamed = plot_func(pd.Series(x_vals.to_numpy()), y_vals.rename(None), stats=False)
    assert (unnamed.x_axis, unnamed.y_axis) == (
        {"label": "Actual"},
        {"label": "Predicted"},
    )
    # integer df column labels name the axes too
    df_int_cols = pd.DataFrame({0: x_vals, 1: y_vals})
    int_cols = plot_func(0, 1, df=df_int_cols, stats=False)
    assert (int_cols.x_axis, int_cols.y_axis) == ({"label": "0"}, {"label": "1"})


@pytest.mark.parametrize(
    "hist_kwargs",
    [
        {},
        {"marginal_bins": 100},
        {"marginal_bins": np.int64(100), "bin_px": 1.5},  # numpy ints are ints too
    ],
)
def test_binned_density_plots(
    df_or_arrays: DfOrArrays, hist_kwargs: dict[str, Any]
) -> None:
    """Binned density plots send raw points, overlays and marginal histograms."""
    df, x, y = df_or_arrays
    widget = pmv.density_hexbin(x, y, df=df, best_fit_line=True, **hist_kwargs)
    assert isinstance(widget, BinnedScatterPlotWidget)
    np.testing.assert_array_equal(widget.series[0]["x"], df_regr[X_COL])
    np.testing.assert_array_equal(widget.series[0]["y"], df_regr[Y_COL])
    assert widget.x_axis == {"label": x if isinstance(x, str) else "Actual"}
    lines = widget.overlays["ref_lines"]
    assert [line["type"] for line in lines] == ["diagonal", "segment"]
    assert all("annotation" not in line for line in lines)  # moved to the badge
    assert widget.annotation.startswith("LS fit: y = ")
    assert "\nMAE = " in widget.annotation
    with_hist = "marginal_bins" in hist_kwargs
    assert bool(widget.marginals) == with_hist
    if with_hist:
        assert widget.marginals["top"] == {"type": "histogram", "bins": 100}
        assert type(widget.marginals["top"]["bins"]) is int
    assert widget.density.get("bin_px") == hist_kwargs.get("bin_px")
    if with_hist:  # user marginals merge per side over the marginal_bins histograms
        custom = pmv.density_hexbin(
            x, y, df=df, marginals={"right": None}, stats=False, **hist_kwargs
        )
        assert custom.marginals == {
            "top": {"type": "histogram", "bins": 100},
            "right": None,
        }


@pytest.mark.parametrize(
    ("plot_func", "kwargs", "match"),
    [
        (pmv.density_hexbin, {"x": [], "y": []}, "at least one finite"),
        (pmv.density_hexbin, {"x": [np.nan, 1], "y": [1, np.nan]}, "at least one"),
        (pmv.density_hexbin, {"x": [np.nan], "y": [1]}, "at least one finite"),
        (pmv.density_scatter, {"x": [np.nan], "y": [1]}, "at least one finite"),
        (pmv.density_scatter, {"x": [1, 2], "y": [1]}, "equal length"),
        (
            pmv.density_hexbin,
            {"x": [1, 2], "y": [1, 2], "marginal_bins": 0},
            "marginal_bins=0 must be a positive int",
        ),
        (
            pmv.density_hexbin,
            {"x": [1, 2], "y": [1, 2], "marginal_bins": True},
            "marginal_bins=True must be a positive int",
        ),
    ],
)
def test_density_plots_invalid_inputs(
    plot_func: Callable[..., Any], kwargs: dict[str, Any], match: str
) -> None:
    """Empty, non-finite or mismatched inputs and bad marginal bins raise."""
    with pytest.raises(ValueError, match=match):
        plot_func(**kwargs)


def test_density_scatter_rejects_large_data() -> None:
    """Over MAX_KDE_POINTS points raise, pointing to density_hexbin."""
    xs = np.zeros(MAX_KDE_POINTS + 1)
    with pytest.raises(ValueError, match="got 10,001: use density_hexbin"):
        pmv.density_scatter(xs, xs)


@pytest.mark.parametrize(
    ("xs", "ys"),
    [([1, 1, 1], [1, 2, 3]), ([1, 2, 3], [5, 5, 5]), ([1, 2, 3], [1, 2, 3])],
)
def test_density_scatter_collinear_kde(xs: list[int], ys: list[int]) -> None:
    """Collinear points have no 2D KDE and get uniform colors."""
    with pytest.warns(UserWarning, match="KDE impossible for 3 point.*uniform"):
        widget = pmv.density_scatter(xs, ys, stats=False)
    assert list(widget.series[0]["color_values"]) == [1, 1, 1]


def test_density_scatter_single_point() -> None:
    """A single point gets a uniform color and skips the auto fit."""
    single_point_df = pd.DataFrame({X_COL: [1.0], Y_COL: [1.0]})
    with pytest.warns(UserWarning, match="KDE impossible for 1 point"):
        widget = pmv.density_scatter(X_COL, Y_COL, df=single_point_df)
    assert widget.annotation == "MAE = 0\nR² = nan"
    assert [line["type"] for line in widget.ref_lines] == ["diagonal"]


@pytest.mark.parametrize(("n_far", "expected_log"), [(0, False), (2, True)])
def test_density_scatter_auto_log(n_far: int, expected_log: bool) -> None:
    """log=None turns on log colors when densities span > 2 decades: a tight
    cluster spans 1.97 decades, 2 far points stretch it to 2.39.
    """
    rng = np.random.default_rng(seed=0)
    xs = [*rng.normal(scale=0.01, size=1000), *rng.normal(1, 0.01, size=n_far)]
    ys = [*rng.normal(scale=0.01, size=1000), *rng.normal(0, 0.01, size=n_far)]
    widget = pmv.density_scatter(xs, ys, stats=False)
    assert widget.color_scale["type"] == ("log" if expected_log else "linear")


def test_density_scatter_kwargs() -> None:
    """Extra kwargs go to the widget, overriding defaults."""
    widget = pmv.density_scatter(
        [1, 2, 3],
        [1, 2, 4],
        color_bar={"title": "Count"},
        color_scale="interpolatePlasma",
        stats={"metrics": ["RMSE"], "fmt": ".2f"},
        identity_line=False,
    )
    assert widget.color_bar == {"title": "Count"}
    # a bare scheme name keeps the automatic log/linear color scale type
    assert widget.color_scale == {"scheme": "interpolatePlasma", "type": "linear"}
    log_widget = pmv.density_scatter(
        [1, 2, 3], [1, 2, 4], log=True, stats=False, color_scale="interpolatePlasma"
    )
    assert log_widget.color_scale == {"scheme": "interpolatePlasma", "type": "log"}
    assert widget.annotation == "RMSE = 0.58"
    assert [line["type"] for line in widget.ref_lines] == ["segment"]
