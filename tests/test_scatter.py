from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

import pymatviz as pmv
from pymatviz.scatter import bin_df_cols
from pymatviz.widgets.binned_scatter_plot import BinnedScatterPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from tests.conftest import df_regr, np_rng


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    from tests.conftest import DfOrArrays


X_COL, Y_COL, *_ = df_regr


@pytest.mark.parametrize("log", [True, False, None])
@pytest.mark.parametrize("n_bins", [None, False, 50])
@pytest.mark.parametrize("density", ["kde", "empirical", None])
def test_density_scatter(
    df_or_arrays: DfOrArrays,
    log: bool | None,
    n_bins: int | Literal[False] | None,
    density: Literal["kde", "empirical"] | None,
) -> None:
    """Density scatter colors points by density and adds parity powerups."""
    df, x, y = df_or_arrays
    widget = pmv.density_scatter(x, y, df=df, log=log, bins=n_bins, density=density)
    assert isinstance(widget, ScatterPlotWidget)
    assert widget.x_axis == {"label": x if isinstance(x, str) else "Actual"}
    assert widget.y_axis == {"label": y if isinstance(y, str) else "Predicted"}
    (srs,) = widget.series
    colors = np.asarray(srs["color_values"])
    assert len(colors) == len(srs["x"]) == len(srs["y"])
    assert np.all(np.diff(colors) >= 0)  # densest points drawn last
    if not n_bins:  # unbinned points are the inputs (reordered by density)
        np.testing.assert_array_equal(np.sort(srs["x"]), np.sort(df_regr[X_COL]))
    if density == "empirical" and not n_bins:
        assert set(colors) == {1}
    assert widget.color_scale["type"] == ("log" if log else "linear")
    assert widget.color_bar == {"title": "Point density"}
    assert [line["type"] for line in widget.ref_lines] == ["diagonal", "segment"]
    # metrics of the unbinned data, also when binning
    true_vals, pred_vals = df_regr[X_COL], df_regr[Y_COL]
    mae = skm.mean_absolute_error(true_vals, pred_vals)
    r2_val = skm.r2_score(true_vals, pred_vals)
    assert widget.annotation == f"MAE = {mae:.3}\nR² = {r2_val:.3}"


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


@pytest.mark.parametrize(
    "hist_kwargs", [{}, {"marginal_bins": 100}, {"marginal_bins": 100, "bin_px": 1.5}]
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
    ],
)
def test_density_plots_invalid_inputs(
    plot_func: Callable[..., Any], kwargs: dict[str, Any], match: str
) -> None:
    """Empty, non-finite or mismatched inputs and bad marginal bins raise."""
    with pytest.raises(ValueError, match=match):
        plot_func(**kwargs)


def test_density_scatter_binning_reduces_points() -> None:
    """Over 1000 points auto-bin into at most 200x200 bin centers with counts."""
    xs = np_rng.normal(size=5000)
    widget = pmv.density_scatter(xs, xs + np_rng.normal(size=5000), stats=False)
    (srs,) = widget.series
    assert len(srs["x"]) < 5000
    assert sum(srs["color_values"]) == 5000  # empirical counts of all points


@pytest.mark.parametrize(
    ("xs", "ys"),
    [([1, 1, 1], [1, 2, 3]), ([1, 2, 3], [5, 5, 5]), ([1, 2, 3], [1, 2, 3])],
)
@pytest.mark.parametrize("n_bins", [False, 2])
def test_density_scatter_collinear_kde(
    xs: list[int], ys: list[int], n_bins: int | Literal[False]
) -> None:
    """Collinear points have no 2D KDE and fall back to empirical density."""
    with pytest.warns(UserWarning, match="KDE impossible for 3 point"):
        widget = pmv.density_scatter(xs, ys, stats=False, density="kde", bins=n_bins)
    assert sum(widget.series[0]["color_values"]) == 3  # counts of all 3 points


def test_density_scatter_single_point() -> None:
    """A single point falls back to empirical density and skips the auto fit."""
    single_point_df = pd.DataFrame({X_COL: [1.0], Y_COL: [1.0]})
    with pytest.warns(UserWarning, match="KDE impossible for 1 point"):
        widget = pmv.density_scatter(X_COL, Y_COL, df=single_point_df)
    assert widget.annotation == "MAE = 0.0\nR² = nan"
    assert [line["type"] for line in widget.ref_lines] == ["diagonal"]


@pytest.mark.parametrize(
    ("density_vals", "expected_log"), [([1, 1, 1], False), ([1, 1000], True)]
)
def test_density_scatter_auto_log(density_vals: list[int], expected_log: bool) -> None:
    """log=None turns on log colors when densities span > 2 decades."""
    xs = np.repeat(np.arange(len(density_vals), dtype=float), density_vals)
    widget = pmv.density_scatter(xs, xs, bins=len(density_vals), stats=False)
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
    xs = np.repeat([0.0, 1.0], [1, 1000])
    log_widget = pmv.density_scatter(
        xs, xs, bins=2, stats=False, color_scale="interpolatePlasma"
    )
    assert log_widget.color_scale == {"scheme": "interpolatePlasma", "type": "log"}
    assert widget.annotation == "RMSE = 0.58"
    assert [line["type"] for line in widget.ref_lines] == ["segment"]


@pytest.mark.parametrize(
    (
        "bin_by_cols",
        "group_by_cols",
        "n_bins",
        "expected_n_bins",
        "verbose",
        "density_col",
        "expected_n_rows",
    ),
    [
        (["A"], [], 2, [2], True, "", 2),
        (["A", "B"], [], 2, [2, 2], True, "kde_bin_counts", 4),
        (["A", "B"], [], [2, 3], [2, 3], False, "kde_bin_counts", 6),
        (["A"], ["B"], 2, [2], False, "", 30),
    ],
)
def test_bin_df_cols(
    bin_by_cols: list[str],
    group_by_cols: list[str],
    n_bins: int | list[int],
    expected_n_bins: list[int],
    verbose: bool,
    density_col: str,
    expected_n_rows: int,
    df_float: pd.DataFrame,
    capsys: pytest.CaptureFixture[str],
) -> None:
    idx_col = "index"
    # don't move this below df_float.copy() line
    df_float.index.name = idx_col

    # keep copy of original DataFrame to assert it is not modified
    # not using df.copy(deep=True) here for extra sensitivity, doc str says
    # not as deep as deepcopy
    df_float_orig = copy.deepcopy(df_float)

    bin_counts_col = "bin_counts"
    capsys.readouterr()
    df_binned = bin_df_cols(
        df_float,
        bin_by_cols,
        group_by_cols=group_by_cols,
        n_bins=n_bins,
        **({"verbose": True} if verbose else {}),  # quiet by default
        bin_counts_col=bin_counts_col,
        density_col=density_col,
    )

    stdout = capsys.readouterr().out
    assert ("sample reduction from binning" in stdout) is verbose
    assert len(df_binned) == expected_n_rows, f"{len(df_binned)=} {expected_n_rows=}"
    assert len(df_binned) <= len(df_float), f"{len(df_binned)=} {len(df_float)=}"
    assert df_binned.index.name == idx_col

    # ensure binned DataFrame has a minimum set of expected columns
    expected_cols = {bin_counts_col, *df_float, *(f"{col}_bins" for col in bin_by_cols)}
    assert {*df_binned} >= expected_cols, (
        f"{set(df_binned)=}\n{expected_cols=},\n{bin_by_cols=}\n{group_by_cols=}"
    )

    # validate the number of unique bins for each binned column
    for col, n_bins_expec in zip(bin_by_cols, expected_n_bins, strict=True):
        assert df_binned[f"{col}_bins"].nunique() == n_bins_expec

    # ensure original DataFrame is not modified
    pd.testing.assert_frame_equal(df_float, df_float_orig)

    # Check that the index values of df_binned are a subset of df_float
    assert set(df_binned.index).issubset(set(df_float.index))

    # Check that bin_counts column exists and contains only integers
    assert bin_counts_col in df_binned
    assert df_binned[bin_counts_col].dtype in [int, "int64"]

    # If density column is specified, it must exist and be scaled to the number
    # of data points (regression: was scaled by the number of binned columns)
    if density_col:
        assert df_binned[density_col].sum() == pytest.approx(len(df_float))
    else:
        assert density_col not in df_binned


def test_bin_df_cols_raises() -> None:
    df_dummy = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [2, 3, 4, 5]})
    bin_by_cols = ["col1", "col2"]

    # test error when passing n_bins as list but list has wrong length
    with pytest.raises(
        ValueError, match=re.escape("len(bin_by_cols)=2 != len(n_bins)=1")
    ):
        bin_df_cols(df_dummy, bin_by_cols, n_bins=[2])
