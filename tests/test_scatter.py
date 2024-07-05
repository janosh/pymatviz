from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest

from pymatviz import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_plotly,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)
from tests.conftest import df_regr, np_rng


if TYPE_CHECKING:
    from tests.conftest import DfOrArrays


x_col, y_col, *_ = df_regr


@pytest.mark.parametrize("log_density", [True, False])
@pytest.mark.parametrize("hist_density_kwargs", [None, {}, dict(bins=20, sort=True)])
@pytest.mark.parametrize(
    "stats",
    [False, True, dict(prefix="test", loc="lower right", prop=dict(fontsize=10))],
)
@pytest.mark.parametrize(
    "kwargs",
    [{"cmap": None}, {"cmap": "Greens"}],
)
def test_density_scatter_mpl(
    df_or_arrays: DfOrArrays,
    log_density: bool,
    hist_density_kwargs: dict[str, int | bool | str] | None,
    stats: bool | dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    df, x, y = df_or_arrays
    ax = density_scatter(
        df=df,
        x=x,
        y=y,
        log_density=log_density,
        hist_density_kwargs=hist_density_kwargs,
        stats=stats,
        **kwargs,
    )
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == x if isinstance(x, str) else "Actual"
    assert ax.get_ylabel() == y if isinstance(y, str) else "Predicted"


@pytest.mark.parametrize("stats", [1, (1,), "foo"])
def test_density_scatter_raises_on_bad_stats_type(stats: Any) -> None:
    match = f"stats must be bool or dict, got {type(stats)} instead."

    vals = [1, 2, 3]
    with pytest.raises(TypeError, match=match):
        density_scatter(x=vals, y=vals, stats=stats)


def test_density_scatter_uses_series_name_as_label() -> None:
    x = pd.Series(np_rng.random(5), name="x")
    y = pd.Series(np_rng.random(5), name="y")
    ax = density_scatter(x=x, y=y, log_density=False)

    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"


def test_density_scatter_with_hist(df_or_arrays: DfOrArrays) -> None:
    df, x, y = df_or_arrays
    density_scatter_with_hist(df=df, x=x, y=y)


@pytest.mark.parametrize(
    "cbar_label, cbar_coords",
    [
        ("foo", (0.95, 0.03, 0.03, 0.7)),
        (None, (1, 1, 1, 1)),
    ],
)
def test_density_hexbin(
    df_or_arrays: DfOrArrays,
    cbar_label: str | None,
    cbar_coords: tuple[float, float, float, float],
) -> None:
    df, x, y = df_or_arrays
    density_hexbin(df=df, x=x, y=y, cbar_label=cbar_label, cbar_coords=cbar_coords)


def test_density_hexbin_with_hist(df_or_arrays: DfOrArrays) -> None:
    df, x, y = df_or_arrays
    density_hexbin_with_hist(df=df, x=x, y=y)


def test_scatter_with_err_bar(df_or_arrays: DfOrArrays) -> None:
    df, x, y = df_or_arrays
    err = abs(df[x] - df[y]) if df is not None else abs(x - y)  # type: ignore[operator]
    scatter_with_err_bar(df=df, x=x, y=y, yerr=err)
    scatter_with_err_bar(df=df, x=x, y=y, xerr=err)


def test_residual_vs_actual(df_or_arrays: DfOrArrays) -> None:
    df, x, y = df_or_arrays
    residual_vs_actual(df=df, y_true=x, y_pred=y)


@pytest.mark.parametrize(
    "log_density, stats, bin_counts_col, n_bins, kwargs",
    [
        (True, True, "custom count col", 1, {"color_continuous_scale": "Viridis"}),
        (
            True,
            dict(prefix="test", x=1, y=1, font=dict(size=10)),
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
    fig = density_scatter_plotly(
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
    bin_counts_col = bin_counts_col or "point density"
    assert fig.layout.coloraxis.colorbar.title.text == bin_counts_col.replace(
        " ", "<br>"
    )

    colorbar = fig.layout.coloraxis.colorbar
    if log_density:
        assert all(isinstance(val, float) for val in colorbar.tickvals)
        assert all(isinstance(text, str) for text in colorbar.ticktext)
    else:
        assert colorbar.tickvals is None
        assert colorbar.ticktext is None


def test_density_scatter_plotly_hover_template() -> None:
    fig = density_scatter_plotly(df=df_regr, x=x_col, y=y_col, log_density=True)
    hover_template = fig.data[0].hovertemplate
    assert "point density" in hover_template
    assert "color" not in hover_template  # Ensure log-count values are not displayed


@pytest.mark.parametrize("stats", [1, (1,), "foo"])
def test_density_scatter_plotly_raises_on_bad_stats_type(stats: Any) -> None:
    with pytest.raises(TypeError, match="stats must be bool or dict"):
        density_scatter_plotly(df=df_regr, x=x_col, y=y_col, stats=stats)


def test_density_scatter_plotly_empty_dataframe() -> None:
    empty_df = pd.DataFrame({x_col: [], y_col: []})
    with pytest.raises(ValueError, match="Cannot cut empty array"):
        density_scatter_plotly(df=empty_df, x=x_col, y=y_col)
