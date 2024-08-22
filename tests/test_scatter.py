from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest

from pymatviz.scatter import (
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
df_tips = px.data.tips()


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
    [("foo", (0.95, 0.03, 0.03, 0.7)), (None, (1, 1, 1, 1))],
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
    bin_counts_col = bin_counts_col or "Point Density"
    colorbar = fig.layout.coloraxis.colorbar
    assert colorbar.title.text.replace("<br>", " ") == bin_counts_col

    if log_density:
        assert all(isinstance(val, float) for val in colorbar.tickvals)
        assert all(isinstance(text, str) for text in colorbar.ticktext)
    else:
        assert colorbar.tickvals is None
        assert colorbar.ticktext is None


def test_density_scatter_plotly_hover_template() -> None:
    fig = density_scatter_plotly(df=df_regr, x=x_col, y=y_col, log_density=True)
    hover_template = fig.data[0].hovertemplate
    assert "Point Density" in hover_template
    assert "color" not in hover_template  # Ensure log-count values are not displayed


@pytest.mark.parametrize("stats", [1, (1,), "foo"])
def test_density_scatter_plotly_raises_on_bad_stats_type(stats: Any) -> None:
    with pytest.raises(TypeError, match="stats must be bool or dict"):
        density_scatter_plotly(df=df_regr, x=x_col, y=y_col, stats=stats)


def test_density_scatter_plotly_empty_dataframe() -> None:
    empty_df = pd.DataFrame({x_col: [], y_col: []})
    with pytest.raises(ValueError, match="input should have multiple elements"):
        density_scatter_plotly(df=empty_df, x=x_col, y=y_col)


def test_density_scatter_plotly_facet() -> None:
    fig = density_scatter_plotly(
        df=df_tips, x="total_bill", y="tip", facet_col="smoker"
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Two traces for smoker/non-smoker
    assert fig.layout.xaxis2 is not None  # Check second x-axis exists for faceting


def test_density_scatter_plotly_facet_log_density() -> None:
    fig = density_scatter_plotly(
        df=df_tips, x="total_bill", y="tip", facet_col="smoker", log_density=True
    )

    assert fig.layout.coloraxis.colorbar.ticktext is not None
    assert fig.layout.coloraxis.colorbar.tickvals is not None


def test_density_scatter_plotly_facet_stats() -> None:
    fig = density_scatter_plotly(
        df=df_tips, x="total_bill", y="tip", facet_col="smoker", stats=True
    )

    # Check there are at least 2 annotations (could be more due to facet labels)
    assert len(fig.layout.annotations) >= 2
    # Check the stat annotations are present
    stat_annotations = [ann for ann in fig.layout.annotations if "MAE" in ann.text]
    assert len(stat_annotations) == 2  # One for each facet


def test_density_scatter_plotly_facet_best_fit_line() -> None:
    fig = density_scatter_plotly(
        df=df_tips, x="total_bill", y="tip", facet_col="smoker", best_fit_line=True
    )

    # Check there are at least 4 shapes (2 identity lines, 2 best fit lines)
    assert len(fig.layout.shapes) == 4
    # Check the best fit lines are present with color navy
    best_fit_lines = [
        shape for shape in fig.layout.shapes if shape.line.color == "navy"
    ]
    assert len(best_fit_lines) == 2  # One for each facet


def test_density_scatter_plotly_facet_custom_bins() -> None:
    fig = density_scatter_plotly(
        df=df_tips, x="total_bill", y="tip", facet_col="smoker", n_bins=10
    )

    # Check that binning has been applied (number of points should be reduced)
    smoker_count = df_tips["smoker"].value_counts()
    assert len(fig.data[0].x) < smoker_count["No"]
    assert len(fig.data[1].x) < smoker_count["Yes"]


def test_density_scatter_plotly_facet_custom_color() -> None:
    fig = density_scatter_plotly(
        df=df_tips,
        x="total_bill",
        y="tip",
        facet_col="smoker",
        color_continuous_scale="Viridis",
    )

    # Check the colorscale is Viridis
    assert [
        color for _val, color in fig.layout.coloraxis.colorscale
    ] == px.colors.sequential.Viridis


@pytest.mark.parametrize("density", ["kde", "empirical"])
def test_density_scatter_plotly_facet_density_methods(
    density: Literal["kde", "empirical"],
) -> None:
    fig = density_scatter_plotly(
        df=df_tips, x="total_bill", y="tip", facet_col="smoker", density=density
    )

    assert isinstance(fig, go.Figure)
    # TODO maybe add asserts to check specific aspects of KDE vs empirical density


def test_density_scatter_plotly_facet_size() -> None:
    fig = density_scatter_plotly(
        df=df_tips, x="total_bill", y="tip", size="size", facet_col="smoker"
    )

    assert "marker.size" in fig.data[0]
    assert "marker.size" in fig.data[1]


def test_density_scatter_plotly_facet_multiple_categories() -> None:
    fig = density_scatter_plotly(df=df_tips, x="total_bill", y="tip", facet_col="day")

    assert len(fig.data) == df_tips["day"].nunique()


def test_density_scatter_plotly_facet_identity_line() -> None:
    fig = density_scatter_plotly(
        df=df_tips, x="total_bill", y="tip", facet_col="smoker", identity_line=True
    )

    assert len(fig.layout.shapes) == 2  # Two identity lines, one for each facet


def test_density_scatter_plotly_facet_hover_template() -> None:
    fig = density_scatter_plotly(
        df=df_tips, x="total_bill", y="tip", facet_col="smoker"
    )

    for trace in fig.data:
        assert "total_bill" in trace.hovertemplate
        assert "tip" in trace.hovertemplate
