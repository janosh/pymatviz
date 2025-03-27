from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest
from sklearn.metrics import r2_score

import pymatviz as pmv
from tests.conftest import df_regr, np_rng


if TYPE_CHECKING:
    from typing import Any, Literal

    from tests.conftest import DfOrArrays


X_COL, Y_COL, *_ = df_regr
DF_TIPS = px.data.tips()


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
    ax = pmv.density_scatter(
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

    # Identity and best fit lines are added by default unless explicitly disabled
    identity_line = kwargs.get("identity_line", True)
    best_fit_line = kwargs.get("best_fit_line", True)

    if identity_line:
        # Check identity line exists (black dashed)
        identity_lines = [
            line
            for line in ax.lines
            if line.get_color() == "black" and line.get_linestyle() == "--"
        ]
        assert len(identity_lines) == 1, "Identity line not found"

    r2_val = r2_score(df[x], df[y]) if isinstance(df, pd.DataFrame) else r2_score(x, y)
    if best_fit_line and r2_val > 0.3:
        # Check best fit line exists (navy solid)
        best_fit_lines = [
            line for line in ax.lines if line.get_color() in ("navy", "lightskyblue")
        ]
        assert len(best_fit_lines) == 1, "Best fit line not found"


@pytest.mark.parametrize("stats", [1, (1,), "foo"])
def test_density_scatter_raises_on_bad_stats_type(stats: Any) -> None:
    match = f"stats must be bool or dict, got {type(stats)} instead."

    vals = [1, 2, 3]
    with pytest.raises(TypeError, match=match):
        pmv.density_scatter(x=vals, y=vals, stats=stats)


def test_density_scatter_uses_series_name_as_label() -> None:
    x = pd.Series(np_rng.random(5), name="x")
    y = pd.Series(np_rng.random(5), name="y")
    ax = pmv.density_scatter(x=x, y=y, log_density=False)

    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"


def test_density_scatter_with_hist(df_or_arrays: DfOrArrays) -> None:
    df, x, y = df_or_arrays
    pmv.density_scatter_with_hist(df=df, x=x, y=y)


@pytest.mark.parametrize(
    ("cbar_label", "cbar_coords", "gridsize"),
    [("foo", (0.95, 0.03, 0.03, 0.7), 50), (None, (1, 1, 1, 1), 100)],
)
def test_density_hexbin(
    df_or_arrays: DfOrArrays,
    cbar_label: str | None,
    cbar_coords: tuple[float, float, float, float],
    gridsize: int,
) -> None:
    df, x, y = df_or_arrays
    ax = pmv.density_hexbin(
        df=df,
        x=x,
        y=y,
        cbar_label=cbar_label,
        cbar_coords=cbar_coords,
        gridsize=gridsize,
    )
    assert isinstance(ax, plt.Axes)

    assert len(ax.collections) == 1
    hexbin = ax.collections[0]
    assert len(hexbin.get_offsets()) > 0


def test_density_hexbin_with_hist(df_or_arrays: DfOrArrays) -> None:
    df, x, y = df_or_arrays
    pmv.density_hexbin_with_hist(df=df, x=x, y=y)


def test_scatter_with_err_bar(df_or_arrays: DfOrArrays) -> None:
    df, x, y = df_or_arrays
    err = abs(df[x] - df[y]) if df is not None else abs(x - y)  # type: ignore[operator]
    pmv.scatter_with_err_bar(df=df, x=x, y=y, yerr=err)
    pmv.scatter_with_err_bar(df=df, x=x, y=y, xerr=err)


def test_residual_vs_actual(df_or_arrays: DfOrArrays) -> None:
    df, x, y = df_or_arrays
    pmv.residual_vs_actual(df=df, y_true=x, y_pred=y)


@pytest.mark.parametrize(
    ("log_density", "stats", "bin_counts_col", "n_bins", "kwargs"),
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

    # Identity and best fit lines are added by default unless explicitly disabled
    identity_line = kwargs.get("identity_line", True)

    if identity_line:
        # Check identity line exists (gray dashed)
        identity_lines = [
            shape
            for shape in fig.layout.shapes
            if shape.line.dash == "dash" and shape.line.color in ("gray", "black")
        ]
        assert len(identity_lines) == 1, "Identity line not found"


def test_density_scatter_plotly_hover_template() -> None:
    fig = pmv.density_scatter_plotly(df=df_regr, x=X_COL, y=Y_COL, log_density=True)
    hover_template = fig.data[0].hovertemplate
    assert "Point Density" in hover_template
    # With the new approach, we preserve the default hover template which includes color
    # Just ensure the bin counts are displayed correctly
    assert f"{X_COL}" in hover_template
    assert f"{Y_COL}" in hover_template
    assert "customdata[0]" in hover_template


@pytest.mark.parametrize("stats", [1, (1,), "foo"])
def test_density_scatter_plotly_raises_on_bad_stats_type(stats: Any) -> None:
    with pytest.raises(TypeError, match="stats must be bool or dict"):
        pmv.density_scatter_plotly(df=df_regr, x=X_COL, y=Y_COL, stats=stats)


def test_density_scatter_plotly_empty_dataframe() -> None:
    empty_df = pd.DataFrame({X_COL: [], Y_COL: []})
    with pytest.raises(ValueError, match="input should have multiple elements"):
        pmv.density_scatter_plotly(df=empty_df, x=X_COL, y=Y_COL)


def test_density_scatter_plotly_facet() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="smoker"
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Two traces for smoker/non-smoker
    for key, val in {
        "anchor": "x2",
        "domain": (0, 1),
        "matches": "y",
        "showticklabels": False,
    }.items():
        assert getattr(fig.layout.yaxis2, key) == val, f"{key=}, {val=}"
    assert fig.layout.xaxis2 is not None  # Check second x-axis exists for faceting
    for key, val in {"anchor": "y2", "domain": (0.51, 1.0), "matches": "x"}.items():
        assert getattr(fig.layout.xaxis2, key) == val, f"{key=}, {val=}"
    assert fig.layout.xaxis2.title.text == "total_bill"


def test_density_scatter_plotly_facet_log_density() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="smoker", log_density=True
    )

    # Check that the colorbar exists and has tick values and labels
    colorbar = fig.layout.coloraxis.colorbar
    tick_values = colorbar.tickvals
    tick_labels = colorbar.ticktext

    # Basic checks
    assert tick_values is not None
    assert len(tick_values) > 0
    assert tick_labels is not None
    assert len(tick_labels) > 0

    # Check that the tick values are in a reasonable range for log-transformed data
    # For log10(x+1) transformed data, values should be between 0 and 1 for most cases
    assert all(0 <= val <= 1 for val in tick_values), (
        "Tick values outside expected range"
    )

    # Check that the labels include both "0" and "1" (common for density plots)
    assert "0" in tick_labels, "Label '0' missing from colorbar"
    assert "1" in tick_labels, "Label '1' missing from colorbar"

    # Check that the tick values are in consistent order
    is_ascending = all(
        tick_values[idx] < tick_values[idx + 1] for idx in range(len(tick_values) - 1)
    )
    is_descending = all(
        tick_values[idx] > tick_values[idx + 1] for idx in range(len(tick_values) - 1)
    )
    assert is_ascending or is_descending, "Tick values not in consistent order"


def test_density_scatter_plotly_facet_stats() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="smoker", stats=True
    )

    # Check there are at least 2 annotations (could be more due to facet labels)
    assert len(fig.layout.annotations) >= 2
    # Check the stat annotations are present
    stat_annotations = [ann for ann in fig.layout.annotations if "MAE" in ann.text]
    assert len(stat_annotations) == 2  # One for each facet


def test_density_scatter_plotly_facet_best_fit_line() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="smoker", best_fit_line=True
    )

    # Check there are shapes (identity and best fit lines)
    assert len(fig.layout.shapes) == 3

    # Check there are annotations for the best fit lines
    best_fit_annotations = [
        anno for anno in fig.layout.annotations if "LS fit: y =" in anno.text
    ]
    # Should have one annotation per facet
    assert len(best_fit_annotations) == 2
    assert best_fit_annotations[0].font.color == "navy"


def test_density_scatter_plotly_facet_custom_bins() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="smoker", n_bins=10
    )

    # Check that binning has been applied (number of points should be reduced)
    smoker_count = DF_TIPS["smoker"].value_counts()
    assert len(fig.data[0].x) < smoker_count["No"]
    assert len(fig.data[1].x) < smoker_count["Yes"]


def test_density_scatter_plotly_facet_custom_color() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS,
        x="total_bill",
        y="tip",
        facet_col="smoker",
        color_continuous_scale="Viridis",
    )

    # Check the colorscale is Viridis
    color_scale = fig.layout.coloraxis.colorscale
    assert [color for _val, color in color_scale] == px.colors.sequential.Viridis


@pytest.mark.parametrize("density", ["kde", "empirical"])
def test_density_scatter_plotly_facet_density_methods(
    density: Literal["kde", "empirical"],
) -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="smoker", density=density
    )

    assert isinstance(fig, go.Figure)
    # TODO maybe add asserts to check specific aspects of KDE vs empirical density


def test_density_scatter_plotly_facet_size() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", size="size", facet_col="smoker"
    )

    assert "marker.size" in fig.data[0]
    assert "marker.size" in fig.data[1]


def test_density_scatter_plotly_facet_multiple_categories() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="day"
    )

    assert len(fig.data) == DF_TIPS["day"].nunique()


def test_density_scatter_plotly_facet_identity_line() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="smoker", identity_line=True
    )

    assert len(fig.layout.shapes) == 2  # Two identity lines, one for each facet


def test_density_scatter_plotly_facet_hover_template() -> None:
    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", facet_col="smoker"
    )

    for trace in fig.data:
        assert "total_bill" in trace.hovertemplate
        assert "tip" in trace.hovertemplate


def test_density_scatter_plotly_colorbar_kwargs() -> None:
    colorbar_kwargs = {"title": "Custom Title", "thickness": 30, "len": 0.8, "x": 1.1}

    fig = pmv.density_scatter_plotly(
        df=DF_TIPS, x="total_bill", y="tip", colorbar_kwargs=colorbar_kwargs
    )

    # Check that colorbar properties were applied correctly
    actual_colorbar = fig.layout.coloraxis.colorbar
    assert actual_colorbar.title.text == "Custom Title"
    assert actual_colorbar.thickness == 30
    assert actual_colorbar.len == 0.8
    assert actual_colorbar.x == 1.1


def test_colorbar_tick_labels_no_trailing_zeros() -> None:
    """Test that colorbar tick labels don't have trailing zeros."""
    # Create a dataframe with a wide range of counts to ensure various tick formats
    n_points = 1000
    df_xy = pd.DataFrame(
        {
            "x": np_rng.random(n_points),
            "y": np_rng.random(n_points),
            "count": np.logspace(0, 4, n_points),  # Values from 1 to 10000
        }
    )

    # Use the count column to simulate different bin densities
    fig = pmv.density_scatter_plotly(
        df=df_xy, x="x", y="y", log_density=True, bin_counts_col="count"
    )

    # Check that the colorbar has tick labels and none have trailing .0
    colorbar = fig.layout.coloraxis.colorbar
    assert colorbar.ticktext is not None
    assert len(colorbar.ticktext) > 0

    for label in colorbar.ticktext:
        assert not label.endswith(".0"), f"Label '{label}' has trailing .0"

        # Check for correct SI prefix usage (if applicable)
        if "k" in label and "." in label.split("k")[0]:
            decimal_part = label.split("k")[0].split(".")[1]
            assert decimal_part != "0", f"Label '{label}' has unnecessary decimal"


def test_colorbar_density_range_and_formatting() -> None:
    """Test colorbar tick formatting for wide range of density values."""
    # Create a dataset with controlled point densities using clusters
    # We'll create 5 clusters with exponentially increasing point densities
    n_clusters = 5
    points_per_cluster = np.logspace(0, 4, n_clusters).astype(
        int
    )  # 1, 10, 100, 1000, 10000 points

    # Create empty arrays for x and y
    x_vals = []
    y_vals = []

    # Create clusters with different densities at specific locations
    for idx, n_pts in enumerate(points_per_cluster):
        # Create a tight cluster of points at position (i, i)
        cluster_x = np_rng.normal(idx, 0.05, size=n_pts)
        cluster_y = np_rng.normal(idx, 0.05, size=n_pts)

        x_vals.extend(cluster_x)
        y_vals.extend(cluster_y)

    # Create DataFrame with the clustered points
    df_xy = pd.DataFrame({"x": x_vals, "y": y_vals})

    # Create plot with log density and fixed number of bins to ensure consistent results
    fig = pmv.density_scatter_plotly(
        df=df_xy, x="x", y="y", log_density=True, n_bins=50
    )

    # Get colorbar properties
    colorbar = fig.layout.coloraxis.colorbar
    tick_values = colorbar.tickvals
    tick_labels = colorbar.ticktext

    # Basic checks
    assert tick_values is not None
    assert len(tick_values) > 0
    assert tick_labels is not None
    assert len(tick_labels) > 0

    # Convert tick labels to numeric values
    numeric_labels = []
    for label in tick_labels:
        # Handle SI prefixes
        if "k" in label:
            value = float(label.replace("k", "")) * 1000
        elif "M" in label:
            value = float(label.replace("M", "")) * 1000000
        else:
            try:
                value = float(label)
            except ValueError:
                continue  # Skip non-numeric labels
        numeric_labels.append(value)

    # Check that we have a sufficient range of values (at least 2 orders of magnitude)
    assert len(numeric_labels) >= 3, "Not enough numeric labels on colorbar"
    assert max(numeric_labels) / min(numeric_labels) >= 100, (
        "Colorbar range is too small"
    )

    # Check that the minimum value is 1 (or close to it)
    assert min(numeric_labels) <= 2, "Minimum colorbar value should be close to 1"

    # Check that the maximum value is at least 100
    assert max(numeric_labels) >= 100, "Maximum colorbar value should be at least 100"

    # Check formatting of labels
    for label in tick_labels:
        # No trailing zeros
        assert not label.endswith(".0"), f"Label '{label}' has trailing .0"

        # If using SI prefix with decimal, ensure decimal is meaningful
        if any(prefix in label for prefix in ["k", "M"]) and "." in label:
            prefix = "k" if "k" in label else "M"
            decimal_part = label.split(prefix)[0].split(".")[1]
            assert decimal_part != "0", f"Label '{label}' has unnecessary decimal"

    # Check that tick values are in ascending order
    assert all(
        tick_values[idx] < tick_values[idx + 1] for idx in range(len(tick_values) - 1)
    ), "Tick values not in ascending order"


def test_density_scatter_plotly_hover_template_with_custom_template() -> None:
    """Test that custom hover templates are preserved with bin counts appended."""
    # Create a custom hover template
    custom_template = "Custom: %{x}<br>Value: %{y}<extra>Additional Info</extra>"

    fig = pmv.density_scatter_plotly(
        df=df_regr, x=X_COL, y=Y_COL, log_density=True, hovertemplate=custom_template
    )

    hover_template = fig.data[0].hovertemplate
    # Check that the custom template is preserved
    assert "Custom:" in hover_template
    assert "Value:" in hover_template
    # Check that the bin counts are added
    assert "Point Density" in hover_template
    # Check that the extra tag is preserved
    assert "Additional Info" in hover_template


def test_density_scatter_plotly_hover_template_without_extra_tag() -> None:
    """Test that bin counts are correctly appended when no </extra> tag exists."""
    # Create a custom hover template without an </extra> tag
    custom_template = "Custom: %{x}<br>Value: %{y}"

    fig = pmv.density_scatter_plotly(
        df=df_regr, x=X_COL, y=Y_COL, log_density=True, hovertemplate=custom_template
    )

    hover_template = fig.data[0].hovertemplate
    # Check that the custom template is preserved
    assert "Custom:" in hover_template
    assert "Value:" in hover_template
    # Check that the bin counts are added
    assert "Point Density" in hover_template
    # Check that an </extra> tag is added
    assert hover_template.endswith("</extra>")


def test_density_scatter_plotly_hover_format() -> None:
    """Test that hover_format parameter correctly formats bin counts."""
    # Test with integer format
    fig_int = pmv.density_scatter_plotly(
        df=df_regr, x=X_COL, y=Y_COL, log_density=True, hover_format=".0f"
    )

    # Test with decimal format
    fig_decimal = pmv.density_scatter_plotly(
        df=df_regr, x=X_COL, y=Y_COL, log_density=True, hover_format=".2f"
    )

    # Check that the format specifiers are included in the hover templates
    assert ":.0f" in fig_int.data[0].hovertemplate
    assert ":.2f" in fig_decimal.data[0].hovertemplate
