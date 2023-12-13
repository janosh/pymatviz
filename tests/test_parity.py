from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pymatviz import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)


if TYPE_CHECKING:
    from tests.conftest import DfOrArrays


@pytest.mark.parametrize("log_cmap", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("cmap", [None, "Greens"])
@pytest.mark.parametrize(
    "stats",
    [False, True, dict(prefix="test", loc="lower right", prop=dict(fontsize=10))],
)
def test_density_scatter(
    df_or_arrays: DfOrArrays,
    log_cmap: bool,
    sort: bool,
    cmap: str | None,
    stats: bool | dict[str, Any],
) -> None:
    df, x, y = df_or_arrays
    ax = density_scatter(
        df=df, x=x, y=y, log_cmap=log_cmap, sort=sort, cmap=cmap, stats=stats
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
    x = pd.Series(np.random.rand(5), name="x")
    y = pd.Series(np.random.rand(5), name="y")
    ax = density_scatter(x=x, y=y)

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
