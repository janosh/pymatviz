from __future__ import annotations

from typing import Any

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
from pymatviz.utils import Array
from tests.conftest import df_x_y


@pytest.mark.parametrize("log_cmap", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("cmap", [None, "Greens"])
@pytest.mark.parametrize(
    "stats",
    [False, True, dict(prefix="test", loc="lower right", prop=dict(fontsize=10))],
)
@pytest.mark.parametrize("df, x, y", df_x_y)
def test_density_scatter(
    df: pd.DataFrame | None,
    x: Array | str,
    y: str | Array,
    log_cmap: bool,
    sort: bool,
    cmap: str | None,
    stats: bool | dict[str, Any],
) -> None:
    ax = density_scatter(
        df=df, x=x, y=y, log_cmap=log_cmap, sort=sort, cmap=cmap, stats=stats
    )
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == x if isinstance(x, str) else "Actual"
    assert ax.get_ylabel() == y if isinstance(y, str) else "Predicted"


def test_density_scatter_uses_series_name_as_label() -> None:
    x = pd.Series(np.random.rand(5), name="x")
    y = pd.Series(np.random.rand(5), name="y")
    ax = density_scatter(x=x, y=y)

    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"


@pytest.mark.parametrize("df, x, y", df_x_y)
def test_density_scatter_with_hist(
    df: pd.DataFrame | None, x: str | Array, y: str | Array
) -> None:
    density_scatter_with_hist(df=df, x=x, y=y)


@pytest.mark.parametrize("df, x, y", df_x_y)
def test_density_hexbin(
    df: pd.DataFrame | None, x: str | Array, y: str | Array
) -> None:
    density_hexbin(df=df, x=x, y=y)


@pytest.mark.parametrize("df, x, y", df_x_y)
def test_density_hexbin_with_hist(
    df: pd.DataFrame | None, x: str | Array, y: str | Array
) -> None:
    density_hexbin_with_hist(df=df, x=x, y=y)


@pytest.mark.parametrize("df, x, y", df_x_y)
def test_scatter_with_err_bar(
    df: pd.DataFrame | None, x: str | Array, y: str | Array
) -> None:
    err = abs(df[x] - df[y]) if df is not None else abs(x - y)  # type: ignore[operator]
    scatter_with_err_bar(df=df, x=x, y=y, yerr=err)
    scatter_with_err_bar(df=df, x=x, y=y, xerr=err)


@pytest.mark.parametrize("df, x, y", df_x_y)
def test_residual_vs_actual(
    df: pd.DataFrame | None, x: str | Array, y: str | Array
) -> None:
    residual_vs_actual(df=df, y_true=x, y_pred=y)
