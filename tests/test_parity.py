from __future__ import annotations

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
from tests.conftest import y_pred, y_true


df = pd.util.testing.makeMixedDataFrame()
data = [[None, y_true, y_pred], [df, *df.columns[:2]]]


@pytest.mark.parametrize("log", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("cmap", [None, "Greens"])
@pytest.mark.parametrize("df, x, y", data)
def test_density_scatter(
    df: pd.DataFrame, x: str, y: str, log: bool, sort: bool, cmap: str | None
) -> None:
    density_scatter(df=df, x=x, y=y, log=log, sort=sort, cmap=cmap)


@pytest.mark.parametrize("df, x, y", data)
def test_density_scatter_with_hist(df: pd.DataFrame, x: str, y: str) -> None:
    density_scatter_with_hist(df=df, x=x, y=y)


@pytest.mark.parametrize("df, x, y", data)
def test_density_hexbin(df: pd.DataFrame, x: str, y: str) -> None:
    density_hexbin(df=df, x=x, y=y)


@pytest.mark.parametrize("df, x, y", data)
def test_density_hexbin_with_hist(df: pd.DataFrame, x: str, y: str) -> None:
    density_hexbin_with_hist(df=df, x=x, y=y)


@pytest.mark.parametrize("df, x, y", data)
def test_scatter_with_err_bar(df: pd.DataFrame, x: str, y: str) -> None:
    if df is not None:
        err = abs(df[x] - df[y])
    else:
        err = abs(x - y)  # type: ignore[operator]
    scatter_with_err_bar(df=df, x=x, y=y, yerr=err)
    scatter_with_err_bar(df=df, x=x, y=y, xerr=err)


@pytest.mark.parametrize("df, x, y", data)
def test_residual_vs_actual(df: pd.DataFrame, x: str, y: str) -> None:
    residual_vs_actual(df=df, y_true=x, y_pred=y)
