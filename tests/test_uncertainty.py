from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest

from pymatviz import error_decay_with_uncert, qq_gaussian
from tests.conftest import df, df_x_y, xs, y_pred, y_true


if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from numpy.typing import ArrayLike


y_std_mock = y_true - y_pred


assert len(df_x_y) == 2
assert len(df_x_y[0]) == 3


@pytest.mark.parametrize(
    "df, x, y, y_std",
    [
        (None, y_true, y_pred, y_std_mock),
        (None, y_true, y_pred, {"y_std_mock": y_std_mock}),
        (df, *df.columns[:2], df.columns[0]),  # single std col
        (df, *df.columns[:2], df.columns[:2]),  # multiple std cols
    ],
)
@pytest.mark.parametrize("n_rand", [10, 100, 1000])
@pytest.mark.parametrize("percentiles", [True, False])
def test_error_decay_with_uncert(
    df: pd.DataFrame,
    x: ArrayLike | str,
    y: ArrayLike | str,
    y_std: ArrayLike | dict[str, ArrayLike] | str | Sequence[str],
    n_rand: int,
    percentiles: bool,
) -> None:
    ax = error_decay_with_uncert(
        x, y, y_std, df=df, n_rand=n_rand, percentiles=percentiles
    )

    assert isinstance(ax, plt.Axes)
    # assert ax.get_title() == "Error Decay"
    assert ax.get_ylabel() == "MAE"
    if percentiles:
        assert ax.get_xlabel() == "Confidence percentiles"
    else:
        assert ax.get_xlabel() == "Excluded samples"


@pytest.mark.parametrize(
    "df, x, y, y_std",
    [
        (*df_x_y[0], xs),
        (*df_x_y[0], {"foo": xs, "bar": 0.1 * xs}),
        (*df_x_y[1], df.columns[0]),
        (*df_x_y[1], df.columns[:2]),
    ],
)
@pytest.mark.parametrize("ax", [None, plt.gca()])
def test_qq_gaussian(
    df: pd.DataFrame,
    x: ArrayLike | str,
    y: ArrayLike | str,
    y_std: ArrayLike | dict[str, ArrayLike],
    ax: plt.Axes,
) -> None:
    ax = qq_gaussian(x, y, y_std, df=df, ax=ax)

    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "Theoretical Quantile"
    assert ax.get_ylabel() == "Observed Quantile"
    assert ax.get_xlim() == ax.get_ylim() == (0, 1)
