from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from pymatviz import error_decay_with_uncert, qq_gaussian
from tests.conftest import DfOrArrays, df_regr, xs, y_pred, y_true


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike


y_std_mock = y_true - y_pred


@pytest.mark.parametrize(
    "y_std",
    [
        y_std_mock,
        {"y_std_mock": y_std_mock},
        df_regr.columns[0],  # single std col
        df_regr.columns[:2],  # multiple std cols
    ],
)
@pytest.mark.parametrize("n_rand", [10, 100, 1000])
@pytest.mark.parametrize("percentiles", [True, False])
def test_error_decay_with_uncert(
    df_or_arrays: DfOrArrays,
    y_std: ArrayLike | dict[str, ArrayLike] | str | Sequence[str],
    n_rand: int,
    percentiles: bool,
) -> None:
    df, x, y = df_or_arrays
    # override y_std if col name but no df provided, would be nonsensical input
    if df is None and isinstance(y_std, (str, pd.Index)):
        y_std = y_std_mock
    ax = error_decay_with_uncert(
        x, y, y_std, df=df, n_rand=n_rand, percentiles=percentiles
    )

    assert isinstance(ax, plt.Axes)
    # check legend labels
    assert {itm.get_text() for itm in ax.get_legend().get_texts()} <= {
        "std",
        "y_std_mock",
        "error",
        "random",
        "y_pred",
        "y_true",
    }
    x_label, y_label = ax.get_xlabel(), ax.get_ylabel()
    assert y_label == "MAE"
    assert x_label == "Confidence percentiles" if percentiles else "Excluded samples"

    assert ax.get_xlim() in ((-3.95, 104.95), (-5.0, 105.0))


@pytest.mark.parametrize(
    "y_std", [xs, {"foo": xs, "bar": 0.1 * xs}, df_regr.columns[0], df_regr.columns[:2]]
)
@pytest.mark.parametrize("ax", [None, plt.gca()])
def test_qq_gaussian(
    df_or_arrays: DfOrArrays, y_std: ArrayLike | dict[str, ArrayLike], ax: plt.Axes
) -> None:
    df, x, y = df_or_arrays
    # override y_std if col name but no df provided, would be nonsensical input
    if df is None and isinstance(y_std, (str, pd.Index)):
        y_std = xs
    ax = qq_gaussian(x, y, y_std, df=df, ax=ax)

    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "Theoretical Quantile"
    assert ax.get_ylabel() == "Observed Quantile"
    assert ax.get_xlim() == ax.get_ylim() == (0, 1)
