from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from pymatviz import error_decay_with_uncert, qq_gaussian
from pymatviz.utils import Array

from .conftest import xs, y_pred, y_true


y_std_mock = y_true - y_pred


@pytest.mark.parametrize("y_std", [y_std_mock, {"y_std_mock": y_std_mock}])
@pytest.mark.parametrize("n_rand", [10, 100, 1000])
@pytest.mark.parametrize("percentiles", [True, False])
def test_error_decay_with_uncert(
    y_std: Array | dict[str, Array], n_rand: int, percentiles: bool
) -> None:
    ax = error_decay_with_uncert(
        y_true, y_pred, y_std, n_rand=n_rand, percentiles=percentiles
    )

    assert isinstance(ax, plt.Axes)
    # assert ax.get_title() == "Error Decay"
    assert ax.get_ylabel() == "MAE"
    if percentiles:
        assert ax.get_xlabel() == "Confidence percentiles"

        assert ax.get_xlim() == pytest.approx((0, 100), abs=5)

    else:
        assert ax.get_xlabel() == "Excluded samples"
        assert ax.get_xlim() == pytest.approx((0, 100), abs=5)


@pytest.mark.parametrize("y_std", [xs, {"foo": xs, "bar": 0.1 * xs}])
@pytest.mark.parametrize("ax", [None, plt.gca()])
def test_qq_gaussian(y_std: Array | dict[str, Array], ax: plt.Axes) -> None:
    ax = qq_gaussian(xs, y_pred, y_std, ax=ax)

    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "Theoretical Quantile"
    assert ax.get_ylabel() == "Observed Quantile"
    assert ax.get_xlim() == ax.get_ylim() == (0, 1)
