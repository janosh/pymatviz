from __future__ import annotations

import pytest
from matplotlib.axes import Axes

from pymatviz import err_decay
from pymatviz.utils import NumArray

from .conftest import y_pred, y_true


y_std_mock = y_true - y_pred


@pytest.mark.parametrize("y_std", [y_std_mock, {"y_std_mock": y_std_mock}])
@pytest.mark.parametrize("n_rand", [10, 100, 1000])
@pytest.mark.parametrize("percentiles", [True, False])
def test_err_decay(
    y_std: NumArray | dict[str, NumArray], n_rand: int, percentiles: bool
) -> None:
    ax = err_decay(y_true, y_pred, y_std, n_rand=n_rand, percentiles=percentiles)

    assert isinstance(ax, Axes)
