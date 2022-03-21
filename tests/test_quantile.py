import pytest
from matplotlib.axes import Axes

from pymatviz import qq_gaussian
from pymatviz.utils import NumArray

from .conftest import xs, y_pred


@pytest.mark.parametrize("y_std", [xs, {"foo": xs, "bar": 0.1 * xs}])
def test_qq_gaussian(y_std: NumArray | dict[str, NumArray]) -> None:
    ax = qq_gaussian(xs, y_pred, y_std)

    assert isinstance(ax, Axes)
