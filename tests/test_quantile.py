from ml_matrics import qq_gaussian

from ._helpers import xs, y_pred


def test_qq_gaussian():
    qq_gaussian(xs, y_pred, xs)

    qq_gaussian(xs, y_pred, {"foo": xs, "bar": 0.1 * xs})
