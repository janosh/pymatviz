from ml_matrics import qq_gaussian

from . import xs, y_pred


def test_std_calibration_single():
    qq_gaussian(xs, y_pred, xs)


def test_std_calibration_with_dict():
    qq_gaussian(xs, y_pred, {"foo": xs, "bar": 0.1 * xs})
