from mlmatrics import std_calibration

from . import xs, y_pred


def test_std_calibration_single():
    std_calibration(xs, y_pred, xs)


def test_std_calibration_with_dict():
    std_calibration(xs, y_pred, {"foo": xs, "bar": 0.1 * xs})
