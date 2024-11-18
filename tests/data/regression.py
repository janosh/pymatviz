"""Test regression data generation."""

import pytest

from pymatviz.data.regression import n_samples, y_pred, y_std, y_true


def test_regression_data_shape() -> None:
    """Test random regression arrays have correct shape and stats."""
    assert y_true.shape == (n_samples,)
    assert y_pred.shape == (n_samples,)
    assert y_std.shape == (n_samples,)

    # Test y_true generated from normal(5, 4)
    assert y_true.mean() == pytest.approx(5.0, abs=0.2)
    assert y_true.std() == pytest.approx(4.0, abs=0.2)
    # Test y_pred = 1.2 * y_true with added noise
    assert y_pred.mean() == pytest.approx(6.0, abs=0.2)

    # Test y_std is non-zero and finite
    assert 10 < y_std.max() < 15
    assert y_std.mean() == pytest.approx(0.0, abs=0.2)
