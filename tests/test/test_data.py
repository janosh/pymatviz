"""Test regression data generation."""

import pytest

from pymatviz.test.data import get_regression_data


def test_get_regression_data() -> None:
    """Test random regression arrays have correct shape and stats."""
    n_samples = 500
    data = get_regression_data(n_samples)

    assert data.y_true.shape == (n_samples,)
    assert data.y_pred.shape == (n_samples,)
    assert data.y_std.shape == (n_samples,)

    # Test y_true generated from normal(5, 4)
    assert data.y_true.mean() == pytest.approx(5.0, abs=0.2)
    assert data.y_true.std() == pytest.approx(4.0, abs=0.2)
    # Test y_pred = 1.2 * y_true with added noise
    assert data.y_pred.mean() == pytest.approx(6.0, abs=0.2)

    # Test y_std is non-zero and finite
    assert 10 < data.y_std.max() < 15
    assert data.y_std.mean() == pytest.approx(0.0, abs=0.2)
