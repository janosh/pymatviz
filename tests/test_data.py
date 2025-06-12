"""Test regression data generation."""

import pytest

import pymatviz as pmv


def test_get_regression_data() -> None:
    """Test random regression arrays have correct shape and stats."""
    n_samples = 500
    y_true, y_pred, y_std = pmv.data.regression(n_samples=n_samples)

    assert y_true.shape == (n_samples,)
    assert y_pred.shape == (n_samples,)
    assert y_std.shape == (n_samples,)

    # Test y_true generated from normal(5, 4)
    assert y_true.mean() == pytest.approx(5.0, abs=0.2)
    assert y_true.std() == pytest.approx(4.0, abs=0.2)
    # Test y_pred = 1.2 * y_true with added noise
    assert y_pred.mean() == pytest.approx(3.801761, abs=0.2)

    # Test y_std is positive and realistic
    assert y_std.min() > 0, "All uncertainties should be positive"
    assert 2 < y_std.max() < 8, "Max uncertainty should be realistic"
    assert 1 < y_std.mean() < 3, "Mean uncertainty should be reasonable"
