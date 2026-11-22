"""Tests for pymatviz.data: dummy regression data, demo structures and test files."""

from __future__ import annotations

import numpy as np
import pytest

import pymatviz as pmv
from pymatviz.data import load_nacl_phonons


def test_get_regression_data() -> None:
    """Regression arrays are seeded, y_pred = 1.2 y_true - 2 + unit-variance noise
    and uncertainties are positive.
    """
    n_samples = 500
    y_true, y_pred, y_std = pmv.data.regression(n_samples=n_samples)
    for arr in (y_true, y_pred, y_std):
        assert arr.shape == (n_samples,)
    # same seed, same data; another seed, other data
    np.testing.assert_array_equal(pmv.data.regression(n_samples=n_samples).y_std, y_std)
    assert not np.allclose(
        pmv.data.regression(n_samples=n_samples, seed=1).y_true, y_true
    )

    # sample stats within ~4 standard errors (4 / sqrt(500) = 0.18) of N(5, 4)
    assert y_true.mean() == pytest.approx(5.0, abs=0.75)
    assert y_true.std() == pytest.approx(4.0, abs=0.5)
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    assert slope == pytest.approx(1.2, abs=0.05)
    assert intercept == pytest.approx(-2, abs=0.3)
    assert np.std(y_pred - (1.2 * y_true - 2)) == pytest.approx(1, abs=0.15)
    # y_std = 0.5 + 0.3 |residual| + 0.8 Exp(1) >= 0.5 + 0.3 |residual|
    assert np.all(y_std >= 0.5 + 0.3 * np.abs(y_true - y_pred))


def test_load_nacl_phonons() -> None:
    """NaCl phonons have a 2-atom primitive cell, 64-atom supercell force constants
    and at Γ near-zero acoustic (unsymmetrized force constants) and optic modes.
    """
    phonons = load_nacl_phonons()
    assert phonons.primitive.symbols == ["Na", "Cl"]
    assert np.shape(phonons.compact_force_constants) == (2, 64, 3, 3)
    (gamma,) = phonons.frequencies([[0, 0, 0]])
    assert np.abs(gamma[:3]).max() < 0.1
    assert gamma[3:].min() > 1
