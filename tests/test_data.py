"""Tests for pymatviz.data: dummy regression data, demo structures and test files."""

from __future__ import annotations

import numpy as np
import pytest

import pymatviz as pmv
from pymatviz.data import load_nacl_phonons


def test_get_regression_data() -> None:
    """Test random regression arrays have correct shape and stats."""
    n_samples = 500
    y_true, y_pred, y_std = pmv.data.regression(n_samples=n_samples)

    assert y_true.shape == (n_samples,)
    assert y_pred.shape == (n_samples,)
    assert y_std.shape == (n_samples,)

    assert y_true.mean() == pytest.approx(5.0, abs=0.2)
    assert y_true.std() == pytest.approx(4.0, abs=0.2)
    # Test y_pred = 1.2 * y_true with added noise
    assert y_pred.mean() == pytest.approx(3.801761, abs=0.2)

    assert y_std.min() > 0, "All uncertainties should be positive"
    assert 2 < y_std.max() < 8, "Max uncertainty should be realistic"
    assert 1 < y_std.mean() < 3, "Mean uncertainty should be reasonable"


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
