"""Test testing utility functions."""

from __future__ import annotations

import numpy as np

from pymatviz.utils.testing import load_nacl_phonons


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
