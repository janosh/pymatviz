"""Test testing utility functions."""

from __future__ import annotations

import pytest


def test_load_phonopy_nacl() -> None:
    """Test load_phonopy_nacl utility function."""
    pytest.importorskip("phonopy")
    from pymatviz.utils.testing import load_phonopy_nacl

    phonopy_nacl = load_phonopy_nacl()
    force_constants = phonopy_nacl.force_constants

    unitcell_symbols = tuple(phonopy_nacl.unitcell.symbols)
    supercell_symbols = tuple(phonopy_nacl.supercell.symbols)
    assert set(unitcell_symbols) == {"Na", "Cl"}
    assert len(supercell_symbols) > len(unitcell_symbols)
    assert set(supercell_symbols) == {"Na", "Cl"}

    assert force_constants is not None
    assert force_constants.ndim == 4
    assert force_constants.shape[1] == len(supercell_symbols)
    assert 0 < force_constants.shape[0] <= len(unitcell_symbols)
    assert force_constants.shape[2:] == (3, 3)
