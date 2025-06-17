"""Test testing utility functions."""

from __future__ import annotations

import pytest


def test_load_phonopy_nacl() -> None:
    """Test load_phonopy_nacl utility function."""
    pytest.importorskip("phonopy")
    from pymatviz.utils.testing import load_phonopy_nacl

    # Load phonopy object
    phonopy_nacl = load_phonopy_nacl()

    # Check that we got a proper Phonopy object
    assert hasattr(phonopy_nacl, "unitcell")
    assert hasattr(phonopy_nacl, "supercell")
    assert hasattr(phonopy_nacl, "force_constants")

    # Check that it's NaCl structure
    symbols = phonopy_nacl.unitcell.symbols
    assert "Na" in symbols
    assert "Cl" in symbols

    # Check that force constants are available
    assert phonopy_nacl.force_constants is not None
