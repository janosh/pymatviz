from __future__ import annotations

import json
import lzma
from dataclasses import fields, replace

import kemio
import numpy as np
import pytest

from pymatviz.phonons.helpers import PhononDBDoc
from pymatviz.utils.testing import TEST_FILES


def test_phonon_db_doc_deserialization() -> None:
    """Decode database fields to kemio objects (equal to the raw JSON) without
    changing dataclass construction or replacement.
    """
    with lzma.open(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-pbe.json.xz", "rt") as source:
        raw = json.load(source)
    original_init = PhononDBDoc.__init__
    doc = PhononDBDoc.from_dict(raw)
    assert isinstance(doc, PhononDBDoc)
    assert isinstance(doc.structure, kemio.Structure)
    assert isinstance(doc.phonon_bandstructure, kemio.PhononBands)
    assert isinstance(doc.phonon_dos, kemio.PhononDos)
    np.testing.assert_array_equal(
        doc.structure.positions, [site["xyz"] for site in raw["structure"]["sites"]]
    )
    np.testing.assert_array_equal(
        doc.phonon_bandstructure.frequencies.T, raw["phonon_bandstructure"]["bands"]
    )
    np.testing.assert_array_equal(
        doc.phonon_dos.total_dos, raw["phonon_dos"]["densities"]
    )
    assert doc.supercell is not None
    assert np.shape(doc.supercell) == (3, 3)
    assert doc.free_energies == raw["free_energies"]
    assert not hasattr(doc, "builder_meta")
    assert replace(doc, formula="SrSe").formula == "SrSe"
    assert doc.formula is None
    assert PhononDBDoc.__init__ is original_init
    kwargs = {field.name: getattr(doc, field.name) for field in fields(doc)}
    assert PhononDBDoc(**kwargs).structure is doc.structure
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        PhononDBDoc(**kwargs, unknown=True)  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError, match="required positional arguments"):
        PhononDBDoc.from_dict({})
