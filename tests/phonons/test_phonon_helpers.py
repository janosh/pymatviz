from __future__ import annotations

import json
from dataclasses import fields, replace

import numpy as np
import pytest
from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.core import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
from pymatgen.phonon.dos import PhononDos

from pymatviz.phonons.helpers import PhononDBDoc
from pymatviz.utils.testing import TEST_FILES


@pytest.mark.parametrize("via_monty", [False, True])
def test_phonon_db_doc_deserialization(via_monty: bool) -> None:
    """Decode database fields to matscikit objects (equal to pymatgen's) without
    changing dataclass construction or replacement.
    """
    import matscikit as mk

    with zopen(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-pbe.json.xz", mode="rt") as source:
        raw = json.load(source)
    original_init = PhononDBDoc.__init__
    doc = (
        MontyDecoder().process_decoded(raw) if via_monty else PhononDBDoc.from_dict(raw)
    )
    assert isinstance(doc, PhononDBDoc)
    assert isinstance(doc.structure, mk.Structure)
    assert isinstance(doc.phonon_bandstructure, mk.PhononBands)
    assert isinstance(doc.phonon_dos, mk.PhononDos)
    pmg_struct = Structure.from_dict(raw["structure"])
    np.testing.assert_array_equal(doc.structure.positions, pmg_struct.cart_coords)
    pmg_bands = PhononBands.from_dict(raw["phonon_bandstructure"])
    np.testing.assert_array_equal(
        doc.phonon_bandstructure.frequencies.T, pmg_bands.bands
    )
    np.testing.assert_array_equal(
        doc.phonon_dos.total_dos, PhononDos.from_dict(raw["phonon_dos"]).densities
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
