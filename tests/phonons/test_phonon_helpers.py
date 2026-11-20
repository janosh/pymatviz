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
    """Decode database fields without changing dataclass construction or replacement."""
    with zopen(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-pbe.json.xz", mode="rt") as source:
        raw = json.load(source)
    original_init = PhononDBDoc.__init__
    doc = (
        MontyDecoder().process_decoded(raw) if via_monty else PhononDBDoc.from_dict(raw)
    )
    assert isinstance(doc, PhononDBDoc)
    assert isinstance(doc.structure, Structure)
    assert isinstance(doc.phonon_bandstructure, PhononBands)
    assert isinstance(doc.phonon_dos, PhononDos)
    assert doc.structure.as_dict() == Structure.from_dict(raw["structure"]).as_dict()
    np.testing.assert_array_equal(
        doc.phonon_dos.densities, raw["phonon_dos"]["densities"]
    )
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
