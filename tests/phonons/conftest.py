from __future__ import annotations

import json
from glob import glob

import pytest
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
from pymatgen.phonon.dos import PhononDos

from pymatviz.utils.testing import TEST_FILES


BandsDoses = dict[str, dict[str, PhononBands | PhononDos]]
bs_key, dos_key = "phonon_bandstructure", "phonon_dos"
# enable loading PhononDBDocParsed with @module set to uninstalled ffonons.dbs.phonondb
# by changing to identical dataclass in pymatviz.phonons module
MSONable.REDIRECT["ffonons.dbs.phonondb"] = {
    "PhononDBDocParsed": {
        "@class": "PhononDBDoc",
        "@module": "pymatviz.phonons.helpers",
    }
}
MSONable.REDIRECT["atomate2.common.schemas.phonons"] = {
    "PhononBSDOSDoc": {"@class": "PhononDBDoc", "@module": "pymatviz.phonons.helpers"}
}


@pytest.fixture
def phonon_bands_doses_mp_2758() -> BandsDoses:
    with zopen(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-pbe.json.xz") as file:
        dft_doc = json.loads(file.read(), cls=MontyDecoder)

    with zopen(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-mace-y7uhwpje.json.xz") as file:
        ml_doc = json.loads(file.read(), cls=MontyDecoder)

    bands = {"DFT": getattr(dft_doc, bs_key), "MACE": getattr(ml_doc, bs_key)}
    doses = {"DFT": getattr(dft_doc, dos_key), "MACE": getattr(ml_doc, dos_key)}
    return {"bands": bands, "doses": doses}


@pytest.fixture
def phonon_bands_doses_mp_2667() -> BandsDoses:
    # with zopen(f"{TEST_FILES}/phonons/mp-2691-Cd4Se4-pbe.json.xz") as file:
    with zopen(f"{TEST_FILES}/phonons/mp-2667-Cs1Au1-pbe.json.xz") as file:
        return json.loads(file.read(), cls=MontyDecoder)


@pytest.fixture
def phonon_doses() -> dict[str, PhononDos]:
    paths = glob(f"{TEST_FILES}/phonons/mp-*-pbe.json.xz")
    assert len(paths) >= 2
    return {
        path.split("/")[-1].split("-pbe")[0]: getattr(
            json.loads(zopen(path).read(), cls=MontyDecoder), dos_key
        )
        for path in paths
    }
