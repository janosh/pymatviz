"""Phonon bands examples."""

# /// script
# dependencies = [
#     "monty>=2025.1.9",
# ]
# ///

import json
from glob import glob

from monty.io import zopen
from monty.json import MontyDecoder

import pymatviz as pmv
from pymatviz.data import TEST_FILES, load_nacl_phonons
from pymatviz.phonons import PhononDBDoc


# %% Overlay phonon bands of DFT and ML force fields
for mp_id, formula in (("mp-2758", "Sr4Se4"), ("mp-23907", "H2")):
    docs: dict[str, PhononDBDoc] = {}
    for path in glob(f"{TEST_FILES}/phonons/{mp_id}-{formula}-*.json.xz"):
        model_label = (
            "CHGNet"
            if "chgnet" in path
            else "MACE"
            if "mace" in path
            else "M3GNet"
            if "m3gnet" in path
            else "PBE"
        )
        with zopen(path, mode="rt") as file:
            docs[model_label] = json.loads(file.read(), cls=MontyDecoder)

    ph_bands = {key: doc.phonon_bandstructure for key, doc in docs.items()}
    widget = pmv.band_structure(band_structs=ph_bands, style="height: 450px")
    widget.show()
    if mp_id == "mp-2758":  # in readme
        widget.to_img(
            f"{pmv.utils.ROOT}/assets/svg/phonon-bands-{mp_id}.svg",
            width=800,
            transparent=True,
        )


# %% Band structure of kemio phonons (from phonopy files) along L-Γ-X-K-Γ-W
nacl_bands = load_nacl_phonons().bands(
    [
        [
            [1 / 3, 1 / 3, 0],
            [0, 0, 0],
            [1 / 2, 0, 0],
            [1 / 2, 1 / 2, 0],
            [0, 0, 0],
            [1 / 3, 1 / 3, 1 / 2],
        ]
    ],
    labels=["L", "Γ", "X", "K", "Γ", "W"],
)
pmv.band_structure(band_structs={"NaCl (kemio)": nacl_bands}).show()
