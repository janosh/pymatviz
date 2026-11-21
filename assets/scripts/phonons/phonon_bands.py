"""Phonon bands examples."""

# %%
import json
from glob import glob

import numpy as np
from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands

import pymatviz as pmv
from pymatviz.phonons import PhononDBDoc
from pymatviz.utils.testing import TEST_FILES


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

    ph_bands: dict[str, PhononBands] = {
        key: doc.phonon_bandstructure for key, doc in docs.items()
    }
    widget = pmv.BandStructureWidget(band_structs=ph_bands, style="height: 450px")
    widget.show()
    if mp_id == "mp-2758":  # in readme
        widget.to_img(
            f"{pmv.utils.ROOT}/assets/svg/phonon-bands-{mp_id}.svg",
            width=800,
            transparent=True,
        )


# %% phonopy band structures work too
try:
    import phonopy
except ImportError:
    raise SystemExit(0) from None  # install phonopy to run this script

phonopy_nacl: phonopy.Phonopy = phonopy.load(
    phonopy_yaml=f"{TEST_FILES}/phonons/NaCl/phonopy_disp.yaml.xz",
    force_sets_filename=f"{TEST_FILES}/phonons/NaCl/force_sets.dat",
)
phonopy_nacl.run_mesh([10, 10, 10])
q_paths = {
    "L -> Γ": np.linspace([1 / 3, 1 / 3, 0], [0, 0, 0], 51),
    "Γ -> X": np.linspace([0, 0, 0], [1 / 2, 0, 0], 51),
    "X -> K": np.linspace([1 / 2, 0, 0], [1 / 2, 1 / 2, 0], 51),
    "K -> Γ": np.linspace([1 / 2, 1 / 2, 0], [0, 0, 0], 51),
    "Γ -> W": np.linspace([0, 0, 0], [1 / 3, 1 / 3, 1 / 2], 51),
}
phonopy_nacl.run_band_structure(paths=list(q_paths.values()))
if phonopy_nacl.band_structure is None:
    raise RuntimeError("phonopy band structure not computed")

pmv.BandStructureWidget(
    band_structs={"NaCl (phonopy)": phonopy_nacl.band_structure}
).show()
