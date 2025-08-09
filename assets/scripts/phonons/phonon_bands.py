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


try:
    import atomate2  # noqa: F401
except ImportError:
    raise SystemExit(0) from None  # need atomate2 for MontyDecoder to load PhononDBDoc


pmv.set_plotly_template("pymatviz_white")


# %% Plot phonon bands and DOS
for mp_id, formula in (
    ("mp-2758", "Sr4Se4"),
    ("mp-23907", "H2"),
):
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
        with zopen(path) as file:
            docs[model_label] = json.loads(file.read(), cls=MontyDecoder)

    ph_bands: dict[str, PhononBands] = {
        key: doc.phonon_bandstructure for key, doc in docs.items()
    }

    acoustic_lines: dict[str, str | float] = {"width": 1.5}
    optical_lines: dict[str, str | float] = {"width": 1}
    if len(ph_bands) == 1:
        acoustic_lines |= dict(dash="dash", color="red", name="Acoustic")
        optical_lines |= dict(dash="dot", color="blue", name="Optical")

    fig = pmv.phonon_bands(
        ph_bands, line_kwargs=dict(acoustic=acoustic_lines, optical=optical_lines)
    )
    fig.layout.title = dict(text=f"{formula} ({mp_id}) Phonon Bands", x=0.5, y=0.98)
    fig.layout.margin = dict(l=0, r=0, b=0, t=40)
    fig.show()
    # pmv.io.save_and_compress_svg(fig, f"phonon-bands-{mp_id}")


# %% phonon bands
try:
    import phonopy
except ImportError:
    raise SystemExit(0) from None  # install phonopy to run this script

phonopy_nacl: phonopy.Phonopy = phonopy.load(
    phonopy_yaml=f"{TEST_FILES}/phonons/NaCl/phonopy_disp.yaml.xz",
    force_sets_filename=f"{TEST_FILES}/phonons/NaCl/force_sets.dat",
)
phonopy_nacl.run_mesh([10, 10, 10])
bands = {
    "L -> Γ": np.linspace([1 / 3, 1 / 3, 0], [0, 0, 0], 51),
    "Γ -> X": np.linspace([0, 0, 0], [1 / 2, 0, 0], 51),
    "X -> K": np.linspace([1 / 2, 0, 0], [1 / 2, 1 / 2, 0], 51),
    "K -> Γ": np.linspace([1 / 2, 1 / 2, 0], [0, 0, 0], 51),
    "Γ -> W": np.linspace([0, 0, 0], [1 / 3, 1 / 3, 1 / 2], 51),
}
phonopy_nacl.run_band_structure(paths=bands.values())

fig = pmv.phonon_bands({"NaCl (phonopy)": phonopy_nacl.band_structure})
fig.show()
