# %%
import json
from glob import glob

from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
from pymatgen.phonon.dos import PhononDos

import pymatviz as pmv
from pymatviz.phonons import PhononDBDoc
from pymatviz.utils.testing import TEST_FILES


try:
    import atomate2  # noqa: F401
except ImportError:
    raise SystemExit(0) from None  # need atomate2 for MontyDecoder to load PhononDBDoc


# %% Plot phonon bands and DOS
for mp_id, formula in (
    ("mp-2758", "Sr4Se4"),
    ("mp-23907", "H2"),
):
    docs: dict[str, PhononDBDoc] = {}
    for path in glob(f"{TEST_FILES}/phonons/{mp_id}-{formula}-*.json.xz"):
        key = path.split("-")[-1].split(".")[0]
        with zopen(path) as file:
            docs[key] = json.loads(file.read(), cls=MontyDecoder)

    ph_bands: dict[str, PhononBands] = {
        key: doc.phonon_bandstructure for key, doc in docs.items()
    }
    ph_doses: dict[str, PhononDos] = {key: doc.phonon_dos for key, doc in docs.items()}

    fig = pmv.phonon_bands_and_dos(ph_bands, ph_doses)
    fig.layout.title = dict(
        text=f"Phonon Bands and DOS of {formula} ({mp_id})", x=0.5, y=0.98
    )
    fig.layout.margin = dict(l=0, r=0, b=0, t=40)
    fig.show()
    pmv.io.save_and_compress_svg(fig, f"phonon-bands-and-dos-{mp_id}")
