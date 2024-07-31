# %%
import json
from glob import glob

from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
from pymatgen.phonon.dos import PhononDos

from pymatviz.enums import Key
from pymatviz.io import save_and_compress_svg
from pymatviz.phonons import phonon_bands, phonon_bands_and_dos, phonon_dos
from pymatviz.utils import TEST_FILES


# %% Plot phonon bands and DOS
for mp_id, formula in (
    ("mp-2758", "Sr4Se4"),
    ("mp-23907", "H2"),
):
    docs = {}
    for path in glob(f"{TEST_FILES}/phonons/{mp_id}-{formula}-*.json.lzma"):
        key = path.split("-")[-1].split(".")[0]
        with zopen(path) as file:
            docs[key] = json.loads(file.read(), cls=MontyDecoder)

    ph_bands: dict[str, PhononBands] = {
        key: getattr(doc, Key.ph_band_structure) for key, doc in docs.items()
    }
    ph_doses: dict[str, PhononDos] = {
        key: getattr(doc, Key.ph_dos) for key, doc in docs.items()
    }

    fig = phonon_bands(ph_bands)
    fig.layout.title = dict(text=f"Phonon Bands of {formula} ({mp_id})", x=0.5, y=0.98)
    fig.layout.margin = dict(l=0, r=0, b=0, t=40)
    save_and_compress_svg(fig, f"phonon-bands-{mp_id}")

    fig = phonon_dos(ph_doses)
    fig.layout.title = dict(text=f"Phonon DOS of {formula} ({mp_id})", x=0.5, y=0.98)
    fig.layout.margin = dict(l=0, r=0, b=0, t=40)
    save_and_compress_svg(fig, f"phonon-dos-{mp_id}")

    fig = phonon_bands_and_dos(ph_bands, ph_doses)
    fig.layout.title = dict(
        text=f"Phonon Bands and DOS of {formula} ({mp_id})", x=0.5, y=0.98
    )
    fig.layout.margin = dict(l=0, r=0, b=0, t=40)
    save_and_compress_svg(fig, f"phonon-bands-and-dos-{mp_id}")
