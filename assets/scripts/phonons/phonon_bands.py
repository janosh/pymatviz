# %%
import json
from glob import glob

from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands

import pymatviz as pmv
from pymatviz.enums import Key


# TODO: ffonons not working properly (see #195)
try:
    import ffonons  # noqa: F401
except ImportError:
    raise SystemExit(0) from None  # install ffonons to run this script


# %% Plot phonon bands and DOS
for mp_id, formula in (
    ("mp-2758", "Sr4Se4"),
    ("mp-23907", "H2"),
):
    docs = {}
    for path in glob(f"{pmv.utils.TEST_FILES}/phonons/{mp_id}-{formula}-*.json.lzma"):
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
        key: getattr(doc, Key.ph_band_structure) for key, doc in docs.items()
    }

    acoustic_lines: dict[str, str | float] = dict(width=1.5)
    optical_lines: dict[str, str | float] = dict(width=1)
    if len(ph_bands) == 1:
        acoustic_lines |= dict(dash="dash", color="red", name="Acoustic")
        optical_lines |= dict(dash="dot", color="blue", name="Optical")

    fig = pmv.phonon_bands(
        ph_bands, line_kwargs=dict(acoustic=acoustic_lines, optical=optical_lines)
    )
    fig.layout.title = dict(text=f"{formula} ({mp_id}) Phonon Bands", x=0.5, y=0.98)
    fig.layout.margin = dict(l=0, r=0, b=0, t=40)
    fig.show()
    pmv.io.save_and_compress_svg(fig, f"phonon-bands-{mp_id}")
