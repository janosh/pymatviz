# %%
import json
from glob import glob

from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.phonon.dos import PhononDos

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.utils.testing import TEST_FILES


# %% Plot phonon bands and DOS
for mp_id, formula in (
    ("mp-2758", "Sr4Se4"),
    ("mp-23907", "H2"),
):
    docs = {}
    for path in glob(f"{TEST_FILES}/phonons/{mp_id}-{formula}-*.json.lzma"):
        key = path.split("-", maxsplit=3)[-1].split(".")[0]
        with zopen(path) as file:
            docs[key] = json.loads(file.read(), cls=MontyDecoder)

    ph_doses: dict[str, PhononDos] = {
        key: getattr(doc, Key.ph_dos) for key, doc in docs.items()
    }

    fig = pmv.phonon_dos(ph_doses)
    fig.layout.title = dict(text=f"Phonon DOS of {formula} ({mp_id})", x=0.5, y=0.98)
    fig.layout.margin = dict(l=0, r=0, b=0, t=40)
    fig.show()
    # pmv.io.save_and_compress_svg(fig, f"phonon-dos-{mp_id}")


# %% plot phonopy TotalDos
try:
    import phonopy
except ImportError:
    raise SystemExit(0) from None  # install phonopy to run this script

phonopy_nacl = phonopy.load(
    phonopy_yaml=f"{TEST_FILES}/phonons/NaCl/phonopy_disp.yaml.xz",
    force_sets_filename=f"{TEST_FILES}/phonons/NaCl/force_sets.dat",
)

phonopy_nacl.run_mesh([10, 10, 10])
phonopy_nacl.run_total_dos()
phonopy_nacl.plot_total_dos()

fig = pmv.phonon_dos(phonopy_nacl.total_dos)
fig.show()
