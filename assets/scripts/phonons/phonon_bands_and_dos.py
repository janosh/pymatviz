"""Phonon bands and DOS examples."""

# %%
import json
from glob import glob

from monty.io import zopen
from monty.json import MontyDecoder

import pymatviz as pmv
from pymatviz.phonons import PhononDBDoc
from pymatviz.utils.testing import TEST_FILES


# %% Bands next to DOS sharing the frequency axis, with the Brillouin zone and k-path
for mp_id, formula in (("mp-2758", "Sr4Se4"), ("mp-23907", "H2")):
    docs: dict[str, PhononDBDoc] = {}
    for path in glob(f"{TEST_FILES}/phonons/{mp_id}-{formula}-*.json.xz"):
        key = path.split("-")[-1].split(".")[0]
        with zopen(path, mode="rt") as file:
            docs[key] = json.loads(file.read(), cls=MontyDecoder)

    widget = pmv.phonon_bands_and_dos(
        {key: doc.phonon_bandstructure for key, doc in docs.items()},
        {key: doc.phonon_dos for key, doc in docs.items()},
        structure=next(iter(docs.values())).structure,
        sigma=0.05,
        style="height: 500px",
    )
    widget.show()
    # PNG since the Brillouin zone inset renders with WebGL
    widget.to_img(
        f"{pmv.ROOT}/assets/png/phonon-bands-and-dos-{mp_id}.png",
        width=1300,  # MatterViz's desktop layout: zone, bands and DOS side by side
        transparent=True,
    )
