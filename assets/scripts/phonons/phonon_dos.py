"""Phonon DOS examples."""

# /// script
# dependencies = [
#     "monty>=2025.1.9",
# ]
# ///

# %%
import json
from glob import glob
from typing import Any

from monty.io import zopen
from monty.json import MontyDecoder

import pymatviz as pmv
from pymatviz.data import TEST_FILES, load_nacl_phonons
from pymatviz.phonons import PhononDBDoc


# %% Overlay the total phonon DOS of DFT and ML force fields
for mp_id, formula in (("mp-2758", "Sr4Se4"), ("mp-23907", "H2")):
    docs: dict[str, PhononDBDoc] = {}
    for path in glob(f"{TEST_FILES}/phonons/{mp_id}-{formula}-*.json.xz"):
        key = path.split("-", maxsplit=3)[-1].split(".")[0]
        with zopen(path, mode="rt") as file:
            docs[key] = json.loads(file.read(), cls=MontyDecoder)

    ph_doses = {key: doc.phonon_dos for key, doc in docs.items()}
    widget = pmv.phonon_dos(ph_doses, sigma=0.05, style="height: 400px")
    widget.show()
    if mp_id == "mp-2758":  # in readme
        widget.to_img(
            f"{pmv.utils.ROOT}/assets/svg/phonon-dos-{mp_id}.svg",
            width=800,
            transparent=True,
        )


# %% Total and element- or site-projected DOS of kemio phonons (from phonopy files)
nacl_dos = load_nacl_phonons().projected_dos([10, 10, 10])
pmv.phonon_dos(nacl_dos).show()
projected_examples: list[dict[str, Any]] = [
    {"project": "element"},
    {"project": "element", "show_total": False, "stack": True},
    {"project": "site"},
    {"project": "element", "normalize": "max"},
]
for plot_kwargs in projected_examples:
    pmv.phonon_dos(nacl_dos, **plot_kwargs).show()


# %% Compare multiple models with element projection
pmv.phonon_dos({"model A": nacl_dos, "model B": nacl_dos}, project="element").show()
