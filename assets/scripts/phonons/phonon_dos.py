"""Phonon DOS examples."""

# %%
import json
from glob import glob
from typing import Any

from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.io.phonopy import get_pmg_structure
from pymatgen.phonon.dos import CompletePhononDos, PhononDos

import pymatviz as pmv
from pymatviz.phonons.helpers import PhononDBDoc
from pymatviz.utils.testing import TEST_FILES, load_phonopy_nacl


# %% Overlay the total phonon DOS of DFT and ML force fields
for mp_id, formula in (("mp-2758", "Sr4Se4"), ("mp-23907", "H2")):
    docs: dict[str, PhononDBDoc] = {}
    for path in glob(f"{TEST_FILES}/phonons/{mp_id}-{formula}-*.json.xz"):
        key = path.split("-", maxsplit=3)[-1].split(".")[0]
        with zopen(path, mode="rt") as file:
            docs[key] = json.loads(file.read(), cls=MontyDecoder)

    ph_doses: dict[str, PhononDos] = {key: doc.phonon_dos for key, doc in docs.items()}
    widget = pmv.phonon_dos(ph_doses, sigma=0.05, style="height: 400px")
    widget.show()
    if mp_id == "mp-2758":  # in readme
        widget.to_img(
            f"{pmv.utils.ROOT}/assets/svg/phonon-dos-{mp_id}.svg",
            width=800,
            transparent=True,
        )


# %% phonopy TotalDos works too
try:
    import phonopy  # noqa: F401
except ImportError:
    raise SystemExit(0) from None  # install phonopy to run this script

phonopy_nacl = load_phonopy_nacl()
phonopy_nacl.run_mesh([10, 10, 10])
phonopy_nacl.run_total_dos()
if phonopy_nacl.total_dos is None:
    raise RuntimeError("phonopy total DOS not computed")
pmv.phonon_dos(phonopy_nacl.total_dos).show()


# %% Element- and site-projected DOS from a CompletePhononDos built with phonopy
phonopy_nacl.run_mesh([10, 10, 10], with_eigenvectors=True, is_mesh_symmetry=False)
phonopy_nacl.run_projected_dos()
phonopy_nacl.run_total_dos()
total_dos, projected_dos = phonopy_nacl.total_dos, phonopy_nacl.projected_dos
if total_dos is None or projected_dos is None:
    raise RuntimeError("phonopy total or projected DOS not computed")
total_densities, site_densities = total_dos.dos, projected_dos.projected_dos
if total_densities is None or site_densities is None:
    raise RuntimeError("phonopy total or projected densities not computed")

struct = get_pmg_structure(phonopy_nacl.primitive)
complete_dos = CompletePhononDos(
    struct,
    PhononDos(total_dos.frequency_points, total_densities),
    {site: site_densities[idx] for idx, site in enumerate(struct)},
)
projected_examples: list[dict[str, Any]] = [
    {"project": "element"},
    {"project": "element", "show_total": False, "stack": True},
    {"project": "site"},
    {"project": "element", "normalize": "max"},
]
for plot_kwargs in projected_examples:
    pmv.phonon_dos(complete_dos, **plot_kwargs).show()


# %% Compare multiple models with element projection
pmv.phonon_dos(
    {"model A": complete_dos, "model B": complete_dos}, project="element"
).show()
