"""Phonon DOS examples."""

# %%
import json
from glob import glob
from typing import Any, cast

import plotly.graph_objects as go
from monty.io import zopen
from monty.json import MontyDecoder
from pymatgen.io.phonopy import get_pmg_structure
from pymatgen.phonon.dos import CompletePhononDos, PhononDos

import pymatviz as pmv
from pymatviz.phonons.helpers import PhononDBDoc
from pymatviz.utils.testing import TEST_FILES, load_phonopy_nacl


pmv.set_plotly_template("pymatviz_white")


class PhonopyDosMissingError(RuntimeError):
    """Raised when phonopy fails to compute a required DOS output."""


def show_figure(plotly_figure: go.Figure, title: str, *, y_pos: float = 0.97) -> None:
    """Apply consistent layout settings and display the figure."""
    plotly_figure.layout.title = dict(text=title, x=0.5, y=y_pos)
    plotly_figure.layout.margin = dict(l=0, r=0, b=0, t=40)
    plotly_figure.show()


# %% Plot phonon DOS (total)
for mp_id, formula in (
    ("mp-2758", "Sr4Se4"),
    ("mp-23907", "H2"),
):
    docs: dict[str, PhononDBDoc] = {}
    for path in glob(f"{TEST_FILES}/phonons/{mp_id}-{formula}-*.json.xz"):
        key = path.split("-", maxsplit=3)[-1].split(".")[0]
        with zopen(path, mode="rt") as file:
            docs[key] = json.loads(file.read(), cls=MontyDecoder)

    ph_doses: dict[str, PhononDos] = {key: doc.phonon_dos for key, doc in docs.items()}

    fig = pmv.phonon_dos(ph_doses)
    show_figure(fig, f"Phonon DOS of {formula} ({mp_id})", y_pos=0.98)
    # pmv.io.save_and_compress_svg(fig, f"phonon-dos-{mp_id}")


# %% plotting a phonopy TotalDos also works
try:
    import phonopy  # noqa: F401
except ImportError:
    raise SystemExit(0) from None  # install phonopy to run this script


phonopy_nacl = load_phonopy_nacl()
phonopy_nacl.run_mesh([10, 10, 10])
phonopy_nacl.run_total_dos()
if phonopy_nacl.total_dos is None:
    raise PhonopyDosMissingError

plt = phonopy_nacl.plot_total_dos()
plt.title("NaCl DOS plotted by phonopy")

fig = pmv.phonon_dos(phonopy_nacl.total_dos)
show_figure(fig, "NaCl DOS plotted by pymatviz")


# %% Element-projected phonon DOS from phonopy
# Build a CompletePhononDos from phonopy's projected DOS
phonopy_nacl_pdos = load_phonopy_nacl()
phonopy_nacl_pdos.run_mesh([10, 10, 10], with_eigenvectors=True, is_mesh_symmetry=False)
phonopy_nacl_pdos.run_projected_dos()
phonopy_nacl_pdos.run_total_dos()
if phonopy_nacl_pdos.total_dos is None:
    raise PhonopyDosMissingError
if phonopy_nacl_pdos.projected_dos is None:
    raise PhonopyDosMissingError

struct = get_pmg_structure(phonopy_nacl_pdos.primitive)
total_dos = PhononDos(
    phonopy_nacl_pdos.total_dos.frequency_points,
    phonopy_nacl_pdos.total_dos.dos,
)
site_dos = {
    site: phonopy_nacl_pdos.projected_dos.projected_dos[idx]
    for idx, site in enumerate(struct)
}
complete_dos = CompletePhononDos(struct, total_dos, site_dos)


# %% Element-projected DOS (default: with total overlay)
projected_examples: list[tuple[str, dict[str, str | bool]]] = [
    ("NaCl Element-Projected Phonon DOS", {"project": "element"}),
    (
        "NaCl Element-Projected Phonon DOS (no total)",
        {"project": "element", "show_total": False},
    ),
    (
        "NaCl Element-Projected Phonon DOS (stacked)",
        {"project": "element", "stack": True, "show_total": False},
    ),
    ("NaCl Site-Projected Phonon DOS", {"project": "site"}),
    (
        "NaCl Element-Projected Phonon DOS (normalized)",
        {"project": "element", "normalize": "max"},
    ),
]
for plot_title, plot_kwargs in projected_examples:
    fig = pmv.phonon_dos(complete_dos, **cast("dict[str, Any]", plot_kwargs))
    show_figure(fig, plot_title)

# pmv.io.save_and_compress_svg(fig, "phonon-dos-element-projected")
# pmv.io.save_and_compress_svg(fig, "phonon-dos-site-projected")


# %% Comparing multiple models with element projection
dos_dict = {"model A": complete_dos, "model B": complete_dos}
fig = pmv.phonon_dos(dos_dict, project="element")
show_figure(fig, "NaCl Multi-Model Element-Projected DOS")
# pmv.io.save_and_compress_svg(fig, "phonon-dos-multi-model-element-projected")
