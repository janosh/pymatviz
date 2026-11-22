"""pymatviz imports and builds every widget with pymatgen, ASE, phonopy, matminer and
monty blocked from import, so none of them creeps back in as a hard dependency.
"""

from __future__ import annotations

import subprocess
import sys

from pymatviz.utils.testing import TEST_FILES


OPTIONAL_DEPS = ("pymatgen", "ase", "phonopy", "matminer", "monty", "moyopy")

SCRIPT = f"""
import sys

for name in {OPTIONAL_DEPS!r}:
    sys.modules[name] = None  # any import of these raises ImportError

import json
import lzma

import matscikit as mk
import numpy as np

import pymatviz as pmv
from pymatviz.notebook import create_widget
from pymatviz.phonons import PhononDBDoc
from pymatviz.utils.testing import fe3co4_disordered, load_nacl_phonons

nacl = mk.Structure(
    np.eye(3) * 5.64, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]], fractional=True
)
structures = {{"NaCl": nacl, "Fe3CO4": fe3co4_disordered}}
phonons = load_nacl_phonons()
nacl_dos = phonons.projected_dos([4, 4, 4])
nacl_bands = phonons.bands([[[0, 0, 0], [0.5, 0, 0]]], labels=["G", "X"])
vasprun = "{TEST_FILES}/vasprun/MgO.vasprun.xml.xz"
hull = mk.PhaseDiagram(
    [mk.Composition(formula) for formula in ("Li", "O", "Li2O")], [-1.9, -4.2, -4.8]
)

widgets = [
    pmv.structure(structure=nacl),
    pmv.trajectory(trajectory=[nacl, nacl]),
    pmv.brillouin_zone(structure=nacl),
    pmv.rdf_plot(structures, mode="full"),
    pmv.rdf_plot(nacl, element_pairs=[("Na", "Cl")]),
    pmv.coordination_bar_plot(structures),
    pmv.coordination_bar_plot(nacl, strategy="crystal_nn"),
    pmv.coordination_vs_cutoff_line(structures),
    pmv.chem_env_hierarchy(nacl, kind="treemap"),
    pmv.chem_sys_hierarchy(["Fe2O3", mk.Composition("NaCl"), nacl]),
    pmv.spacegroup_sunburst([nacl, nacl]),
    pmv.spacegroup_bar_plot([nacl, nacl]),
    pmv.spacegroup_bar_plot(["Fm-3m", "P2_1/c"]),
    pmv.xrd_plot(structures),
    pmv.phonon_dos(nacl_dos, project="element"),
    pmv.band_structure(band_structs=nacl_bands),
    pmv.band_structure(band_structs=vasprun),
    pmv.dos(doses=vasprun),
    pmv.bands_and_dos(band_structs=vasprun, doses=vasprun),
    pmv.convex_hull(entries=hull),
    pmv.composition(composition=mk.Composition("Fe2O3")),
    pmv.ptable_heatmap(pmv.count_elements(["Fe2O3", mk.Composition("NaCl")])),
    pmv.cluster_compositions(
        __import__("pandas").DataFrame({{"composition": ["Fe2O3", "NaCl", "LiCoO2"]}}),
        embedding_method="magpie",
        projection="pca",
    ),
]
for obj in (nacl, mk.Composition("NaCl"), nacl_dos, nacl_bands, hull):
    widgets.append(create_widget(obj))
with lzma.open("{TEST_FILES}/phonons/mp-2758-Sr4Se4-pbe.json.xz", "rt") as file:
    doc = PhononDBDoc.from_dict(json.load(file))
bands, dos = doc.phonon_bandstructure, doc.phonon_dos
widgets.append(pmv.bands_and_dos(band_structs=bands, doses=dos))
for widget in widgets:
    widget._repr_mimebundle_()  # syncs every trait as the frontend would get it
loaded = [
    name for name in sys.modules
    if name.split(".")[0] in {OPTIONAL_DEPS!r} and sys.modules[name] is not None
]
assert not loaded, loaded
print(len(widgets))
"""


def test_widgets_without_optional_deps() -> None:
    """Every structure, composition, phonon, XRD and hull widget builds and serializes
    with pymatgen, ASE, phonopy, matminer, monty and moyopy unimportable.
    """
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.split()[-1]) >= 25
