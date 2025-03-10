# %%
from matminer.datasets import load_dataset
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Species

import pymatviz as pmv
from pymatviz.enums import ElemColorScheme, Key, SiteCoords


df_phonons = load_dataset("matbench_phonons")


# %% 3d example
supercells = {
    key: struct.make_supercell(2, in_place=False)
    for key, struct in df_phonons[Key.structure].head(6).items()
}
fig = pmv.structure_3d_plotly(
    supercells,
    elem_colors=ElemColorScheme.jmol,
    # show_unit_cell={"edge": dict(color="white", width=1.5)},
    hover_text=SiteCoords.cartesian_fractional,
    show_bonds=True,
)
fig.show()
# pmv.io.save_and_compress_svg(fig, "matbench-phonons-structures-3d-plotly")


# %% BaTiO3 = https://materialsproject.org/materials/mp-5020
batio3 = Structure(
    lattice=Lattice.cubic(4.0338),
    species=["Ba", "Ti", "O", "O", "O"],
    coords=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
)
# Add oxidation states to help with bond determination
batio3.add_oxidation_state_by_element({"Ba": 2, "Ti": 4, "O": -2})

fig = pmv.structure_3d_plotly(
    batio3,
    show_unit_cell={"edge": dict(color="white", width=2)},
    show_bonds=True,
)
fig.show()
# pmv.io.save_and_compress_svg(fig, "bato3-structure-3d-plotly")


# %% Create a high-entropy alloy structure CoCrFeNiMn with FCC structure
lattice = Lattice.cubic(3.59)
hea_structure = Structure(
    lattice=lattice,
    species=["Co", "Cr", "Fe", "Ni", "Mn"],
    coords=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5)],
)
fig = pmv.structure_3d_plotly(
    hea_structure.make_supercell([2, 3, 2], in_place=False),
    show_unit_cell={"edge": dict(color="white", width=2)},
)
title = "CoCrFeNiMn High-Entropy Alloy"
fig.layout.title = title
fig.show()
pmv.io.save_and_compress_svg(fig, "hea-structure-3d-plotly")


# %% Li-ion battery cathode material with Li vacancies: Li0.8CoO2
lco_lattice = Lattice.hexagonal(2.82, 14.05)
lco_supercell = Structure(
    lattice=lco_lattice,
    species=[Species("Li", 0.8), "Co", "O", "O"],  # Partially occupied Li site
    coords=[(0, 0, 0), (0, 0, 0.5), (0, 0, 0.25), (0, 0, 0.75)],
).make_supercell([3, 3, 1])

fig = pmv.structure_3d_plotly(
    lco_supercell,
    show_unit_cell={"edge": dict(color="white", width=1.5)},
    elem_colors=ElemColorScheme.jmol,
    site_labels="symbol",
)
title = "Li0.8CoO2 with Li Vacancies"
fig.layout.title = title
fig.show()
pmv.io.save_and_compress_svg(fig, "lco-structure-3d-plotly")
