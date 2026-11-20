"""Interactive 3D structure examples."""

# %%
from IPython.display import display
from matminer.datasets import load_dataset
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Species

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.structure import disordered_demo_structures


df_phonons = load_dataset("matbench_phonons")


# %% Grid of 2x2x2 supercells of the first 6 Matbench phonon structures
supercells = {
    key: struct.make_supercell(2, in_place=False)
    for key, struct in df_phonons[Key.structure].head(6).items()
}
display(pmv.structure_3d(supercells, show_bonds=True, color_scheme="Jmol"))


# %% BaTiO3 = https://materialsproject.org/materials/mp-5020
batio3 = Structure(
    lattice=Lattice.cubic(4.0338),
    species=["Ba", "Ti", "O", "O", "O"],
    coords=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
)
batio3.add_oxidation_state_by_element({"Ba": 2, "Ti": 4, "O": -2})
display(pmv.structure_3d(batio3, show_bonds=True, cell_edge_color="white"))


# %% Disordered sites are drawn as spheres split by species occupancy
display(pmv.structure_3d(disordered_demo_structures, n_cols=2, show_site_labels=True))


# %% High-entropy alloy CoCrFeNiMn with FCC structure
hea_structure = Structure(
    lattice=Lattice.cubic(3.59),
    species=["Co", "Cr", "Fe", "Ni", "Mn"],
    coords=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5)],
)
display(pmv.structure_3d(hea_structure.make_supercell([2, 3, 2], in_place=False)))


# %% Li-ion battery cathode material with Li vacancies: Li0.8CoO2
lco_supercell = Structure(
    lattice=Lattice.hexagonal(2.82, 14.05),
    species=[Species("Li", 0.8), "Co", "O", "O"],  # Partially occupied Li site
    coords=[(0, 0, 0), (0, 0, 0.5), (0, 0, 0.25), (0, 0, 0.75)],
).make_supercell([3, 3, 1])
display(pmv.structure_3d(lco_supercell, color_scheme="Jmol"))


# %% Conventional standard cell of a structure with negative fractional coordinates
shifted = Structure(Lattice.cubic(4), ["Fe", "O"], [[-0.1, 0, 0], [0.5, 0.5, 0.5]])
display(pmv.structure_3d(shifted, standardize_struct=True, show_image_atoms=False))
