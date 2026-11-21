"""Interactive 3D structure examples."""

# %%
from IPython.display import display
from matminer.datasets import load_dataset
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Species

import pymatviz as pmv
from pymatviz.utils.testing import disordered_demo_structures


df_phonons = load_dataset("matbench_phonons")


def show_and_save(widget: pmv.widgets.StructureWidget, name: str) -> None:
    """Display a widget and export it as a transparent readme PNG."""
    display(widget)
    widget.to_img(
        f"{pmv.utils.ROOT}/assets/png/{name}-structure-3d.png",
        width=500,
        transparent=True,
    )


# %% Grid of 2x2x2 supercells of the first 6 Matbench phonon structures
supercells = {
    key: struct.make_supercell(2, in_place=False)
    for key, struct in df_phonons["structure"].head(6).items()
}
display(
    pmv.widget_grid(
        {
            key: pmv.structure(
                structure=struct, show_bonds="always", color_scheme="Jmol"
            )
            for key, struct in supercells.items()
        }
    )
)


# %% BaTiO3 = https://materialsproject.org/materials/mp-5020
batio3 = Structure(
    lattice=Lattice.cubic(4.0338),
    species=["Ba", "Ti", "O", "O", "O"],
    coords=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
)
batio3.add_oxidation_state_by_element({"Ba": 2, "Ti": 4, "O": -2})
show_and_save(pmv.structure(structure=batio3, show_bonds="always"), "batio3")


# %% Disordered sites are drawn as spheres split by species occupancy
display(
    pmv.widget_grid(
        {
            name: pmv.structure(structure=struct, show_site_labels=True)
            for name, struct in disordered_demo_structures.items()
        },
        n_cols=2,
    )
)


# %% High-entropy alloy CoCrFeNiMn with FCC structure
hea_structure = Structure(
    lattice=Lattice.cubic(3.59),
    species=["Co", "Cr", "Fe", "Ni", "Mn"],
    coords=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5)],
)
show_and_save(
    pmv.structure(structure=hea_structure.make_supercell([2, 3, 2], in_place=False)),
    "hea",
)


# %% Li-ion battery cathode material with Li vacancies: Li0.8CoO2
lco_supercell = Structure(
    lattice=Lattice.hexagonal(2.82, 14.05),
    species=[Species("Li", 0.8), "Co", "O", "O"],  # Partially occupied Li site
    coords=[(0, 0, 0), (0, 0, 0.5), (0, 0, 0.25), (0, 0, 0.75)],
).make_supercell([3, 3, 1])
# the wide, flat slab leaves the viewer's default 500px height half empty
show_and_save(
    pmv.structure(
        structure=lco_supercell, color_scheme="Jmol", style="--struct-height: 380px"
    ),
    "lco",
)


# %% 2H-MoS2 = https://materialsproject.org/materials/mp-2815, layers stacked along c
mos2 = Structure(
    lattice=Lattice.hexagonal(3.16, 12.29),
    species=["Mo", "S", "S", "Mo", "S", "S"],
    coords=[(1 / 3, 2 / 3, z) for z in (0.25, 0.621, 0.879)]
    + [(2 / 3, 1 / 3, z) for z in (0.75, 0.121, 0.379)],
).make_supercell([3, 3, 1])
show_and_save(pmv.structure(structure=mos2, show_bonds="always"), "mos2")


# %% Conventional standard cell of a structure with negative fractional coordinates
shifted = Structure(Lattice.cubic(4), ["Fe", "O"], [[-0.1, 0, 0], [0.5, 0.5, 0.5]])
display(
    pmv.structure(structure=shifted, cell_type="conventional", show_image_atoms=False)
)
