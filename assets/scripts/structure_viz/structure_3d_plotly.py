"""Plotly 3D structure examples."""

# %%
from matminer.datasets import load_dataset
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import DummySpecies, Species

import pymatviz as pmv
from pymatviz.enums import ElemColorScheme, Key, SiteCoords


df_phonons = load_dataset("matbench_phonons")


# %% 3d example
n_structs = 6
supercells = {
    key: struct.make_supercell(2, in_place=False)
    for key, struct in df_phonons[Key.structure].head(n_structs).items()
}
fig = pmv.structure_3d_plotly(
    supercells,
    elem_colors=ElemColorScheme.jmol,
    # show_cell={"edge": dict(color="white", width=1.5)},
    hover_text=SiteCoords.cartesian_fractional,
    show_bonds=True,
)
fig.layout.title = f"{n_structs} Matbench phonon structures (3D supercells)"
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

# Demonstrate custom legend positioning and sizing
fig = pmv.structure_3d_plotly(
    batio3, show_cell={"edge": dict(color="white", width=2)}, show_bonds=True
)
fig.show()
# pmv.io.save_and_compress_svg(fig, "bato3-structure-3d-plotly")


# %% Example: Disordered site rendering (multiple spheres in 3D)
# Create structures with disordered sites to demonstrate multiple sphere rendering
disordered_lattice = Lattice.cubic(4.0)
disordered_struct = Structure(
    lattice=disordered_lattice,
    species=[
        "Ba",  # Pure Ba site
        {"Ti": 0.6, "Zr": 0.4},  # Disordered Ti/Zr site
        "O",
        "O",
        "O",  # Pure O sites
    ],
    coords=[
        (0, 0, 0),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0),
        (0.5, 0, 0.5),
        (0, 0.5, 0.5),
    ],
)

# Li-ion cathode with vacancy disorder
lico2_disordered = Structure(
    lattice=Lattice.hexagonal(2.82, 14.05),
    species=[
        {"Li": 0.75, DummySpecies("X"): 0.25},  # Li site with vacancies (X = vacancy)
        "Co",
        "O",
        "O",
    ],
    coords=[
        (0, 0, 0),
        (0, 0, 0.5),
        (0, 0, 0.25),
        (0, 0, 0.75),
    ],
)

# High-entropy alloy with ternary disorder (3 species on one site)
hea_disordered = Structure(
    lattice=Lattice.cubic(3.6),
    species=[
        {"Fe": 0.5, "Ni": 0.3, "Cr": 0.2},  # Ternary disordered site
        "Al",  # Pure Al site
    ],
    coords=[
        (0, 0, 0),
        (0.5, 0.5, 0.5),
    ],
)

# Complex multisite disorder - solid solution with multiple disordered sites
solid_solution = Structure(
    lattice=Lattice.cubic(5.0),
    species=[
        {"Ca": 0.6, "Sr": 0.4},  # Disordered A-site
        {"Ti": 0.8, "Zr": 0.15, "Hf": 0.05},  # Ternary disordered B-site
        "O",  # Pure O sites
        "O",
        "O",
    ],
    coords=[
        (0, 0, 0),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0),
        (0.5, 0, 0.5),
        (0, 0.5, 0.5),
    ],
)

disordered_3d_structures = {
    "BaTi₀.₆Zr₀.₄O₃": disordered_struct,
    "Li₀.₇₅CoO₂ (with vacancies)": lico2_disordered,
    "Fe₀.₅Ni₀.₃Cr₀.₂Al": hea_disordered,
    "Ca₀.₆Sr₀.₄Ti₀.₈Zr₀.₁₅Hf₀.₀₅O₃": solid_solution,
}

fig = pmv.structure_3d_plotly(
    disordered_3d_structures,
    elem_colors=ElemColorScheme.jmol,
    n_cols=2,
    show_cell={"edge": dict(color="darkgray", width=2)},
    site_labels="symbol",
    hover_text=SiteCoords.cartesian_fractional,
)

fig.layout.title = dict(
    text="3D Disordered Sites: Spherical Wedges Show Species Occupancy",
    x=0.5,
    font=dict(size=16),
)
# fig.layout.update(width=1200, height=800)
fig.show()
# pmv.io.save_and_compress_svg(fig, "disordered-sites-3d-plotly-spherical-wedges")


# %% Create a high-entropy alloy structure CoCrFeNiMn with FCC structure
lattice = Lattice.cubic(3.59)
hea_structure = Structure(
    lattice=lattice,
    species=["Co", "Cr", "Fe", "Ni", "Mn"],
    coords=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5)],
)
fig = pmv.structure_3d_plotly(
    hea_structure.make_supercell([2, 3, 2], in_place=False),
    show_cell={"edge": dict(color="white", width=2)},
)
title = "CoCrFeNiMn High-Entropy Alloy"
fig.layout.title = title
fig.show()
# pmv.io.save_and_compress_svg(fig, "hea-structure-3d-plotly")


# %% Li-ion battery cathode material with Li vacancies: Li0.8CoO2
lco_lattice = Lattice.hexagonal(2.82, 14.05)
lco_supercell = Structure(
    lattice=lco_lattice,
    species=[Species("Li", 0.8), "Co", "O", "O"],  # Partially occupied Li site
    coords=[(0, 0, 0), (0, 0, 0.5), (0, 0, 0.25), (0, 0, 0.75)],
).make_supercell([3, 3, 1])

fig = pmv.structure_3d_plotly(
    lco_supercell,
    show_cell={"edge": dict(color="white", width=1.5)},
    elem_colors=ElemColorScheme.jmol,
)
title = "Li0.8CoO2 with Li Vacancies"
fig.layout.title = title
fig.show()
# pmv.io.save_and_compress_svg(fig, "lco-structure-3d-plotly")


# %% 2x2 Grid showcasing multiple customization options
# Structure 1: Diamond cubic silicon with vdW color scheme
si_diamond = Structure(
    lattice=Lattice.cubic(5.43),
    species=["Si"] * 8,
    coords=[
        (0, 0, 0),
        (0.25, 0.25, 0.25),
        (0.5, 0.5, 0),
        (0.75, 0.75, 0.25),
        (0.5, 0, 0.5),
        (0.75, 0.25, 0.75),
        (0, 0.5, 0.5),
        (0.25, 0.75, 0.75),
    ],
)

# Structure 2: Perovskite CaTiO3 with CPK colors
catio3 = Structure(
    lattice=Lattice.cubic(3.84),
    species=["Ca", "Ti", "O", "O", "O"],
    coords=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
)
catio3.add_oxidation_state_by_element({"Ca": 2, "Ti": 4, "O": -2})

# Structure 3: Zinc blende ZnS with VESTA colors
zns = Structure(
    lattice=Lattice.cubic(5.41),
    species=["Zn", "S", "Zn", "S"],
    coords=[(0, 0, 0), (0.25, 0.25, 0.25), (0.5, 0.5, 0), (0.75, 0.75, 0.25)],
).make_supercell([2, 2, 1])

# Structure 4: Layered MoS2 with accessible colors
mos2_lattice = Lattice.hexagonal(3.16, 12.30)
mos2 = Structure(
    lattice=mos2_lattice,
    species=["Mo", "S", "S"],
    coords=[(0, 0, 0.5), (0.333, 0.667, 0.375), (0.333, 0.667, 0.625)],
).make_supercell([2, 2, 2])

structures_grid = {
    "Si Diamond (Jmol colors)": si_diamond,
    "CaTiO₃ Perovskite (VESTA colors)": catio3,
    "ZnS Zinc Blende (Alloy colors)": zns,
    "MoS₂ Layered (Pastel colors)": mos2,
}

fig = pmv.structure_3d_plotly(
    structures_grid,
    elem_colors={  # different color schemes to showcase variety
        "Si Diamond (Jmol colors)": ElemColorScheme.jmol,
        "CaTiO₃ Perovskite (VESTA colors)": ElemColorScheme.vesta,
        "ZnS Zinc Blende (Alloy colors)": ElemColorScheme.alloy,
        "MoS₂ Layered (Pastel colors)": ElemColorScheme.pastel,
    },
    # Show cells with different styling for each subplot
    show_cell={
        "Si Diamond (Jmol colors)": {"edge": dict(color="red", width=2.5)},
        "CaTiO₃ Perovskite (VESTA colors)": {"edge": dict(color="blue", width=2)},
        "ZnS Zinc Blende (Alloy colors)": {"edge": dict(color="green", width=1.5)},
        "MoS₂ Layered (Pastel colors)": {"edge": dict(color="purple", width=3)},
    },
    hover_text=SiteCoords.cartesian_fractional,
    show_bonds={  # Show bonds for some structures but not others
        "Si Diamond (Jmol colors)": True,
        "CaTiO₃ Perovskite (VESTA colors)": True,
        "ZnS Zinc Blende (Alloy colors)": False,
        "MoS₂ Layered (Pastel colors)": True,
    },
)

fig.update_layout(
    title=dict(text="Kitchen Sink of Customization Options", x=0.5),
    showlegend=True,
    width=1200,
    height=900,
)

fig.show()
# pmv.io.save_and_compress_svg(fig, "structures-2x2-grid-comprehensive-options")


# %% Simple cubic structure to show effect of cell_boundary_tol
LiF_cubic = Structure(
    lattice=Lattice.cubic(3.0),
    species=["Li", "F"],
    coords=[(0, 0, 0), (0.5, 0.5, 0.5)],
)

# Show same structure with different cell_boundary_tol values
cell_boundary_tols = {"Strict boundaries (tol=0)": 0, "Loose (tol=0.5)": 0.5}
fig = pmv.structure_3d_plotly(
    dict.fromkeys(cell_boundary_tols, LiF_cubic)
    | {"Via properties": LiF_cubic.copy(properties={"cell_boundary_tol": 1})},
    cell_boundary_tol=cell_boundary_tols,
    show_image_sites=True,
    show_sites=True,
    site_labels="symbol",
)

title = "Effect of cell_boundary_tol on Image Site Rendering"
subtitle = "Higher tolerance values include more atoms outside the unit cell"
fig.layout.title = dict(text=f"{title}<br><sub>{subtitle}</sub>", x=0.5, font_size=16)
fig.layout.update(height=600, width=800, margin_t=50)

fig.show()
