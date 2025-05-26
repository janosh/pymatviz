"""Plotly 2D structure examples."""

# %%
from matminer.datasets import load_dataset
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Species

import pymatviz as pmv
from pymatviz.enums import ElemColorScheme, Key, SiteCoords


df_phonons = load_dataset("matbench_phonons")


# %% Plot Matbench phonon structures with plotly
fig = pmv.structure_2d_plotly(
    df_phonons[Key.structure].head(6).to_dict(),
    # show_cell={"edge": dict(color="white", width=1.5)},
    # show_sites=dict(line=None),
    elem_colors=ElemColorScheme.jmol,
    n_cols=3,
    subplot_title=lambda _struct, _key: dict(font=dict(color="black")),
    hover_text=lambda site: f"<b>{site.frac_coords}</b>",
)
fig.layout.paper_bgcolor = "rgba(255,255,255,0.4)"
fig.show()
# pmv.io.save_and_compress_svg(fig, "matbench-phonons-structures-2d-plotly")


# %% 2d example with supercells
supercells = {
    key: struct.make_supercell(2, in_place=False)
    for key, struct in df_phonons[Key.structure].head(6).items()
}
fig = pmv.structure_2d_plotly(
    supercells,
    elem_colors=ElemColorScheme.jmol,
    # show_cell={"edge": dict(color="white", width=1.5)},
    hover_text=SiteCoords.cartesian_fractional,
    show_bonds=True,
)
fig.show()
# pmv.io.save_and_compress_svg(fig, "matbench-phonons-structures-2d-plotly-supercells")


# %% BaTiO3 = https://materialsproject.org/materials/mp-5020
batio3 = Structure(
    lattice=Lattice.cubic(4.0338),
    species=["Ba", "Ti", "O", "O", "O"],
    coords=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
)
# Add oxidation states to help with bond determination
batio3.add_oxidation_state_by_element({"Ba": 2, "Ti": 4, "O": -2})

# Demonstrate custom legend positioning and sizing
fig = pmv.structure_2d_plotly(
    batio3, show_cell={"edge": dict(color="white", width=2)}, show_bonds=True
)
fig.show()
# pmv.io.save_and_compress_svg(fig, "bato3-structure-2d-plotly")


# %% Create a high-entropy alloy structure CoCrFeNiMn with FCC structure
lattice = Lattice.cubic(3.59)
hea_structure = Structure(
    lattice=lattice,
    species=["Co", "Cr", "Fe", "Ni", "Mn"],
    coords=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5)],
)
fig = pmv.structure_2d_plotly(
    hea_structure.make_supercell([2, 3, 2], in_place=False),
    show_cell={"edge": dict(color="white", width=2)},
)
fig.layout.title = "CoCrFeNiMn High-Entropy Alloy"
fig.show()
# pmv.io.save_and_compress_svg(fig, "hea-structure-2d-plotly")


# %% Li-ion battery cathode material with Li vacancies: Li0.8CoO2
lco_lattice = Lattice.hexagonal(2.82, 14.05)
lco_supercell = Structure(
    lattice=lco_lattice,
    species=[Species("Li", 0.8), "Co", "O", "O"],  # Partially occupied Li site
    coords=[(0, 0, 0), (0, 0, 0.5), (0, 0, 0.25), (0, 0, 0.75)],
).make_supercell([3, 3, 1])

fig = pmv.structure_2d_plotly(
    lco_supercell,
    show_cell={"edge": dict(color="white", width=1.5)},
    elem_colors=ElemColorScheme.jmol,
)
fig.layout.title = "Li0.8CoO2 with Li Vacancies"
fig.show()
# pmv.io.save_and_compress_svg(fig, "lco-structure-2d-plotly")


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

fig = pmv.structure_2d_plotly(
    structures_grid,
    n_cols=2,
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

fig.layout.title.update(text="Kitchen Sink of Customization Options", x=0.5)
fig.layout.update(width=1200, height=900, showlegend=True)

fig.show()
# pmv.io.save_and_compress_svg(fig, "structures-2x2-grid-comprehensive-options-2d")
