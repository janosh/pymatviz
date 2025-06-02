"""Plotly 2D structure examples."""

# %%
import os

from matminer.datasets import load_dataset
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Species

import pymatviz as pmv
from pymatviz.enums import ElemColorScheme, Key, SiteCoords
from pymatviz.utils.testing import TEST_FILES


df_phonons = load_dataset("matbench_phonons")


# %% Plot Matbench phonon structures with plotly (12 structures with bonds)
n_structs = 12
fig = pmv.structure_2d_plotly(
    df_phonons[Key.structure].iloc[:n_structs].to_dict(),
    show_bonds=True,
    elem_colors=ElemColorScheme.jmol,
    n_cols=4,
    subplot_title=lambda _struct, _key: dict(font=dict(color="black")),
    hover_text=SiteCoords.cartesian_fractional,
)
fig.layout.title = f"{n_structs} Matbench phonon structures"
fig.layout.paper_bgcolor = "rgba(255,255,255,0.4)"
fig.show()
# pmv.io.save_and_compress_svg(fig, "matbench-phonons-structures-2d-plotly")


# %% 2D plots of disordered structures
struct_mp_ids = ("mp-19017", "mp-12712")
structure_dir = f"{TEST_FILES}/structures"

os.makedirs(structure_dir, exist_ok=True)
for mp_id in struct_mp_ids:
    struct_file = f"{structure_dir}/{mp_id}.json.gz"
    if not os.path.isfile(struct_file):
        if os.getenv("CI"):
            raise FileNotFoundError(
                f"structure for {mp_id} not found, run this script locally to fetch it."
            )
        from mp_api.client import MPRester

        struct: Structure = MPRester().get_structure_by_material_id(
            mp_id, conventional_unit_cell=True
        )
        struct.to_file(struct_file)

    else:
        struct = Structure.from_file(f"{structure_dir}/{mp_id}.json.gz")

    for site in struct:  # disorder structures in-place
        if "Fe" in site.species:
            site.species = {"Fe": 0.4, "C": 0.4, "Mn": 0.2}
        elif "Zr" in site.species:
            site.species = {"Zr": 0.5, "Hf": 0.5}

    fig = pmv.structure_2d_plotly(struct)
    spg_num = struct.get_space_group_info()[1]

    formula = struct.formula.replace(" ", "")
    text = f"{formula}<br>disordered {mp_id}, spg_num = {spg_num}"
    href = f"https://materialsproject.org/materials/{mp_id}"

    # Add title with link
    fig.layout.title = dict(
        text=f'<a href="{href}" target="_blank">{text}</a>',
        x=0.5,
        font=dict(size=16, color="black"),
    )

    fig.layout.update(width=800, height=800)
    fig.show()
    img_name = f"struct-2d-plotly-{mp_id}-{formula}-disordered"
    # pmv.io.save_and_compress_svg(fig, img_name)


# %% Example: Disordered site rendering (pie slices in 2D)
# Create a test structure with disordered Ti/Zr site to demonstrate pie slice rendering
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

# Additional disordered structures for comparison
high_entropy_alloy = Structure(
    lattice=Lattice.cubic(3.6),
    species=[
        {"Fe": 0.25, "Co": 0.25, "Ni": 0.25, "Cr": 0.25},  # Fully disordered site
        {"Fe": 0.8, "Mn": 0.2},  # Partially disordered site
    ],
    coords=[(0, 0, 0), (0.5, 0.5, 0.5)],
)

disordered_structures = {
    "BaTi₀.₆Zr₀.₄O₃ (Perovskite)": disordered_struct,
    "FeCoNiCr HEA": high_entropy_alloy,
}

fig = pmv.structure_2d_plotly(
    disordered_structures,
    elem_colors=ElemColorScheme.jmol,
    n_cols=2,
    show_cell={"edge": dict(color="darkgray", width=2)},
    site_labels="symbol",
    hover_text=SiteCoords.cartesian_fractional,
)

fig.layout.title = dict(
    text="Disordered Site Rendering: Pie Slices Show Species Occupancy",
    x=0.5,
    font=dict(size=16),
)
fig.layout.update(width=800, height=400)
fig.show()
# pmv.io.save_and_compress_svg(fig, "disordered-sites-2d-plotly-pie-slices")


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
