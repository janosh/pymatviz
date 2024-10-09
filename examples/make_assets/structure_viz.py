# %%
import os
from typing import cast

import matplotlib.pyplot as plt
from matminer.datasets import load_dataset
from mp_api.client import MPRester
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.enums import ElemColorScheme, Key
from pymatviz.utils import TEST_FILES


df_steels = load_dataset("matbench_steels")
df_phonons = load_dataset("matbench_phonons")


# %% Plot Matbench phonon structures
n_structs = 12
fig, axs = pmv.structure_2d(
    df_phonons[Key.structure].iloc[:n_structs],
    show_bonds=True,
    bond_kwargs=dict(facecolor="gray", linewidth=2, linestyle="dotted"),
    elem_colors=ElemColorScheme.jmol,
)
title = f"{n_structs} Matbench phonon structures"
fig.suptitle(title, fontweight="bold", fontsize=20)
# pmv.io.save_and_compress_svg(fig, "matbench-phonons-structures-2d")
fig.show()


# %% Plot some disordered structures in 2D
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

    ax = cast(plt.Axes, pmv.structure_2d(struct))
    _, spacegroup = struct.get_space_group_info()

    formula = struct.formula.replace(" ", "")
    text = f"{formula}\ndisordered {mp_id}, {spacegroup = }"
    href = f"https://materialsproject.org/materials/{mp_id}"
    ax.text(
        0.5, 1, text, url=href, ha="center", transform=ax.transAxes, fontweight="bold"
    )

    ax.figure.set_size_inches(8, 8)

    plt.show()
    pmv.io.save_and_compress_svg(ax, f"struct-2d-{mp_id}-{formula}-disordered")


# %% Plot Matbench phonon structures with plotly
fig = pmv.structure_2d_plotly(
    df_phonons[Key.structure].head(6).to_dict(),
    # show_unit_cell={"edge": dict(color="white", width=1.5)},
    # show_sites=dict(line=None),
    elem_colors=ElemColorScheme.jmol,
    n_cols=3,
    subplot_title=lambda _struct, _key: dict(font=dict(color="black")),
    hover_text=lambda site: f"<b>{site.frac_coords}</b>",
)

fig.show()
pmv.io.save_and_compress_svg(fig, "matbench-phonons-structures-2d-plotly")


# %% 3d example
fig = pmv.structure_3d_plotly(
    df_phonons[Key.structure].head(6).to_dict(),
    elem_colors=ElemColorScheme.jmol,
    # show_unit_cell={"edge": dict(color="white", width=1.5)},
    hover_text="cartesian+fractional",
)
fig.show()
pmv.io.save_and_compress_svg(fig, "matbench-phonons-structures-3d-plotly")
