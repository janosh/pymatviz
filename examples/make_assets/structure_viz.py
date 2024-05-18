# %%
import matplotlib.pyplot as plt
from matminer.datasets import load_dataset
from mp_api.client import MPRester
from pymatgen.core import Structure

from pymatviz.enums import Key
from pymatviz.io import save_and_compress_svg
from pymatviz.structure_viz import plot_structure_2d


struct: Structure  # for type hinting

df_steels = load_dataset("matbench_steels")
df_phonons = load_dataset("matbench_phonons")


# %% Plot Matbench phonon structures
n_rows, n_cols = 3, 4
fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
title = f"{len(axs.flat)} Matbench phonon structures"
fig.suptitle(title, fontweight="bold", fontsize=20)

for idx, (row, ax) in enumerate(zip(df_phonons.itertuples(), axs.flat), start=1):
    struct = row[Key.structure]
    spg_num = struct.get_space_group_info()[1]
    struct.add_oxidation_state_by_guess()

    plot_structure_2d(
        struct,
        ax=ax,
        show_bonds=True,
        bond_kwargs=dict(facecolor="gray", linewidth=2, linestyle="dotted"),
    )
    sub_title = f"{idx + 1}. {struct.formula} ({spg_num})"
    ax.set_title(sub_title, fontweight="bold")

fig.show()
save_and_compress_svg(fig, "matbench-phonons-structures-2d")


# %% Plot some disordered structures in 2D
disordered_structs = {
    mp_id: MPRester().get_structure_by_material_id(mp_id, conventional_unit_cell=True)
    for mp_id in ["mp-19017", "mp-12712"]
}

for mp_id, struct in disordered_structs.items():
    for site in struct:  # disorder structures in-place
        if "Fe" in site.species:
            site.species = {"Fe": 0.4, "C": 0.4, "Mn": 0.2}
        elif "Zr" in site.species:
            site.species = {"Zr": 0.5, "Hf": 0.5}

    ax = plot_structure_2d(struct)
    _, spacegroup = struct.get_space_group_info()

    formula = struct.formula.replace(" ", "")
    text = f"{formula}\ndisordered {mp_id}, {spacegroup = }"
    href = f"https://materialsproject.org/materials/{mp_id}"
    ax.text(
        0.5, 1, text, url=href, ha="center", transform=ax.transAxes, fontweight="bold"
    )

    ax.figure.set_size_inches(8, 8)

    save_and_compress_svg(ax, f"struct-2d-{mp_id}-{formula}-disordered")
    plt.show()
