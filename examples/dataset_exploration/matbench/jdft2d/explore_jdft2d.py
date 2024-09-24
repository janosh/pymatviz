"""Stats for the matbench_jdft2d dataset.

Input: Pymatgen Structure of the material.
Target variable: Exfoliation energy (meV).
Entries: 636

Matbench v0.1 dataset for predicting exfoliation energies from crystal structure
(computed with the OptB88vdW and TBmBJ functionals). Adapted from the JARVIS DFT DB.


https://ml.materialsproject.org/projects/matbench_jdft2d
"""

# %%
from __future__ import annotations

from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv
from pymatviz import count_elements, ptable_heatmap, spacegroup_bar, spacegroup_sunburst
from pymatviz.enums import Key


# %%
df_2d = load_dataset("matbench_jdft2d")

df_2d[[Key.spg_symbol, Key.spg_num]] = [
    struct.get_space_group_info() for struct in tqdm(df_2d[Key.structure])
]

df_2d.describe()


# %%
ax = df_2d.hist(column="exfoliation_en", bins=50, log=True)
pmv.save_fig(ax, "jdft2d-exfoliation-energy-hist.pdf")


# %%
df_2d[Key.volume] = [x.volume for x in df_2d[Key.structure]]
df_2d[Key.formula] = [x.formula for x in df_2d[Key.structure]]

fig = ptable_heatmap(count_elements(df_2d[Key.formula]), log=True, return_type="figure")
fig.suptitle("Elemental prevalence in the Matbench Jarvis DFT 2D dataset")
pmv.save_fig(fig, "jdft2d-ptable-heatmap.pdf")


# %%
ax = spacegroup_bar(df_2d[Key.spg_num], log=True)
pmv.save_fig(ax, "jdft2d-spacegroup-hist.pdf")


# %%
fig = spacegroup_sunburst(df_2d[Key.spg_num], show_counts="percent")
fig.layout.title = "Spacegroup sunburst of the JARVIS DFT 2D dataset"
fig.write_image("jdft2d-spacegroup-sunburst.pdf")
fig.show()
