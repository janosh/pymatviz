"""Stats for the matbench_jdft2d dataset.

Input: Pymatgen Structure of the material.
Target variable: Exfoliation energy (meV).
Entries: 636

Matbench v0.1 dataset for predicting exfoliation energies from crystal structure
(computed with the OptB88vdW and TBmBJ functionals). Adapted from the JARVIS DFT DB.


https://ml.materialsproject.org/projects/matbench_jdft2d
"""

# %%
from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.enums import Key


# %%
df_2d = load_dataset("matbench_jdft2d")

df_2d[[Key.spg_symbol, Key.spg_num]] = [
    struct.get_space_group_info() for struct in tqdm(df_2d[Key.structure])
]

df_2d.describe()


# %%
fig = df_2d["exfoliation_en"].hist(nbins=200, log_y=True, opacity=0.8, backend="plotly")
title = "<b>Exfoliation energy histogram of the Matbench Jarvis DFT 2D dataset</b>"
fig.layout.title.update(text=title, x=0.5)
fig.layout.showlegend = False
fig.show()
# pmv.save_fig(fig, "jdft2d-exfoliation-energy-hist.pdf")


# %%
df_2d[Key.volume] = [x.volume for x in df_2d[Key.structure]]
df_2d[Key.formula] = [x.formula for x in df_2d[Key.structure]]

fig = pmv.ptable_heatmap_plotly(df_2d[Key.formula], log=True)
fig.layout.title.update(
    text="Elemental prevalence in the Matbench Jarvis DFT 2D dataset"
)
fig.show()
# pmv.save_fig(fig, "jdft2d-ptable-heatmap.pdf")


# %%
fig = pmv.spacegroup_bar(df_2d[Key.spg_num], log=True)
title = "Spacegroup histogram of the JARVIS DFT 2D dataset"
fig.layout.title.update(text=title, y=0.98)
fig.layout.margin.update(b=10, l=10, r=10, t=70)
fig.show()
# pmv.save_fig(fig, "jdft2d-spacegroup-hist.pdf")


# %%
fig = pmv.spacegroup_sunburst(df_2d[Key.spg_num], show_counts="percent")
fig.layout.title.update(text="Spacegroup sunburst of the JARVIS DFT 2D dataset", x=0.5)
fig.layout.margin.update(b=0, l=0, r=0, t=40)
fig.show()
# pmv.save_fig(fig, "jdft2d-spacegroup-sunburst.pdf")
