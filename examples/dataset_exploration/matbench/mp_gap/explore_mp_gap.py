"""Stats for the matbench_mp_gap dataset.

Input: Pymatgen Structure of the material.
Target variable: The band gap (E_g) as calculated by PBE DFT from the Materials Project
    in eV.
Entries: 106113

https://ml.materialsproject.org/projects/matbench_mp_gap
"""

# %%
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


# %%
df_gap = load_dataset("matbench_mp_gap")
gap_col = "gap pbe"


# %%
fig = df_gap[gap_col].hist(bins=50, log_y=True, backend="plotly")
title = "PBE band gap histogram of the Matbench MP band gap dataset"
fig.layout.title.update(text=title, x=0.5)
fig.layout.showlegend = False
fig.show()
# pmv.save_fig(ax, "pbe_gap_hist.pdf")


# %%
df_gap[Key.n_sites] = df_gap[Key.structure].map(len)
df_gap[Key.volume] = df_gap[Key.structure].map(lambda cryst: cryst.volume)
df_gap[Key.vol_per_atom] = df_gap[Key.volume] / df_gap[Key.n_sites]
df_gap[Key.formula] = df_gap[Key.structure].map(lambda cryst: cryst.formula)

fig = pmv.ptable_heatmap_plotly(df_gap[Key.formula], log=True)
fig.layout.title.update(text="Elemental prevalence in the Matbench MP band gap dataset")
fig.show()
# pmv.save_fig(fig, "mp_gap-ptable-heatmap.pdf")


# %%
fig = df_gap[Key.vol_per_atom].hist(bins=50, log_y=True, backend="plotly")
title = "Volume per atom histogram of the Matbench MP band gap dataset"
fig.layout.title.update(text=title, x=0.5)
fig.layout.showlegend = False
fig.show()
# pmv.save_fig(ax, "volume_per_atom_hist.pdf")
