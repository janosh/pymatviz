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

df_gap[Key.n_sites] = df_gap[Key.structure].map(len)
df_gap[Key.volume] = df_gap[Key.structure].map(lambda cryst: cryst.volume)
df_gap[Key.vol_per_atom] = df_gap[Key.volume] / df_gap[Key.n_sites]
df_gap[Key.formula] = df_gap[Key.structure].map(lambda cryst: cryst.formula)


# %%
# PBE band gap histogram of the Matbench MP band gap dataset
widget = pmv.histogram(df_gap[gap_col], bins=50, log_y=True)
widget.show()
# widget.to_img("pbe-gap-hist.pdf")


# %%
title = "Elemental prevalence in the Matbench MP band gap dataset"
pmv.ptable_heatmap(df_gap[Key.formula], log=True, color_bar={"title": title}).show()


# %%
# Volume per atom histogram of the Matbench MP band gap dataset
widget = pmv.histogram(df_gap[Key.vol_per_atom], bins=50, log_y=True)
widget.show()
# widget.to_img("volume-per-atom-hist.pdf")
