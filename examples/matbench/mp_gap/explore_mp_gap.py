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


# %%
df_gap = load_dataset("matbench_mp_gap")
gap_col = "gap pbe"

df_gap["n_sites"] = df_gap["structure"].map(len)
df_gap["volume"] = df_gap["structure"].map(lambda cryst: cryst.volume)
df_gap["vol_per_atom"] = df_gap["volume"] / df_gap["n_sites"]
df_gap["formula"] = df_gap["structure"].map(lambda cryst: cryst.formula)


# %%
# PBE band gap histogram of the Matbench MP band gap dataset
widget = pmv.histogram(df_gap[gap_col], bins=50, log_y=True)
widget.show()
# widget.to_img("pbe-gap-hist.pdf")


# %%
title = "Elemental prevalence in the Matbench MP band gap dataset"
pmv.ptable_heatmap(df_gap["formula"], log=True, color_bar={"title": title}).show()


# %%
# Volume per atom histogram of the Matbench MP band gap dataset
widget = pmv.histogram(df_gap["vol_per_atom"], bins=50, log_y=True)
widget.show()
# widget.to_img("volume-per-atom-hist.pdf")
