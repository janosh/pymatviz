"""Stats for the matbench_phonons dataset.

Input: Pymatgen Structure of the material.
Target variable: Frequency of the highest frequency optical phonon mode peak
    in units of 1/cm; may be used as an estimation of dominant longitudinal
    optical phonon frequency.
Entries: 1265

Matbench v0.1 dataset for predicting vibration properties from crystal structure.
Original data retrieved from Petretto et al. Original calculations done via ABINIT
in the harmonic approximation based on density functional perturbation theory.
Removed entries having a formation energy (or energy above the convex hull) more
than 150meV.

https://ml.materialsproject.org/projects/matbench_phonons
"""

# %%
from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv


# %%
df_phonon = load_dataset(data_name := "matbench_phonons")
last_dos_peak = "last phdos peak"

df_phonon[["spg_symbol", "spg_num"]] = [
    struct.get_space_group_info() for struct in tqdm(df_phonon["structure"])
]


# %%
# Last phonon DOS peak histogram of the Matbench Phonons dataset
widget = pmv.histogram(df_phonon[last_dos_peak], bins=50)
widget.show()
# widget.to_img("phonons-last-dos-peak-hist.pdf")


# %%
df_phonon["formula"] = df_phonon["structure"].map(lambda cryst: cryst.formula)
df_phonon["volume"] = df_phonon["structure"].map(lambda cryst: cryst.volume)

title = "Elemental prevalence in the Matbench phonons dataset"
pmv.ptable_heatmap(df_phonon["formula"], log=True, color_bar={"title": title}).show()


# %%
pmv.spacegroup_bar_plot(df_phonon["spg_num"]).show()
