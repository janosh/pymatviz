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

from pymatviz import ptable_heatmap, spacegroup_hist
from pymatviz.io import save_fig


# %%
df_phonon = load_dataset(data_name := "matbench_phonons")
formula_col, volume_col, spg_col = "formula", "volume", "spg_num"

df_phonon[["spg_symbol", spg_col]] = [
    struct.get_space_group_info() for struct in tqdm(df_phonon.structure)
]


# %%
ax = df_phonon.hist(column="last phdos peak", bins=50)
save_fig(ax, "phonons-last-dos-peak-hist.pdf")


# %%
df_phonon[formula_col] = df_phonon.structure.map(lambda cryst: cryst.formula)
df_phonon[volume_col] = df_phonon.structure.map(lambda cryst: cryst.volume)

ax = ptable_heatmap(df_phonon[formula_col], log=True)
ax.set(title="Elemental prevalence in the Matbench phonons dataset")
save_fig(ax, "phonons-ptable-heatmap.pdf")


# %%
ax = spacegroup_hist(df_phonon[spg_col])
save_fig(ax, "phonons-spacegroup-hist.pdf")
