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

from pymatviz import count_elements, ptable_heatmap, spacegroup_hist
from pymatviz.enums import Key
from pymatviz.io import save_fig


# %%
df_phonon = load_dataset(data_name := "matbench_phonons")
last_dos_peak = "last phdos peak"

df_phonon[[Key.spg_symbol, Key.spg_num]] = [
    struct.get_space_group_info() for struct in tqdm(df_phonon[Key.structure])
]


# %%
ax = df_phonon.hist(column=last_dos_peak, bins=50)
save_fig(ax, "phonons-last-dos-peak-hist.pdf")


# %%
df_phonon[Key.formula] = df_phonon[Key.structure].map(lambda cryst: cryst.formula)
df_phonon[Key.volume] = df_phonon[Key.structure].map(lambda cryst: cryst.volume)

fig = ptable_heatmap(
    count_elements(df_phonon[Key.formula]), log=True, return_type="figure"
)
fig.suptitle("Elemental prevalence in the Matbench phonons dataset")
save_fig(fig, "phonons-ptable-heatmap.pdf")


# %%
ax = spacegroup_hist(df_phonon[Key.spg_num])
save_fig(ax, "phonons-spacegroup-hist.pdf")
