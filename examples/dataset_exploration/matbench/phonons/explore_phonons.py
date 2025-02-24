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
from pymatviz.enums import Key


# %%
df_phonon = load_dataset(data_name := "matbench_phonons")
last_dos_peak = "last phdos peak"

df_phonon[[Key.spg_symbol, Key.spg_num]] = [
    struct.get_space_group_info() for struct in tqdm(df_phonon[Key.structure])
]


# %%
fig = df_phonon[last_dos_peak].hist(bins=50, backend="plotly")
fig.layout.title.update(
    text="Last phonon DOS peak histogram of Matbench Phonons dataset"
)
fig.show()
# pmv.save_fig(ax, "phonons-last-dos-peak-hist.pdf")


# %%
df_phonon[Key.formula] = df_phonon[Key.structure].map(lambda cryst: cryst.formula)
df_phonon[Key.volume] = df_phonon[Key.structure].map(lambda cryst: cryst.volume)

fig = pmv.ptable_heatmap_plotly(df_phonon[Key.formula], log=True)
fig.layout.title.update(text="Elemental prevalence in the Matbench phonons dataset")
fig.show()
# pmv.save_fig(fig, "phonons-ptable-heatmap.pdf")


# %%
fig = pmv.spacegroup_bar(df_phonon[Key.spg_num])
fig.layout.title.update(text="Spacegroup histogram of Matbench Phonons dataset", y=0.98)
fig.layout.margin.update(b=10, l=10, r=10, t=60)
fig.show()
# pmv.save_fig(fig, "phonons-spacegroup-hist.pdf")
