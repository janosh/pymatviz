# %%
from matminer.datasets import load_dataset

from pymatviz import ptable_heatmap
from pymatviz.plot_defaults import plt


"""Stats for the matbench_mp_gap dataset.

Input: Pymatgen Structure of the material.
Target variable: The band gap (E_g) as calculated by PBE DFT from the Materials Project
    in eV.
Entries: 106113

https://ml.materialsproject.org/projects/matbench_mp_gap
"""


# %%
df_gap = load_dataset("matbench_mp_gap")


# %%
df_gap.hist(column="gap pbe", bins=50, log=True)
plt.xlabel("eV")
plt.savefig("pbe_gap_hist.pdf")


# %%
df_gap["volume/atom"] = df_gap.structure.map(
    lambda cryst: cryst.volume / cryst.num_sites
)
df_gap["num_sites"] = df_gap.structure.map(lambda cryst: cryst.num_sites)

df_gap["formula"] = df_gap.structure.map(lambda cryst: cryst.formula)


# %%
ptable_heatmap(df_gap.formula, log=True)
plt.title("Elemental prevalence in the Matbench MP band gap dataset")
plt.savefig("mp_gap-ptable-heatmap.pdf")


# %%
df_gap.hist(column="volume/atom", bins=50, log=True)
plt.savefig("volume_per_atom_hist.pdf")
