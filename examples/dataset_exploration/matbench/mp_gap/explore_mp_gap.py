# %%
from matminer.datasets import load_dataset

from pymatviz import count_elements, ptable_heatmap
from pymatviz.enums import Key
from pymatviz.io import save_fig


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
ax = df_gap.hist(column="gap pbe", bins=50, log=True)
ax.set(xlabel="eV")
save_fig(ax, "pbe_gap_hist.pdf")


# %%
df_gap[Key.n_sites] = df_gap[Key.structure].map(len)
df_gap[Key.volume] = df_gap[Key.structure].map(lambda cryst: cryst.volume)
df_gap[Key.vol_per_atom] = df_gap[Key.volume] / df_gap[Key.n_sites]
df_gap[Key.formula] = df_gap[Key.structure].map(lambda cryst: cryst.formula)

fig = ptable_heatmap(
    count_elements(df_gap[Key.formula]),
    log=True,
    value_kwargs={"fontsize": 10},
    return_type="figure",
)
fig.suptitle("Elemental prevalence in the Matbench MP band gap dataset")
save_fig(fig, "mp_gap-ptable-heatmap.pdf")


# %%
df_gap.hist(column=Key.vol_per_atom, bins=50, log=True)
save_fig(ax, "volume_per_atom_hist.pdf")
