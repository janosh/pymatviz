# %%
from matminer.datasets import load_dataset

from pymatviz import ptable_heatmap
from pymatviz.enums import Key
from pymatviz.io import save_fig


"""Stats for the matbench_mp_e_form dataset.

Input: Pymatgen Structure of the material.
Target variable: Formation energy in eV as calculated by the Materials Project.
Entries: 132,752

Adapted from Materials Project database. Removed entries having
formation energy more than 3.0eV and those containing noble gases.
Retrieved April 2, 2019.

https://ml.materialsproject.org/projects/matbench_mp_e_form
"""

# %%
df_e_form = load_dataset("matbench_mp_e_form")


# %%
ax = df_e_form.hist(column="e_form", bins=50, log=True)
save_fig(ax, "mp_e_form_hist.pdf")


# %%
df_e_form[Key.formula] = df_e_form[Key.structure].map(lambda struct: struct.formula)


# %%
ax = ptable_heatmap(df_e_form[Key.formula], log=True)
ax.set(title="Elemental prevalence in the Matbench formation energy dataset")
save_fig(ax, "mp_e_form-ptable-heatmap.pdf")
