# %%
from matminer.datasets import load_dataset

from dataset_exploration.plot_defaults import plt
from pymatviz import ptable_heatmap


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
df_e_form.hist(column="e_form", bins=50, log=True)
plt.savefig("mp_e_form_hist.pdf")


# %%
df_e_form["formula"] = df_e_form.structure.apply(lambda struct: struct.formula)


# %%
ptable_heatmap(df_e_form.formula, log=True)
plt.title("Elemental prevalence in the Matbench formation energy dataset")
plt.savefig("mp_e_form-ptable-heatmap.pdf")
