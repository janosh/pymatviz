# %%
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz import count_elements, ptable_heatmap
from pymatviz.enums import Key


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
pmv.save_fig(ax, "mp_e_form_hist.pdf")


# %%
df_e_form[Key.formula] = df_e_form[Key.structure].map(lambda struct: struct.formula)

fig = ptable_heatmap(
    count_elements(df_e_form[Key.formula]),
    log=True,
    value_kwargs={"fontsize": 10},
    return_type="figure",
)
fig.suptitle("Elemental prevalence in the Matbench formation energy dataset")
pmv.save_fig(fig, "mp_e_form-ptable-heatmap.pdf")
