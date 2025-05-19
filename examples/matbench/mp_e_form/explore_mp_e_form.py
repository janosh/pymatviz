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
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


# %%
df_e_form = load_dataset("matbench_mp_e_form")


# %%
fig = df_e_form["e_form"].hist(log_y=True, backend="plotly")
title = "<b>Formation energy histogram of the Matbench formation energy dataset</b>"
fig.layout.title.update(text=title, x=0.5)
fig.layout.showlegend = False
fig.show()
# pmv.save_fig(fig, "mp-e-form-hist.pdf")


# %%
df_e_form[Key.formula] = df_e_form[Key.structure].map(lambda struct: struct.formula)

fig = pmv.ptable_heatmap_plotly(df_e_form[Key.formula], log=True)
title = "<b>Elemental prevalence in the Matbench formation energy dataset</b>"
fig.layout.title.update(text=title, x=0.5)
fig.show()
# pmv.save_fig(fig, "mp_e_form-ptable-heatmap.pdf")
