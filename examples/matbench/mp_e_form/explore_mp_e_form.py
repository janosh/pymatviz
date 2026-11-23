"""Stats for the matbench_mp_e_form dataset.

Input: Pymatgen Structure of the material.
Target variable: Formation energy in eV as calculated by the Materials Project.
Entries: 132,752

Adapted from Materials Project database. Removed entries having
formation energy more than 3.0eV and those containing noble gases.
Retrieved April 2, 2019.

https://ml.materialsproject.org/projects/matbench_mp_e_form
"""

# /// script
# dependencies = [
#     "matminer>=0.10.1",
# ]
# ///

from matminer.datasets import load_dataset

import pymatviz as pmv


# %%
df_e_form = load_dataset("matbench_mp_e_form")


# %%
# Formation energy histogram of the Matbench formation energy dataset
widget = pmv.histogram(df_e_form["e_form"], log_y=True)
widget.show()
# widget.to_img("mp-e-form-hist.pdf")


# %%
df_e_form["formula"] = df_e_form["structure"].map(lambda struct: struct.formula)

title = "Elemental prevalence in the Matbench formation energy dataset"
pmv.ptable_heatmap(df_e_form["formula"], log=True, color_bar={"title": title}).show()
