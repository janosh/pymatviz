"""Stats for the matbench_steels dataset.

Input: Chemical formula.
Target variable: Experimentally measured steel yield strengths in MPa.
Entries: 312

https://ml.materialsproject.org/projects/matbench_steels
"""

# %%
from __future__ import annotations

from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz import count_elements, ptable_heatmap
from pymatviz.enums import Key


# %%
df_steels = load_dataset("matbench_steels")


# %%
ax = df_steels.hist(column="yield strength", bins=50)
pmv.save_fig(ax, "steels-yield-strength-hist.pdf")


# %%
fig = ptable_heatmap(
    count_elements(df_steels[Key.composition]), log=True, return_type="figure"
)
fig.suptitle("Elemental prevalence in the Matbench steels dataset")
pmv.save_fig(fig, "steels-ptable-heatmap.pdf")
