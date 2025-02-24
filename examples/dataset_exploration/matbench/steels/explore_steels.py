"""Stats for the matbench_steels dataset.

Input: Chemical formula.
Target variable: Experimentally measured steel yield strengths in MPa.
Entries: 312

https://ml.materialsproject.org/projects/matbench_steels
"""

# %%
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


# %%
df_steels = load_dataset("matbench_steels")


# %%
fig = df_steels["yield strength"].hist(nbins=100, backend="plotly")
fig.layout.title.update(text="Yield strength histogram of Matbench Steels dataset")
fig.layout.showlegend = False
fig.show()
# pmv.save_fig(fig, "steels-yield-strength-hist.pdf")


# %%
fig = pmv.ptable_heatmap_plotly(df_steels[Key.composition], log=True)
fig.layout.title.update(text="Elemental prevalence in the Matbench steels dataset")
fig.show()
# pmv.save_fig(fig, "steels-ptable-heatmap.pdf")
