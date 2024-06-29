"""Stats for the matbench_steels dataset.

Input: Chemical formula.
Target variable: Experimentally measured steel yield strengths in MPa.
Entries: 312

https://ml.materialsproject.org/projects/matbench_steels
"""

# %%
from matminer.datasets import load_dataset

from pymatviz import ptable_heatmap
from pymatviz.io import save_fig


# %%
df_steels = load_dataset("matbench_steels")


# %%
ax = df_steels.hist(column="yield strength", bins=50)
save_fig(ax, "steels-yield-strength-hist.pdf")


# %%
ax = ptable_heatmap(df_steels.composition, log=True)
ax.set(title="Elemental prevalence in the Matbench steels dataset")
save_fig(ax, "steels-ptable-heatmap.pdf")
