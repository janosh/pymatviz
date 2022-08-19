"""Stats for the matbench_steels dataset.

Input: Chemical formula.
Target variable: Experimentally measured steel yield strengths in MPa.
Entries: 312

https://ml.materialsproject.org/projects/matbench_steels
"""


# %%
from matminer.datasets import load_dataset

from dataset_exploration.plot_defaults import plt
from pymatviz import ptable_heatmap


# %%
df_steels = load_dataset("matbench_steels")


# %%
df_steels.hist(column="yield strength", bins=50)
plt.savefig("steels-yield-strength-hist.pdf")


# %%
ptable_heatmap(df_steels.composition, log=True)
plt.title("Elemental prevalence in the Matbench steels dataset")
plt.savefig("steels-ptable-heatmap.pdf")
