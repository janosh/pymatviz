"""Stats for the matbench_steels dataset.

Input: Chemical formula.
Target variable: Experimentally measured steel yield strengths in MPa.
Entries: 312

https://ml.materialsproject.org/projects/matbench_steels
"""


# %%
import matplotlib.pyplot as plt
from matminer.datasets import load_dataset

from pymatviz import ptable_heatmap


plt.rc("font", size=16)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("figure", dpi=150, titlesize=18)
plt.rcParams["figure.constrained_layout.use"] = True


# %%
df_steels = load_dataset("matbench_steels")


# %%
df_steels.hist(column="yield strength", bins=50)
plt.savefig("steels-yield-strength-hist.pdf")


# %%
ptable_heatmap(df_steels.composition, log=True)
plt.title("Elemental prevalence in the Matbench steels dataset")
plt.savefig("steels-ptable-heatmap.pdf")
