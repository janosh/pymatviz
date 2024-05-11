# %%
import matplotlib.pyplot as plt
from matminer.datasets import load_dataset
from tqdm import tqdm

from pymatviz import ptable_heatmap, spacegroup_hist, spacegroup_sunburst
from pymatviz.enums import Key


"""Stats for the matbench_jdft2d dataset.

Input: Pymatgen Structure of the material.
Target variable: Exfoliation energy (meV).
Entries: 636

Matbench v0.1 dataset for predicting exfoliation energies from crystal structure
(computed with the OptB88vdW and TBmBJ functionals). Adapted from the JARVIS DFT DB.


https://ml.materialsproject.org/projects/matbench_jdft2d
"""

# %%
df_2d = load_dataset("matbench_jdft2d")

df_2d[[Key.spacegroup_symbol, Key.spacegroup]] = [
    struct.get_space_group_info() for struct in tqdm(df_2d.structure)
]

df_2d.describe()


# %%
df_2d.hist(column="exfoliation_en", bins=50, log=True)
plt.savefig("jdft2d-exfoliation-energy-hist.pdf")


# %%
df_2d[Key.volume] = [x.volume for x in df_2d.structure]
df_2d[Key.formula] = [x.formula for x in df_2d.structure]

ptable_heatmap(df_2d.formula, log=True)
plt.title("Elemental prevalence in the Matbench Jarvis DFT 2D dataset")
plt.savefig("jdft2d-ptable-heatmap.pdf")


# %%
spacegroup_hist(df_2d[Key.spacegroup], log=True)
plt.savefig("jdft2d-spacegroup-hist.pdf")


# %%
fig = spacegroup_sunburst(df_2d[Key.spacegroup], show_counts="percent")
fig.update_layout(title="Spacegroup sunburst of the JARVIS DFT 2D dataset")
fig.write_image("jdft2d-spacegroup-sunburst.pdf")
fig.show()
