# %%
from matminer.datasets import load_dataset
from tqdm import tqdm

from pymatviz import ptable_heatmap, spacegroup_hist, spacegroup_sunburst
from pymatviz.enums import Key
from pymatviz.io import save_fig


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
    struct.get_space_group_info() for struct in tqdm(df_2d[Key.structure])
]

df_2d.describe()


# %%
ax = df_2d.hist(column="exfoliation_en", bins=50, log=True)
save_fig(ax, "jdft2d-exfoliation-energy-hist.pdf")


# %%
df_2d[Key.volume] = [x.volume for x in df_2d[Key.structure]]
df_2d[Key.formula] = [x.formula for x in df_2d[Key.structure]]

ax = ptable_heatmap(df_2d[Key.formula], log=True)
ax.set(title="Elemental prevalence in the Matbench Jarvis DFT 2D dataset")
save_fig(ax, "jdft2d-ptable-heatmap.pdf")


# %%
ax = spacegroup_hist(df_2d[Key.spacegroup], log=True)
save_fig(ax, "jdft2d-spacegroup-hist.pdf")


# %%
fig = spacegroup_sunburst(df_2d[Key.spacegroup], show_counts="percent")
fig.layout.title = "Spacegroup sunburst of the JARVIS DFT 2D dataset"
fig.write_image("jdft2d-spacegroup-sunburst.pdf")
fig.show()
