"""Stats for the matbench_jdft2d dataset.

Input: Pymatgen Structure of the material.
Target variable: Exfoliation energy (meV).
Entries: 636

Matbench v0.1 dataset for predicting exfoliation energies from crystal structure
(computed with the OptB88vdW and TBmBJ functionals). Adapted from the JARVIS DFT DB.


https://ml.materialsproject.org/projects/matbench_jdft2d
"""

# /// script
# dependencies = [
#     "matminer>=0.10.1",
#     "tqdm>=4.67",
# ]
# ///

# %%
from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv


# %%
df_2d = load_dataset("matbench_jdft2d")

df_2d[["spg_symbol", "spg_num"]] = [
    struct.get_space_group_info() for struct in tqdm(df_2d["structure"])
]

df_2d.describe()


# %%
# Exfoliation energy histogram of the Matbench Jarvis DFT 2D dataset
widget = pmv.histogram(df_2d["exfoliation_en"], bins=200, log_y=True)
widget.show()
# widget.to_img("jdft2d-exfoliation-energy-hist.pdf")


# %%
df_2d["volume"] = [x.volume for x in df_2d["structure"]]
df_2d["formula"] = [x.formula for x in df_2d["structure"]]

title = "Elemental prevalence in the Matbench Jarvis DFT 2D dataset"
pmv.ptable_heatmap(df_2d["formula"], log=True, color_bar={"title": title}).show()


# %%
pmv.spacegroup_bar_plot(df_2d["spg_num"], log_y=True).show()


# %%
# Spacegroup sunburst of the JARVIS DFT 2D dataset
widget = pmv.spacegroup_sunburst(df_2d["spg_num"], label_text="label+percent")
widget.show()
# widget.to_img("jdft2d-spacegroup-sunburst.pdf")
