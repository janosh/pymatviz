"""Stats for the matbench_expt_gap dataset.

Input: Chemical formula.
Target variable: Experimentally measured band gap (E_g) in eV.
Entries: 4604

Matbench v0.1 dataset for predicting experimental band gaps from composition alone.
Retrieved from Zhuo et al. supplementary information. Deduplicated according to
composition, removing compositions with reported band gaps spanning more than a 0.1 eV
range; remaining compositions were assigned values based on the closest experimental
value to the mean experimental value for that composition among all reports.

To get likely MP IDs for each chemical formula, see https://git.io/JmpVe:

Likely mp-ids were chosen from among computed materials in the MP database
(version 2021.03) that were 1) not marked 'theoretical', 2) had structures matching at
least one ICSD material, and 3) were within 200 meV of the DFT-computed stable energy
hull (e_above_hull < 0.2 eV). Among these candidates, we chose the mp-id with the lowest
e_above_hull.

https://ml.materialsproject.org/projects/matbench_expt_gap
"""

# /// script
# dependencies = [
#     "matminer>=0.10.1",
# ]
# ///

# %%
import kemio
from matminer.datasets import load_dataset

import pymatviz as pmv


# %%
df_gap = load_dataset("matbench_expt_gap")
gap_expt_key = "gap expt"

compositions = df_gap["composition"].map(kemio.Composition)
df_gap["n_atoms"] = [comp.num_atoms for comp in compositions]
df_gap["n_elements"] = compositions.map(len)
df_gap["mean_mass"] = [comp.weight / comp.num_atoms for comp in compositions]


# %%
elem_counts = pmv.count_elements(
    df_gap.query("~composition.str.contains('Xe')")["composition"]
)
title = "Elements in Matbench experimental band gap dataset"
pmv.ptable_heatmap(elem_counts, log=True, color_bar={"title": title}).show()


# %% marker size = mean atomic mass, color = number of elements
widget = pmv.scatter_plot(
    series=[
        {
            "x": df_gap["n_atoms"],
            "y": df_gap[gap_expt_key],
            "color_values": df_gap["n_elements"],
            "size_values": df_gap["mean_mass"],
            "markers": "points",
        }
    ],
    x_axis={"label": "Number of Atoms", "scale_type": "log"},
    y_axis={"label": "Experimental band gap (eV)"},
    color_bar={"title": "Number of Elements"},
)
widget.show()
