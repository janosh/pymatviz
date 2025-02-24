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

# %%
from __future__ import annotations

import plotly.express as px
from matminer.datasets import load_dataset
from pymatgen.core import Composition

import pymatviz as pmv
from pymatviz.enums import Key


pmv.set_plotly_template("pymatviz_white")


# %%
df_gap = load_dataset("matbench_expt_gap")
gap_expt_key = "gap expt"

df_gap["pmg_comp"] = df_gap[Key.composition].map(Composition)
df_gap[Key.n_atoms] = [x.num_atoms for x in df_gap.pmg_comp]
df_gap[Key.n_elements] = df_gap.pmg_comp.map(len)


def mean_atomic_prop(comp: Composition, prop: str) -> float | None:
    """Get the mean value of an atomic property for a given pymatgen composition."""
    try:
        return sum(getattr(el, prop) * amt for el, amt in comp.items()) / comp.num_atoms
    except (ZeroDivisionError, TypeError):
        print(f"Could not compute mean {prop} for {comp}")
        return None


df_gap["mean_mass"] = [mean_atomic_prop(x, "atomic_mass") for x in df_gap.pmg_comp]
df_gap["mean_radius"] = [mean_atomic_prop(x, "atomic_radius") for x in df_gap.pmg_comp]


# %%
elem_counts = pmv.count_elements(
    df_gap.query("~composition.str.contains('Xe')")[Key.composition]
)
fig = pmv.ptable_heatmap_plotly(elem_counts, log=True)
fig.layout.title.update(text="Elements in Matbench experimental band gap dataset")
fig.show()
# pmv.save_fig(fig, "expt-gap-ptable-heatmap.pdf")


# %%
fig = px.scatter(
    df_gap,
    x=Key.n_atoms,
    y=gap_expt_key,
    color=Key.n_elements,
    size="mean_mass",
    hover_name=Key.composition,
    log_x=True,
)
fig.layout.title.update(text="Marker size = mean atomic mass", x=0.45, y=0.97)
fig.show()
