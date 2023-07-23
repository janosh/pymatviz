# %%
from __future__ import annotations

from matminer.datasets import load_dataset
from pymatgen.core import Composition

from pymatviz import ptable_heatmap
from pymatviz.plot_defaults import plt, px


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
df_gap = load_dataset("matbench_expt_gap")

df_gap["pmg_comp"] = df_gap.composition.map(Composition)
df_gap["n_atoms"] = [x.num_atoms for x in df_gap.pmg_comp]
df_gap["n_elems"] = df_gap.pmg_comp.map(len)


def mean_atomic_prop(comp: Composition, prop: str) -> float | None:
    """Get the mean value of an atomic property for a given pymatgen composition."""
    try:
        return sum(getattr(el, prop) * amt for el, amt in comp.items()) / comp.num_atoms
    except Exception:
        print(f"Could not compute mean {prop} for {comp}")
        return None


df_gap["mean_mass"] = [mean_atomic_prop(x, "atomic_mass") for x in df_gap.pmg_comp]
df_gap["mean_radius"] = [mean_atomic_prop(x, "atomic_radius") for x in df_gap.pmg_comp]


# %%
ptable_heatmap(
    df_gap.query("~composition.str.contains('Xe')").composition,
    log=True,
    text_color="black",
)
plt.title("Elements in Matbench experimental band gap dataset")
plt.savefig("expt-gap-ptable-heatmap.pdf")


# %%
fig = px.scatter(
    df_gap,
    x="n_atoms",
    y="gap expt",
    color="n_elems",
    size="mean_mass",
    hover_name="composition",
    log_x=True,
)
fig.update_layout(title="Marker size = mean atomic mass")
fig.write_image("expt-gap-scatter.pdf")
fig.show()
