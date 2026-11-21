"""Stats for the matbench_log_gvrh and matbench_log_kvrh datasets.

Input: Pymatgen Structure of the material.
Target variable(s): Base 10 logarithm of the DFT Voigt-Reuss-Hill average shear (g_vrh)
    and bulk (k_vrh) moduli in GPa.
Entries: 10987 (each)

https://ml.materialsproject.org/projects/matbench_log_gvrh
https://ml.materialsproject.org/projects/matbench_log_kvrh
"""

# %%
from time import perf_counter
from typing import get_args

import numpy as np
import pandas as pd
from matbench_discovery.structure import prototype
from matminer.datasets import load_dataset
from pymatgen.core import Structure
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.typing import CrystalSystem


crystal_systems = get_args(CrystalSystem)  # ordered triclinic to cubic


# %%
df_grvh = load_dataset("matbench_log_gvrh")
df_kvrh = load_dataset("matbench_log_kvrh")

df_sym = pd.DataFrame(
    struct.get_symmetry_dataset(backend="moyopy", return_raw_dataset=True).as_dict()
    for struct in df_grvh[Key.structure]
)
df_sym[Key.crystal_system] = df_sym["number"].map(pmv.utils.spg_to_crystal_sys)
df_grvh[Key.protostructure] = [
    prototype.get_protostructure_label(struct)
    for struct in tqdm(df_grvh[Key.structure], desc="matbench_log_gvrh Wyckoff strings")
]
df_kvrh[Key.protostructure] = df_grvh[Key.protostructure]

for df in (df_grvh, df_kvrh):
    df[[Key.spg_num, Key.wyckoff_symbols]] = df_sym[["number", "wyckoffs"]]
    df[Key.crystal_system] = df_sym[Key.crystal_system]

    df[Key.n_wyckoff_pos] = df[Key.protostructure].map(
        prototype.count_wyckoff_positions
    )
    df[Key.formula] = [x.formula for x in df[Key.structure]]


# %%
print("Number of materials with shear modulus of 0:")
print(sum(df_grvh["log10(G_VRH)"] == 0))  # expect 31


# %%
print("Number of materials with bulk modulus of 0:")
print(sum(df_kvrh["log10(K_VRH)"] == 0))  # expect 14


# %%
ax = df_kvrh.hist(column="log10(K_VRH)", bins=50, alpha=0.8)

df_grvh.hist(column="log10(G_VRH)", bins=50, ax=ax, alpha=0.8)
# ax.flat[0].figure.savefig("log_g+kvrh-target-hist.pdf")


# %%
df_grvh[Key.volume] = [x.volume for x in df_grvh[Key.structure]]
df_grvh[Key.formula] = [x.formula for x in df_grvh[Key.structure]]

# Volume histogram of the Matbench bulk/shear modulus datasets
widget = pmv.histogram(
    df_grvh[Key.volume], bins=100, log_y=True, x_axis={"label": Key.volume.label}
)
widget.show()
# widget.to_img("gvrh-volume-hist.pdf")


# %%
start = perf_counter()
radius = 5
df_grvh[f"neighbor_list_r{radius}"] = [
    x.get_neighbor_list(r=radius) for x in df_grvh[Key.structure]
]
print(f"took {perf_counter() - start:.3f} sec")

df_kvrh[f"neighbor_list_r{radius}"] = [
    x.get_neighbor_list(r=radius) for x in df_kvrh[Key.structure]
]


# %%
start = perf_counter()


def has_isolated_atom(crystal: Structure, radius: float = 5) -> bool:
    """Check if crystal has isolated atoms within specified neighborhood radius."""
    dists = crystal.distance_matrix
    np.fill_diagonal(dists, np.inf)
    return bool((dists.min(1) > radius).any())


df_grvh["isolated_r5"] = df_grvh[Key.structure].map(has_isolated_atom)
print(f"took {perf_counter() - start:.3f} sec")


# %%
df_grvh["graph_size"] = df_grvh[f"neighbor_list_r{radius}"].map(lambda lst: len(lst[0]))


# %%
for idx, structure, target, *_ in df_grvh.query("graph_size == 0").itertuples():
    print(f"\n{idx = }")
    print(f"{structure = }")
    print(f"{target = }")


# %%
df_grvh[Key.volume] = df_grvh[Key.structure].map(lambda struct: struct.volume)

pmv.histogram(
    df_grvh[Key.volume], bins=100, log_y=True, x_axis={"label": Key.volume.label}
).show()


# %%
df_grvh[Key.formula] = df_grvh[Key.structure].map(lambda struct: struct.formula)

title = "Element counts in the Matbench bulk/shear modulus datasets"
pmv.ptable_heatmap(df_grvh[Key.formula], log=True, color_bar={"title": title}).show()


# %%
pmv.spacegroup_bar(df_grvh[Key.spg_num]).show()


# %%
# Spacegroup sunburst of the Matbench bulk/shear modulus datasets
widget = pmv.spacegroup_sunburst(df_grvh[Key.spg_num], label_text="label+percent")
# widget.to_img("gvrh-spacegroup-sunburst.pdf")
widget.show()


# %% Number of Wyckoff positions per structure by crystal system
# MatterViz has no rich-text tick labels, so per-group stats go into series labels
series = []
for cry_sys, df_group in sorted(
    df_grvh.groupby(Key.crystal_system), key=lambda x: crystal_systems.index(x[0])
):
    n_wyckoff_mean = df_group[Key.n_wyckoff_pos].mean()
    label = (
        f"{cry_sys} {len(df_group):,} = {len(df_group) / len(df_grvh):.0%} "
        f"mean={n_wyckoff_mean:.1f}"
    )
    series += [{"y": df_group[Key.n_wyckoff_pos], "label": label}]

widget = pmv.BoxPlotWidget(
    series,
    kind="violin",
    points="strip",
    title="Matbench bulk/shear modulus datasets",
    show_legend=False,
    y_axis={"label": Key.n_wyckoff_pos.label, "scale_type": "log"},
)
widget.show()
# widget.to_img("grvh-violin-num-wyckoffs.pdf")
