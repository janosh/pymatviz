"""Stats for the matbench_log_gvrh and matbench_log_kvrh datasets.

Input: Pymatgen Structure of the material.
Target variable(s): Base 10 logarithm of the DFT Voigt-Reuss-Hill average shear (g_vrh)
    and bulk (k_vrh) moduli in GPa.
Entries: 10987 (each)

https://ml.materialsproject.org/projects/matbench_log_gvrh
https://ml.materialsproject.org/projects/matbench_log_kvrh
"""

# /// script
# dependencies = [
#     "matminer>=0.10.1",
# ]
# ///

import re
from time import perf_counter
from typing import get_args

import kemio
import numpy as np
from kemio.interop import to_structure
from kemio.options import CrystalSystem
from matminer.datasets import load_dataset

import pymatviz as pmv


crystal_systems = get_args(CrystalSystem)  # ordered triclinic to cubic


def count_wyckoff_positions(protostructure_label: str) -> int:
    """Number of occupied Wyckoff positions in an AFLOW-style protostructure label
    like ``A4B3_hP28_159_ab2c_2c:N-Si`` (a digit before a letter repeats that site).
    """
    wyckoff_part = protostructure_label.split(":", 1)[0].split("_", 3)[3]
    site_repeats = re.findall(r"(\d*)[a-z]", wyckoff_part)
    return sum(int(repeat or 1) for repeat in site_repeats)


# %% both datasets share the same structures, only their targets differ
df_grvh = load_dataset("matbench_log_gvrh")
df_kvrh = load_dataset("matbench_log_kvrh")
kemio_structs = [to_structure(struct) for struct in df_grvh["structure"]]

sym_summary = kemio.symmetry_summary(kemio_structs, symprec=0.01)
protostructures = kemio.protostructure_labels(kemio_structs)

for df in (df_grvh, df_kvrh):
    df["spg_num"] = sym_summary.number
    df["wyckoff_symbols"] = sym_summary.wyckoffs
    df["crystal_system"] = df["spg_num"].map(kemio.crystal_system)
    df["protostructure"] = protostructures
    df["n_wyckoff_pos"] = df["protostructure"].map(
        count_wyckoff_positions, na_action="ignore"
    )
    df["formula"] = [struct.composition.formula for struct in kemio_structs]
    df["volume"] = [struct.volume for struct in kemio_structs]


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


# %% Volume histogram of the Matbench bulk/shear modulus datasets
widget = pmv.histogram(
    df_grvh["volume"], bins=100, log_y=True, x_axis={"label": "Volume (Å³)"}
)
widget.show()
# widget.to_img("gvrh-volume-hist.pdf")


# %%
start = perf_counter()
radius = 5
neighbor_lists = [struct.neighbor_list(radius) for struct in kemio_structs]
print(f"took {perf_counter() - start:.3f} sec")

for df in (df_grvh, df_kvrh):
    df[f"neighbor_list_r{radius}"] = neighbor_lists


# %%
start = perf_counter()


def has_isolated_atom(crystal: kemio.Structure, radius: float = 5) -> bool:
    """Check if crystal has isolated atoms within specified neighborhood radius."""
    dists = crystal.distance_matrix
    np.fill_diagonal(dists, np.inf)
    return bool((dists.min(1) > radius).any())


df_grvh["isolated_r5"] = [has_isolated_atom(struct) for struct in kemio_structs]
print(f"took {perf_counter() - start:.3f} sec")


# %%
df_grvh["graph_size"] = df_grvh[f"neighbor_list_r{radius}"].map(lambda lst: len(lst[0]))


# %%
for idx, structure, target, *_ in df_grvh.query("graph_size == 0").itertuples():
    print(f"\n{idx = }")
    print(f"{structure = }")
    print(f"{target = }")


# %%
title = "Element counts in the Matbench bulk/shear modulus datasets"
pmv.ptable_heatmap(df_grvh["formula"], log=True, color_bar={"title": title}).show()


# %%
pmv.spacegroup_bar_plot(df_grvh["spg_num"]).show()


# %%
# Spacegroup sunburst of the Matbench bulk/shear modulus datasets
widget = pmv.spacegroup_sunburst(df_grvh["spg_num"], label_text="label+percent")
# widget.to_img("gvrh-spacegroup-sunburst.pdf")
widget.show()


# %% Number of Wyckoff positions per structure by crystal system
# MatterViz has no rich-text tick labels, so per-group stats go into series labels
series = []
for cry_sys, df_group in sorted(
    df_grvh.groupby("crystal_system"), key=lambda x: crystal_systems.index(x[0])
):
    n_wyckoff_mean = df_group["n_wyckoff_pos"].mean()
    label = (
        f"{cry_sys} {len(df_group):,} = {len(df_group) / len(df_grvh):.0%} "
        f"mean={n_wyckoff_mean:.1f}"
    )
    series += [{"y": df_group["n_wyckoff_pos"], "label": label}]

widget = pmv.box_plot(
    series,
    kind="violin",
    points="strip",
    title="Matbench bulk/shear modulus datasets",
    show_legend=False,
    y_axis={"label": "Number of Wyckoff Positions", "scale_type": "log"},
)
widget.show()
# widget.to_img("grvh-violin-num-wyckoffs.pdf")
