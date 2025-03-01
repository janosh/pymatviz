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

import numpy as np
import pandas as pd
import plotly.express as px
from matbench_discovery.structure.prototype import (
    count_wyckoff_positions,
    get_protostructure_label,
)
from matminer.datasets import load_dataset
from pymatgen.core import Structure
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.enums import Key


# %%
df_grvh = load_dataset("matbench_log_gvrh")
df_kvrh = load_dataset("matbench_log_kvrh")

df_sym = pd.DataFrame(
    struct.get_symmetry_dataset(backend="moyopy", return_raw_dataset=True).as_dict()
    for struct in df_grvh[Key.structure]
)
df_sym[Key.crystal_system] = df_sym["number"].map(pmv.utils.crystal_sys_from_spg_num)
df_grvh[Key.protostructure] = [
    get_protostructure_label(struct)
    for struct in tqdm(df_grvh[Key.structure], desc="matbench_log_gvrh Wyckoff strings")
]
df_kvrh[Key.protostructure] = df_grvh[Key.protostructure]

for df in (df_grvh, df_kvrh):
    df[[Key.spg_num, Key.wyckoff_symbols]] = df_sym[["number", "wyckoffs"]]
    df[Key.crystal_system] = df_sym[Key.crystal_system]

    df[Key.n_wyckoff_pos] = df[Key.protostructure].map(count_wyckoff_positions)
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
# pmv.save_fig(ax, "log_g+kvrh-target-hist.pdf")


# %%
df_grvh[Key.volume] = [x.volume for x in df_grvh[Key.structure]]
df_grvh[Key.formula] = [x.formula for x in df_grvh[Key.structure]]

fig = df_grvh[Key.volume].hist(nbins=100, log_y=True, opacity=0.8, backend="plotly")
title = "<b>Volume histogram of the Matbench bulk/shear modulus datasets</b>"
fig.layout.title.update(text=title, x=0.5)
fig.layout.showlegend = False
fig.show()
# pmv.save_fig(fig, "gvrh-volume-hist.pdf")


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
    return (dists.min(1) > radius).any()


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

fig = df_grvh[Key.volume].hist(nbins=100, backend="plotly", log_y=True)
title = "<b>Volume histogram of the Matbench bulk/shear modulus datasets</b>"
fig.layout.title.update(text=title, x=0.5)
fig.layout.showlegend = False
fig.show()


# %%
df_grvh[Key.formula] = df_grvh[Key.structure].map(lambda struct: struct.formula)

fig = pmv.ptable_heatmap_plotly(pmv.count_elements(df_grvh[Key.formula]), log=True)
title = "<b>Element counts in the Matbench<br>bulk/shear modulus datasets</b>"
fig.layout.title.update(text=title)
fig.show()
# pmv.save_fig(fig, "gvrh-ptable-heatmap.pdf")


# %%
fig = pmv.spacegroup_bar(df_grvh[Key.spg_num])
fig.layout.margin.update(b=10, l=10, r=10, t=50)
fig.layout.title.update(
    text="Spacegroup histogram of the Matbench bulk/shear modulus datasets"
)
fig.show()
# pmv.save_fig(fig, "gvrh-spacegroup-hist.pdf")


# %%
fig = pmv.spacegroup_sunburst(df_grvh[Key.spg_num], show_counts="percent")
title = "Spacegroup sunburst of the Matbench bulk/shear modulus datasets"
fig.layout.title.update(text=title, x=0.5)
# pmv.save_fig(fig, "gvrh-spacegroup-sunburst.pdf")
fig.show()


# %%
fig = px.violin(
    df_grvh,
    color=Key.crystal_system,
    x=Key.crystal_system,
    y=Key.n_wyckoff_pos,
    points="all",
    hover_data=[Key.spg_num],
    hover_name=Key.formula,
    category_orders={Key.crystal_system: pmv.crystal_sys_order},
    log_y=True,
).update_traces(jitter=1)


def rgb_color(val: float, max_val: float) -> str:
    """Convert a value between 0 and max to a color between red and blue."""
    return f"rgb({255 * val / max_val:.1f}, 0, {255 * (max_val - val) / max_val:.1f})"


x_ticks = {}
for cry_sys, df_group in sorted(
    df_grvh.groupby(Key.crystal_system), key=lambda x: pmv.crystal_sys_order.index(x[0])
):
    n_wyckoff_top = df_group[Key.n_wyckoff_pos].mean()
    clr = rgb_color(n_wyckoff_top, 14)
    x_ticks[cry_sys] = (
        f"<b>{cry_sys}</b><br>"
        f"{len(df_group):,} = {len(df_group) / len(df_grvh):.0%}<br>"
        f"mean = <span style='color:{clr}'><b>{n_wyckoff_top:.1f}</b></span>"
    )

title = (
    "Matbench bulk/shear modulus datasets<br>Number of Wyckoff positions "
    "per structure by crystal system"
)
fig.layout.title.update(text=title, x=0.5)
fig.layout.margin.update(b=10, l=10, r=10, t=50)
fig.layout.showlegend = False
fig.layout.update(width=1000, height=400)
fig.layout.xaxis = dict(tickvals=list(range(7)), ticktext=list(x_ticks.values()))
fig.show()
# pmv.save_fig(fig, "grvh-violin-num-wyckoffs.pdf")
