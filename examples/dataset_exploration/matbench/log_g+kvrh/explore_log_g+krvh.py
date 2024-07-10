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
import plotly.express as px
from aviary.wren.utils import count_wyckoff_positions, get_aflow_label_from_spglib
from matminer.datasets import load_dataset
from pymatgen.core import Structure
from tqdm import tqdm

from pymatviz import (
    count_elements,
    crystal_sys_order,
    ptable_heatmap,
    spacegroup_hist,
    spacegroup_sunburst,
)
from pymatviz.enums import Key
from pymatviz.io import save_fig
from pymatviz.utils import crystal_sys_from_spg_num


# %%
df_grvh = load_dataset("matbench_log_gvrh")
df_kvrh = load_dataset("matbench_log_kvrh")

# getting space group symbols and numbers for 10,987 structures takes about 45 sec
df_grvh[[Key.spg_symbol, Key.spg_num]] = [
    struct.get_space_group_info()
    for struct in tqdm(
        df_grvh[Key.structure], desc="Getting matbench_log_gvrh spacegroups"
    )
]
df_grvh[Key.crystal_system] = [
    crystal_sys_from_spg_num(x) for x in df_grvh[Key.spg_num]
]

df_grvh["wyckoff"] = [
    get_aflow_label_from_spglib(struct)
    for struct in tqdm(
        df_grvh[Key.structure], desc="Getting matbench_log_gvrh Wyckoff strings"
    )
]
df_grvh["n_wyckoff"] = df_grvh.wyckoff.map(count_wyckoff_positions)
df_grvh[Key.formula] = [x.formula for x in df_grvh[Key.structure]]


# %%
print("Number of materials with shear modulus of 0:")
print(sum(df_grvh["log10(G_VRH)"] == 0))  # sum is 31


# %%
print("Number of materials with bulk modulus of 0:")
print(sum(df_kvrh["log10(K_VRH)"] == 0))  # sum is 14


# %%
ax = df_kvrh.hist(column="log10(K_VRH)", bins=50, alpha=0.8)

df_grvh.hist(column="log10(G_VRH)", bins=50, ax=ax, alpha=0.8)
save_fig(ax, "log_g+kvrh-target-hist.pdf")


# %%
df_grvh[Key.volume] = [x.volume for x in df_grvh[Key.structure]]
df_grvh[Key.formula] = [x.formula for x in df_grvh[Key.structure]]

ax = df_grvh.hist(column=Key.volume, bins=50, log=True, alpha=0.8)
save_fig(ax, "log_gvrh-volume-hist.pdf")


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

df_grvh.hist(column=Key.volume, bins=50, log=True)


# %%
df_grvh[Key.formula] = df_grvh[Key.structure].map(lambda struct: struct.formula)

fig = ptable_heatmap(count_elements(df_grvh[Key.formula]), log=True)
fig.suptitle("Elemental prevalence in the Matbench bulk/shear modulus datasets")
save_fig(fig, "log_gvrh-ptable-heatmap.pdf")


# %%
ax = spacegroup_hist(df_grvh[Key.spg_num])
save_fig(ax, "log_gvrh-spacegroup-hist.pdf")


# %%
fig = spacegroup_sunburst(df_grvh[Key.spg_num], show_counts="percent")
fig.layout.title = "Spacegroup sunburst of the JARVIS DFT 2D dataset"
fig.write_image("log_gvrh-spacegroup-sunburst.pdf")
fig.show()


# %%
fig = px.violin(
    df_grvh,
    color=Key.crystal_system,
    x=Key.crystal_system,
    y="n_wyckoff",
    points="all",
    hover_data=[Key.spg_num],
    hover_name=Key.formula,
    category_orders={Key.crystal_system: crystal_sys_order},
    log_y=True,
).update_traces(jitter=1)


def rgb_color(val: float, max_val: float) -> str:
    """Convert a value between 0 and max to a color between red and blue."""
    return f"rgb({255 * val / max_val:.1f}, 0, {255 * (max_val - val) / max_val:.1f})"


x_ticks = {}
for cry_sys, df_group in sorted(
    df_grvh.groupby(Key.crystal_system), key=lambda x: crystal_sys_order.index(x[0])
):
    n_wyckoff_top = df_group.n_wyckoff.mean()
    clr = rgb_color(n_wyckoff_top, 14)
    x_ticks[cry_sys] = (
        f"<b>{cry_sys}</b><br>"
        f"{len(df_group):,} = {len(df_group) / len(df_grvh):.0%}<br>"
        f"mean = <span style='color:{clr}'><b>{n_wyckoff_top:.1f}</b></span>"
    )

title = "Matbench dielectric: Number of Wyckoff positions by crystal system"
fig.layout.title = dict(text=title, x=0.5)
fig.layout.margin = dict(b=10, l=10, r=10, t=50)
fig.layout.showlegend = False
fig.layout.update(width=1000, height=400)
fig.layout.xaxis = dict(tickvals=list(range(7)), ticktext=list(x_ticks.values()))

# fig.write_image("log_grvh-violin-num-wyckoffs.pdf")
fig.show()
