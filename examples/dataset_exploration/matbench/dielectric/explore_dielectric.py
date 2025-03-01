"""matbench_dielectric dataset.

Input: Pymatgen Structure of the material.
Target variable: refractive index.
Entries: 636

Matbench v0.1 test dataset for predicting refractive index from structure. Adapted from
Materials Project database. Removed entries having a formation energy (or energy above
the convex hull) more than 150meV and those having refractive indices less than 1 and
those containing noble gases. Retrieved April 2, 2019.

https://ml.materialsproject.org/projects/matbench_dielectric
"""

# %%
import plotly.express as px
from matbench_discovery.structure.prototype import (
    count_wyckoff_positions,
    get_protostructure_label,
)
from matminer.datasets import load_dataset
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.enums import Key


# %%
df_diel = load_dataset("matbench_dielectric")

df_diel[[Key.spg_symbol, Key.spg_num]] = [
    struct.get_space_group_info()
    for struct in tqdm(df_diel[Key.structure], desc="Getting spacegroups")
]

proto_label_key = f"{Key.protostructure}_moyo"
n_wyckoff_pos_key = f"{Key.n_wyckoff_pos}_moyo"
df_diel[proto_label_key] = [
    get_protostructure_label(struct)
    for struct in tqdm(df_diel[Key.structure], desc="Getting Wyckoff strings")
]
df_diel[n_wyckoff_pos_key] = df_diel[proto_label_key].map(count_wyckoff_positions)

df_diel[Key.crystal_system] = df_diel[Key.spg_num].map(
    pmv.utils.crystal_sys_from_spg_num
)

df_diel[Key.volume] = [x.volume for x in df_diel[Key.structure]]
df_diel[Key.formula] = [x.formula for x in df_diel[Key.structure]]


# %%
fig = pmv.ptable_heatmap_plotly(df_diel[Key.formula], log=True, colorscale="viridis")
title = "<b>Elements in Matbench Dielectric</b>"
fig.layout.title = dict(text=title, x=0.4, y=0.94, font_size=20)
fig.show()
# pmv.save_fig(fig, "dielectric-ptable-heatmap-plotly.pdf")


# %%
fig = pmv.spacegroup_bar(df_diel[Key.spg_num])
fig.layout.title.update(text="<b>Space group histogram</b>")
fig.layout.margin.update(b=10, l=10, r=10, t=50)
# pmv.save_fig(fig, "dielectric-spacegroup-hist.pdf")
fig.show()


# %%
fig = pmv.spacegroup_sunburst(df_diel[Key.spg_num], show_counts="percent")
fig.layout.title.update(text="<b>Space group sunburst</b>", x=0.5)
# pmv.save_fig(fig, "dielectric-spacegroup-sunburst.pdf")
fig.show()


# %%
fig = px.violin(
    df_diel,
    color=Key.crystal_system,
    x=Key.crystal_system,
    y="n",
    points="all",
    hover_data=[Key.spg_num],
    hover_name=Key.formula,
).update_traces(jitter=1)

x_ticks = {}  # custom x axis tick labels
for cry_sys, df_group in sorted(
    df_diel.groupby(Key.crystal_system), key=lambda x: pmv.crystal_sys_order.index(x[0])
):
    x_ticks[cry_sys] = (
        f"<b>{cry_sys}</b><br>"
        f"{len(df_group):,} = {len(df_group) / len(df_diel):.0%}<br>"
    )


title = "<b>Refractive index distribution by crystal system</b>"
fig.layout.title = dict(text=title, x=0.5)
fig.layout.margin = dict(b=10, l=10, r=10, t=50)
fig.layout.showlegend = False
fig.layout.xaxis = reusable_x_axis = dict(
    tickvals=list(range(len(pmv.crystal_sys_order))), ticktext=list(x_ticks.values())
)


# pmv.save_fig(fig, "dielectric-violin.pdf")
fig.show()


# %%
fig = px.violin(
    df_diel,
    color=Key.crystal_system,
    x=Key.crystal_system,
    y=n_wyckoff_pos_key,
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
    df_diel.groupby(Key.crystal_system), key=lambda x: pmv.crystal_sys_order.index(x[0])
):
    avg_n_wyckoff = df_group[n_wyckoff_pos_key].mean()
    clr = rgb_color(avg_n_wyckoff, 14)
    x_ticks[cry_sys] = (
        f"<b>{cry_sys}</b><br>"
        f"{len(df_group):,} = {len(df_group) / len(df_diel):.0%}<br>"
        f"mean = <span style='color:{clr}'><b>{avg_n_wyckoff:.1f}</b></span>"
    )

title = "<b>Matbench dielectric: Number of Wyckoff positions by crystal system</b>"
fig.layout.title = dict(text=title, x=0.5)
fig.layout.margin = dict(b=10, l=10, r=10, t=50)
fig.layout.showlegend = False
fig.layout.update(width=1000, height=400, xaxis=reusable_x_axis)

# pmv.save_fig(fig, "dielectric-violin-num-wyckoffs.pdf")
fig.show()


# %%
fig = px.scatter(
    df_diel.round(2),
    x=Key.volume,
    y="n",
    color=Key.crystal_system,
    size="n",
    hover_data=[Key.spg_num],
    hover_name=Key.formula,
    range_x=[0, 1500],
)
title = "<b>Matbench Dielectric: Refractive Index vs. Volume</b>"
fig.layout.title = dict(text=title, x=0.5, font_size=20)
fig.layout.legend = dict(x=1, y=1, xanchor="right")

# slightly increase scatter point size (lower sizeref means larger)
fig.update_traces(marker_sizeref=0.08, selector=dict(mode="markers"))
fig.layout.margin.update(b=10, l=10, r=10, t=40)

# pmv.save_fig(fig, "dielectric-scatter.pdf")
fig.show()
