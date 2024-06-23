# %%
import matplotlib.pyplot as plt
import plotly.express as px
from aviary.wren.utils import count_wyckoff_positions, get_aflow_label_from_spglib
from matminer.datasets import load_dataset
from tqdm import tqdm

from pymatviz import (
    crystal_sys_order,
    ptable_heatmap,
    ptable_heatmap_plotly,
    spacegroup_hist,
    spacegroup_sunburst,
)
from pymatviz.enums import Key
from pymatviz.utils import crystal_sys_from_spg_num


"""matbench_dielectric dataset

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
df_diel = load_dataset("matbench_dielectric")

df_diel[[Key.spacegroup_symbol, Key.spacegroup]] = [
    struct.get_space_group_info()
    for struct in tqdm(df_diel[Key.structure], desc="Getting spacegroups")
]

df_diel["wyckoff"] = [
    get_aflow_label_from_spglib(struct)
    for struct in tqdm(df_diel[Key.structure], desc="Getting Wyckoff strings")
]
df_diel["n_wyckoff"] = df_diel.wyckoff.map(count_wyckoff_positions)

df_diel[Key.crystal_system] = df_diel[Key.spacegroup].map(crystal_sys_from_spg_num)

df_diel[Key.volume] = [x.volume for x in df_diel[Key.structure]]
df_diel[Key.formula] = [x.formula for x in df_diel[Key.structure]]


# %%
ptable_heatmap(df_diel.formula, log=True)
plt.title("Elemental prevalence in the Matbench dielectric dataset")
plt.savefig("dielectric-ptable-heatmap.pdf")


# %%
fig = ptable_heatmap_plotly(df_diel.formula, log=True, colorscale="viridis")
title = "<b>Elements in Matbench Dielectric</b>"
fig.layout.title = dict(text=title, x=0.4, y=0.94, font_size=20)
# fig.write_image("dielectric-ptable-heatmap-plotly.pdf")


# %%
ax = spacegroup_hist(df_diel[Key.spacegroup])
ax.set_title("Space group histogram", y=1.1)
plt.savefig("dielectric-spacegroup-hist.pdf")


# %%
fig = spacegroup_sunburst(df_diel[Key.spacegroup], show_counts="percent")
fig.layout.title = "Space group sunburst"
# fig.write_image("dielectric-spacegroup-sunburst.pdf")
fig.show()


# %%
fig = px.violin(
    df_diel,
    color=Key.crystal_system,
    x=Key.crystal_system,
    y="n",
    points="all",
    hover_data=[Key.spacegroup],
    hover_name=Key.formula,
).update_traces(jitter=1)

x_ticks = {}  # custom x axis tick labels
for cry_sys, df_group in sorted(
    df_diel.groupby(Key.crystal_system), key=lambda x: crystal_sys_order.index(x[0])
):
    x_ticks[cry_sys] = (
        f"<b>{cry_sys}</b><br>"
        f"{len(df_group):,} = {len(df_group) / len(df_diel):.0%}<br>"
    )

xaxis = dict(
    tickvals=list(range(len(crystal_sys_order))), ticktext=list(x_ticks.values())
)
fig.update_layout(
    title="<b>Refractive index distribution by crystal system</b>",
    title_x=0.5,
    margin=dict(b=10, l=10, r=10, t=50),
    showlegend=False,
    xaxis=xaxis,
)

# fig.write_image("dielectric-violin.pdf")
fig.show()


# %%
fig = px.violin(
    df_diel,
    color=Key.crystal_system,
    x=Key.crystal_system,
    y="n_wyckoff",
    points="all",
    hover_data=[Key.spacegroup],
    hover_name=Key.formula,
    category_orders={Key.crystal_system: crystal_sys_order},
    log_y=True,
).update_traces(jitter=1)


def rgb_color(val: float, max_val: float) -> str:
    """Convert a value between 0 and max to a color between red and blue."""
    return f"rgb({255 * val / max_val:.1f}, 0, {255 * (max_val - val) / max_val:.1f})"


x_ticks = {}
for cry_sys, df_group in sorted(
    df_diel.groupby(Key.crystal_system), key=lambda x: crystal_sys_order.index(x[0])
):
    n_wyckoff = df_group.n_wyckoff.mean()
    clr = rgb_color(n_wyckoff, 14)
    x_ticks[cry_sys] = (
        f"<b>{cry_sys}</b><br>"
        f"{len(df_group):,} = {len(df_group) / len(df_diel):.0%}<br>"
        f"mean = <span style='color:{clr}'><b>{n_wyckoff:.1f}</b></span>"
    )

fig.update_layout(
    title="<b>Matbench dielectric: Number of Wyckoff positions by crystal system</b>",
    title_x=0.5,
    margin=dict(b=10, l=10, r=10, t=50),
    showlegend=False,
    width=1000,
    height=400,
    xaxis=xaxis,
)

# fig.write_image("dielectric-violin-num-wyckoffs.pdf")
fig.show()


# %%
fig = px.scatter(
    df_diel.round(2),
    x=Key.volume,
    y="n",
    color=Key.crystal_system,
    size="n",
    hover_data=[Key.spacegroup],
    hover_name=Key.formula,
    range_x=[0, 1500],
)
title = "<b>Matbench Dielectric: Refractive Index vs. Volume</b>"
fig.update_layout(
    title=dict(text=title, x=0.5, font_size=20),
    legend=dict(x=1, y=1, xanchor="right"),
)
# slightly increase scatter point size (lower sizeref means larger)
fig.update_traces(marker_sizeref=0.08, selector=dict(mode="markers"))

# fig.write_image("dielectric-scatter.pdf")
fig.show()
