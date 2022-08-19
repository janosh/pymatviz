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
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from aviary.wren.utils import count_wyks, get_aflow_label_from_spglib
from matminer.datasets import load_dataset
from tqdm import tqdm

from pymatviz import (
    ptable_heatmap,
    ptable_heatmap_plotly,
    spacegroup_hist,
    spacegroup_sunburst,
)
from pymatviz.utils import get_crystal_sys


pio.templates.default = "plotly_white"

plt.rc("font", size=16)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("figure", dpi=150, titlesize=18)
plt.rcParams["figure.constrained_layout.use"] = True


# %%
df_diel = load_dataset("matbench_dielectric")

df_diel[["spg_symbol", "spg_num"]] = [
    struct.get_space_group_info()
    for struct in tqdm(df_diel.structure, desc="Getting spacegroups")
]

df_diel["wyckoff"] = [
    get_aflow_label_from_spglib(struct)
    for struct in tqdm(df_diel.structure, desc="Getting Wyckoff strings")
]
df_diel["n_wyckoff"] = df_diel.wyckoff.map(count_wyks)

df_diel["crystal_sys"] = df_diel.spg_num.map(get_crystal_sys)

df_diel["volume"] = [x.volume for x in df_diel.structure]
df_diel["formula"] = [x.formula for x in df_diel.structure]


# %%
ptable_heatmap(df_diel.formula, log=True)
plt.title("Elemental prevalence in the Matbench dielectric dataset")
plt.savefig("dielectric-ptable-heatmap.pdf")


# %%
fig = ptable_heatmap_plotly(df_diel.formula)
title = "Elements in Matbench Dielectric"
fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.4, y=0.94, font_size=20))
# fig.write_image("dielectric-ptable-heatmap-plotly.pdf")


# %%
ax = spacegroup_hist(df_diel.spg_num)
ax.set_title("Space group histogram", y=1.1)
plt.savefig("dielectric-spacegroup-hist.pdf")


# %%
fig = spacegroup_sunburst(df_diel.spg_num, show_counts="percent")
fig.update_layout(title="Space group sunburst")
# fig.write_image("dielectric-spacegroup-sunburst.pdf")
fig.show()


# %%
plot_labels = {
    "crystal_sys": "Crystal system",
    "n": "Refractive index n",
    "spg_num": "Space group",
    "n_wyckoff": "Number of Wyckoff positions",
}
cry_sys_order = (
    "cubic hexagonal trigonal tetragonal orthorhombic monoclinic triclinic".split()
)


# %%
fig = px.violin(
    df_diel,
    color="crystal_sys",
    x="crystal_sys",
    y="n",
    labels=plot_labels,
    points="all",
    hover_data=["spg_num"],
    hover_name="formula",
).update_traces(jitter=1)

x_ticks = {}  # custom x axis tick labels
for cry_sys, df_group in sorted(
    df_diel.groupby("crystal_sys"), key=lambda x: cry_sys_order.index(x[0])
):
    x_ticks[cry_sys] = (
        f"<b>{cry_sys}</b><br>"
        f"{len(df_group):,} = {len(df_group)/len(df_diel):.0%}<br>"
    )

xaxis = dict(tickvals=list(range(len(cry_sys_order))), ticktext=list(x_ticks.values()))
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
    color="crystal_sys",
    x="crystal_sys",
    y="n_wyckoff",
    labels=plot_labels,
    points="all",
    hover_data=["spg_num"],
    hover_name="formula",
    category_orders={"crystal_sys": cry_sys_order},
    log_y=True,
).update_traces(jitter=1)


def rgb_color(val: float, max: float) -> str:
    """Convert a value between 0 and max to a color between red and blue."""
    return f"rgb({255 * val / max:.1f}, 0, {255 * (max - val) / max:.1f})"


x_ticks = {}
for cry_sys, df_group in sorted(
    df_diel.groupby("crystal_sys"), key=lambda x: cry_sys_order.index(x[0])
):
    n_wyckoff = df_group.n_wyckoff.mean()
    clr = rgb_color(n_wyckoff, 14)
    x_ticks[cry_sys] = (
        f"<b>{cry_sys}</b><br>"
        f"{len(df_group):,} = {len(df_group)/len(df_diel):.0%}<br>"
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
    x="volume",
    y="n",
    color="crystal_sys",
    labels=plot_labels,
    size="n",
    hover_data=["spg_num"],
    hover_name="formula",
    range_x=[0, 1500],
)
title = "Matbench Dielectric: Refractive Index vs. Volume"
fig.update_layout(
    title=dict(text=f"<b>{title}</b>", x=0.5, font_size=20),
    legend=dict(x=1, y=1, xanchor="right"),
)
# slightly increase scatter point size (lower sizeref means larger)
fig.update_traces(marker_sizeref=0.08, selector=dict(mode="markers"))

# fig.write_image("dielectric-scatter.pdf")
fig.show()
