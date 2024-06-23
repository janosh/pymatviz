# %%
import pandas as pd
import plotly.express as px

from pymatviz import crystal_sys_order, ptable_heatmap_plotly, spacegroup_sunburst
from pymatviz.enums import Key
from pymatviz.utils import crystal_sys_from_spg_num


__author__ = "Janosh Riebesell"
__date__ = "2022-08-18"


# %% download wbm-summary.csv (12 MB)
df_wbm = pd.read_csv("https://figshare.com/ndownloader/files/44225498").set_index(
    Key.mat_id, drop=False
)

df_wbm["batch_idx"] = df_wbm.index.str.split("-").str[2].astype(int)
df_wbm[Key.spacegroup] = df_wbm.wyckoff.str.split("_").str[2].astype(int)
df_wbm[Key.crystal_system] = df_wbm[Key.spacegroup].map(crystal_sys_from_spg_num)
df_wbm[Key.energy_per_atom] = df_wbm.energy / df_wbm.n_sites


# %%
df_wbm.hist(bins=100, figsize=(16, 10))


# %%
fig = ptable_heatmap_plotly(df_wbm.formula)
title = "<b>Elements in WBM Dataset</b>"
fig.layout.title = dict(text=title, x=0.4, y=0.94, font_size=20)
# fig.write_image("wbm-ptable-heatmap-plotly.pdf")


# %% plot elemental prevalence heatmap by iteration number of the elemental substitution
for idx, df in df_wbm.groupby("batch_idx"):
    fig = ptable_heatmap_plotly(df.formula)
    title = f"<b>Elements in WBM batch {idx} of size {len(df):,}</b>"
    fig.layout.title = dict(text=title, x=0.4, y=0.94, font_size=20)
    # fig.write_image(f"wbm-ptable-heatmap-plotly-batch-{idx}.pdf")
    fig.show()


# %%
df_wbm_non_metals = df_wbm.query("bandgap_pbe > 0")

fig = px.violin(
    df_wbm_non_metals,
    color=Key.crystal_system,
    x=Key.crystal_system,
    y=Key.energy_per_atom,
    hover_name=Key.formula,
).update_traces(jitter=1)

x_ticks = {}  # custom x axis tick labels
for cry_sys, df_group in sorted(
    df_wbm_non_metals.groupby(Key.crystal_system),
    key=lambda x: crystal_sys_order.index(x[0]),
):
    x_ticks[cry_sys] = (
        f"<b>{cry_sys}</b><br>"
        f"{len(df_group):,} = {len(df_group) / len(df_wbm_non_metals):.0%}<br>"
    )

xaxis = dict(
    tickvals=list(range(len(crystal_sys_order))), ticktext=list(x_ticks.values())
)
fig.update_layout(
    title="<b>Energy distribution by crystal system</b>",
    title_x=0.5,
    margin=dict(b=10, l=10, r=10, t=50),
    showlegend=False,
    xaxis=xaxis,
)
fig.update_traces(hoverinfo="skip", hovertemplate=None)

# fig.write_image("wbm-energy-violin-by-crystal-sys.pdf")
fig.show()


# %%
fig = spacegroup_sunburst(df_wbm[Key.spacegroup], show_counts="percent")
fig.layout.title = "Matbench Perovskites spacegroup sunburst"

fig.write_image("wbm-spacegroup-sunburst.pdf")
fig.show()


# %%
fig = px.scatter(df_wbm, x=Key.spacegroup, y=Key.volume, color=Key.crystal_system)

fig.layout.title = "WBM volume by spacegroup number"


# %%
fig = px.scatter(df_wbm, x=Key.spacegroup, y=Key.energy, color=Key.crystal_system)

fig.layout.title = "WBM energy by spacegroup number"
