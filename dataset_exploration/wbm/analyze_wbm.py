# %%
import pandas as pd
import plotly.express as px
import plotly.io as pio

from pymatviz import ptable_heatmap_plotly, spacegroup_sunburst
from pymatviz.utils import get_crystal_sys


__author__ = "Janosh Riebesell"
__date__ = "2022-08-18"


# https://github.com/plotly/Kaleido/issues/122#issuecomment-994906924
pio.kaleido.scope.mathjax = None


# %% download wbm-steps-summary.csv (23.31 MB)
df_wbm = pd.read_csv(
    "https://figshare.com/ndownloader/files/36714216?private_link=ff0ad14505f9624f0c05"
).set_index("material_id", drop=False)

df_wbm["batch_idx"] = df_wbm.index.str.split("-").str[2].astype(int)
df_wbm["spg_num"] = df_wbm.wyckoff.str.split("_").str[2].astype(int)
df_wbm["crystal_sys"] = df_wbm.spg_num.map(get_crystal_sys)
df_wbm["energy_per_atom"] = df_wbm.energy / df_wbm.n_sites


# %%
df_wbm.hist(bins=100, figsize=(16, 10))


# %%
fig = ptable_heatmap_plotly(df_wbm.formula)
title = "Elements in WBM Dataset"
fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.4, y=0.94, font_size=20))
# fig.write_image("wbm-ptable-heatmap-plotly.pdf")


# %% plot elemental prevalence heatmap by iteration number of the elemental substitution
for idx, df in df_wbm.groupby("batch_idx"):
    fig = ptable_heatmap_plotly(df.formula)
    title = f"Elements in WBM batch {idx} of size {len(df):,}"
    fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.4, y=0.94, font_size=20))
    # fig.write_image(f"wbm-ptable-heatmap-plotly-batch-{idx}.pdf")
    fig.show()


# %%
plot_labels = {
    "crystal_sys": "Crystal system",
    "spg_num": "Space group",
    "n_wyckoff": "Number of Wyckoff positions",
    "n_sites": "Number of unit cell sites",
    "energy_per_atom": "Energy (eV/atom)",
}
cry_sys_order = (
    "cubic hexagonal trigonal tetragonal orthorhombic monoclinic triclinic".split()
)

df_wbm_non_metals = df_wbm.query("bandgap_pbe > 0")

fig = px.violin(
    df_wbm_non_metals,
    color="crystal_sys",
    x="crystal_sys",
    y="energy_per_atom",
    labels=plot_labels,
    hover_name="formula",
).update_traces(jitter=1)

x_ticks = {}  # custom x axis tick labels
for cry_sys, df_group in sorted(
    df_wbm_non_metals.groupby("crystal_sys"), key=lambda x: cry_sys_order.index(x[0])
):
    x_ticks[cry_sys] = (
        f"<b>{cry_sys}</b><br>"
        f"{len(df_group):,} = {len(df_group)/len(df_wbm_non_metals):.0%}<br>"
    )

xaxis = dict(tickvals=list(range(len(cry_sys_order))), ticktext=list(x_ticks.values()))
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
fig = spacegroup_sunburst(df_wbm.spg_num, show_counts="percent")
fig.update_layout(title="Matbench Perovskites spacegroup sunburst")

fig.write_image("wbm-spacegroup-sunburst.pdf")
fig.show()


# %%
fig = px.scatter(df_wbm, x="spg_num", y="volume", color="crystal_sys")

fig.update_layout(title="WBM volume by spacegroup number")


# %%
fig = px.scatter(df_wbm, x="spg_num", y="energy", color="crystal_sys")

fig.update_layout(title="WBM energy by spacegroup number")
