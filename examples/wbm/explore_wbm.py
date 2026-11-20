"""The WBM dataset named after the author initials was published in
https://nature.com/articles/s41524-020-00481-6.

See also https://matbench-discovery.materialsproject.org/data.
"""

# %%
from typing import get_args

import pandas as pd

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.typing import CrystalSystem


crystal_systems = get_args(CrystalSystem)  # ordered triclinic to cubic


__author__ = "Janosh Riebesell"
__date__ = "2022-08-18"


# %% download wbm-summary.csv (12 MB)
df_wbm = pd.read_csv("https://figshare.com/ndownloader/files/44225498")
df_wbm = df_wbm.set_index(Key.mat_id, drop=False)

df_wbm["batch_idx"] = df_wbm.index.str.split("-").str[2].astype(int)
df_wbm[Key.spg_num] = df_wbm[f"{Key.wyckoff}_spglib"].str.split("_").str[2].astype(int)
df_wbm[Key.crystal_system] = df_wbm[Key.spg_num].map(pmv.utils.spg_to_crystal_sys)
df_wbm[Key.energy_per_atom] = df_wbm[Key.energy] / df_wbm[Key.n_sites]


# %%
df_wbm.hist(bins=100, figsize=(16, 10))


# %%
widget = pmv.ptable_heatmap(
    df_wbm[Key.formula], color_bar={"title": "Elements in WBM dataset"}
)
widget.show()


# %% plot elemental prevalence heatmap by iteration number of the elemental substitution
for idx, df in df_wbm.groupby("batch_idx"):
    title = f"Elements in WBM batch {idx} of size {len(df):,}"
    pmv.ptable_heatmap(df[Key.formula], color_bar={"title": title}).show()


# %%
df_wbm_non_metals = df_wbm.query("bandgap_pbe > 0")

# Energy distribution by crystal system
# MatterViz has no rich-text tick labels, so per-group counts go into series labels
series = []
for cry_sys, df_group in sorted(
    df_wbm_non_metals.groupby(Key.crystal_system),
    key=lambda x: crystal_systems.index(x[0]),
):
    frac = len(df_group) / len(df_wbm_non_metals)
    label = f"{cry_sys} {len(df_group):,} = {frac:.0%}"
    series += [{"y": df_group["e_form_per_atom_wbm"], "label": label}]

widget = pmv.BoxPlotWidget(
    series,
    kind="violin",
    title="Energy distribution by crystal system",
    show_legend=False,
    y_axis={"label": "e_form_per_atom_wbm"},
)
widget.show()
# widget.to_img("wbm-energy-violin-by-crystal-sys.pdf")


# %%
widget = pmv.spacegroup_sunburst(df_wbm[Key.spg_num], label_text="label+percent")
widget.show()
widget.to_img("wbm-spacegroup-sunburst.pdf")


# %% WBM volume and energy by spacegroup number
for y_key in (Key.volume, Key.energy):
    widget = pmv.ScatterPlotWidget(
        series=[
            {
                "x": df_group[Key.spg_num],
                "y": df_group[y_key],
                "label": cry_sys,
                "markers": "points",
            }
            for cry_sys, df_group in df_wbm.groupby(Key.crystal_system)
        ],
        x_axis={"label": Key.spg_num.label},
        y_axis={"label": y_key.label},
    )
    widget.show()
