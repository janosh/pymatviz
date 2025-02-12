"""Plot treemap distribution of chemical systems."""

# %%
from __future__ import annotations

import pandas as pd
import plotly.express as px

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.utils import ROOT


pmv.set_plotly_template("plotly_dark")


# %% Basic example with different group_by options
formulas = (
    "Pb(Zr0.52Ti0.48)O3 La0.7Sr0.3MnO3 Li0.5Na0.5O LiNaO2 Li2O LiFeO2 "
    "LiFeO3 Al2O3 MgO".split()
)
for group_by in ("formula", "reduced_formula", "chem_sys"):
    fig = pmv.chem_sys_treemap(formulas, group_by=group_by, show_counts="value+percent")
    title = f"Basic tree map grouped by {group_by}"
    fig.layout.title = dict(text=title, x=0.5, y=0.8, font_size=18)
    fig.show()
    group_suffix = group_by.replace("_", "-").replace("chem_sys", "")
    # pmv.io.save_and_compress_svg(fig, f"chem-sys-treemap-{group_suffix}")


# %% Load the Ward metallic glass dataset https://pubs.acs.org/doi/10.1021/acs.chemmater.6b04153
data_path = "ward-metallic-glasses/ward-metallic-glasses-set.csv.xz"
df_mg = pd.read_csv(
    f"{ROOT}/examples/dataset_exploration/{data_path}", na_values=()
).query("comment.isna()")

fig = pmv.chem_sys_treemap(
    df_mg[Key.composition],
    group_by="chem_sys",
    show_counts="value+percent",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
title = "Ward Metallic Glass Dataset - Grouped by chemical system"
fig.layout.title = dict(text=title, x=0.5, y=0.85, font_size=18)
fig.layout.update(height=500)
fig.show()


# %% Create a plot focusing on glass-forming ability (GFA)
for key, df_sub in df_mg.groupby("gfa_type"):
    fig = pmv.chem_sys_treemap(
        df_sub[Key.composition],
        show_counts="value+percent",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    title = f"Ward Metallic Glass Dataset - {key} Compositions"
    fig.layout.title = dict(text=title, x=0.5, y=0.8, font_size=18)
    fig.show()
    # pmv.io.save_and_compress_svg(fig, f"chem-sys-treemap-ward-bmg-{key.lower()}")
