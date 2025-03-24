"""Plot sunburst distribution of chemical systems."""

# %%
from __future__ import annotations

import pandas as pd
import plotly.express as px

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.utils import ROOT


# %% Basic examples: Different grouping modes for the same formulas
formulas = [
    "Fe2O3",  # binary
    "Fe4O6",  # same as Fe2O3 when group_by="reduced_formula"
    "FeO",  # different formula but same system when group_by="chem_sys"
    "Li2O",  # binary
    "LiFeO2",  # ternary
]

# Count each formula separately
fig = pmv.chem_sys_sunburst(
    formulas,
    group_by="formula",
    title="Group by formula (count each formula separately)",
)
fig.show()

# Group by reduced formulas (Fe2O3 and Fe4O6 count as same)
fig = pmv.chem_sys_sunburst(
    formulas,
    group_by="reduced_formula",
    title="Group by reduced formula (same stoichiometry)",
)
fig.show()

# Group by chemical systems (default, all Fe-O formulas count as same)
fig = pmv.chem_sys_sunburst(
    formulas,
    group_by="chem_sys",
    title="Group by chemical system (same elements)",
)
fig.show()


# %% Example 2: Complex formulas with fractional occupancies
complex_formulas = [
    "Pb(Zr0.52Ti0.48)O3",  # PZT with fractional occupancy
    "La0.7Sr0.3MnO3",  # LSMO with fractional composition
    "Li0.5Na0.5O",  # mixed alkali
    "LiNaO2",  # same system, different notation
]
fig = pmv.chem_sys_sunburst(
    complex_formulas,
    show_counts="value+percent",
    title="Complex formulas with fractional occupancies",
)
fig.show()


# %% Load the Ward metallic glasses https://pubs.acs.org/doi/10.1021/acs.chemmater.6b04153
data_path = "ward_metallic_glasses/ward-metallic-glasses.csv.xz"
df_mg = pd.read_csv(
    f"{ROOT}/examples/dataset_exploration/{data_path}", na_values=()
).query("comment.isna()")

fig = pmv.chem_sys_sunburst(
    df_mg[Key.composition],
    group_by="chem_sys",
    show_counts="value+percent",
    title="Ward Metallic Glasses - Grouped by chemical system",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
title = "Ward Metallic Glasses - Grouped by chemical system"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18)
fig.layout.update(height=500)
fig.show()


# %% Create a plot focusing on glass-forming ability (GFA)
for key, df_group in df_mg.groupby("gfa_type"):
    fig = pmv.chem_sys_sunburst(
        df_group[Key.composition],
        show_counts="value+percent",
        title=f"Ward Metallic Glasses - {key} Compositions",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    title = f"Ward Metallic Glasses - {key} Compositions"
    fig.layout.title = dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18)
    fig.layout.update(height=500)
    fig.show()
    # pmv.io.save_and_compress_svg(fig, "chem-sys-sunburst-ward-bmg")
    break


# %% Show only top 3 systems per arity with "Other" slices
max_slices = 15
fig = pmv.chem_sys_sunburst(
    df_mg[Key.composition],
    group_by="chem_sys",
    show_counts="value+percent",
    max_slices=max_slices,
    max_slices_mode="other",  # combine remaining systems into "Other" slices
    color_discrete_sequence=px.colors.qualitative.Set2,
)
title = f"Ward Metallic Glasses - Top {max_slices} Systems per Arity (with Other)"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.5, y=0.96, font_size=18)
fig.layout.update(height=500)
fig.show()
# pmv.io.save_and_compress_svg(fig, "chem-sys-sunburst-ward-bmg-top3-other")


# %% Show only top 3 systems per arity, dropping the rest
max_slices = 12
fig = pmv.chem_sys_sunburst(
    df_mg[Key.composition],
    group_by="chem_sys",
    show_counts="value+percent",
    max_slices=max_slices,
    max_slices_mode="drop",  # discard remaining systems
    color_discrete_sequence=px.colors.qualitative.Set2,
)
title = f"Ward Metallic Glasses<br>Top {max_slices} Systems per Arity (dropped)"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.5, y=0.9, font_size=18)
fig.layout.update(height=500)
fig.show()
# pmv.io.save_and_compress_svg(fig, "chem-sys-sunburst-ward-bmg-top3-drop")
