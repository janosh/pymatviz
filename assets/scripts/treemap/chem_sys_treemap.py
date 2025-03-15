"""Plot treemap distribution of chemical systems."""

# %%
from __future__ import annotations

import pandas as pd
import plotly.express as px

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.utils import ROOT


pmv.set_plotly_template("plotly_dark")


# %% Basic example with different group_by options and customizations
formulas = (
    "Pb(Zr0.52Ti0.48)O3 La0.7Sr0.3MnO3 Li0.5Na0.5O LiNaO2 Li2O LiFeO2 "  # noqa: SIM905
    "LiFeO3 Al2O3 MgO".split()
)
for group_by in ("formula", "reduced_formula", "chem_sys"):
    fig = pmv.chem_sys_treemap(formulas, group_by=group_by, show_counts="value+percent")
    # Add customizations: rounded corners and custom hover info
    fig.update_traces(
        marker=dict(cornerradius=5),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: "
        "%{percentRoot:.1%} of total<extra></extra>",
    )
    title = f"Basic tree map grouped by {group_by} with rounded corners"
    fig.layout.title = dict(text=title, x=0.5, y=0.8, font_size=18)
    fig.show()
    group_suffix = group_by.replace("_", "-").replace("chem_sys", "")
    # pmv.io.save_and_compress_svg(fig, f"chem-sys-treemap-{group_suffix}")


# %% Load the Ward metallic glass dataset https://pubs.acs.org/doi/10.1021/acs.chemmater.6b04153
data_path = "ward_metallic_glasses/ward-metallic-glasses.csv.xz"
df_mg = pd.read_csv(
    f"{ROOT}/examples/dataset_exploration/{data_path}", na_values=()
).query("comment.isna()")

fig = pmv.chem_sys_treemap(
    df_mg[Key.composition],
    group_by="chem_sys",
    show_counts="value+percent",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
# Add customizations: custom text display and root color
fig.update_traces(textinfo="label+value+percent entry", root_color="lightgrey")
title = "Ward Metallic Glass Dataset - With custom text display and root color"
fig.layout.title = dict(text=title, x=0.5, y=0.85, font_size=18)
fig.layout.update(height=500)
fig.show()


# %% Create a plot focusing on glass-forming ability (GFA) with patterns
for key, df_sub in df_mg.groupby("gfa_type"):
    fig = pmv.chem_sys_treemap(
        df_sub[Key.composition],
        show_counts="value+percent",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    # Add customizations: patterns/textures and maximum depth
    patterns = {
        "unary": "|",
        "binary": "/",
        "ternary": "x",
        "quaternary": "+",
        "quinary": ".",
    }
    fig.update_traces(
        maxdepth=2,  # Limit depth for clarity
        marker_pattern_shape=[
            next((val for key, val in patterns.items() if key in parent), "")
            for parent in fig.data[0].parents
        ],
    )
    title = f"Ward Metallic Glass Dataset - {key} Compositions<br>"
    title += "with patterns and limited depth"
    fig.layout.title = dict(text=title, x=0.5, y=0.8, font_size=18)
    fig.show()
    # pmv.io.save_and_compress_svg(fig, f"chem-sys-treemap-ward-bmg-{key.lower()}")


# %% Demonstrate the max_cells parameter with custom hover and rounded corners
fig = pmv.chem_sys_treemap(
    df_mg[Key.composition],
    group_by="chem_sys",
    show_counts="value+percent",
    max_cells=5,  # Limit systems per arity
    color_discrete_sequence=px.colors.qualitative.Set2,
)
# Add customizations: rounded corners and hover info
fig.update_traces(
    marker=dict(cornerradius=8),
    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentRoot:.1%}"
    " of total<extra></extra>",
)
title = "Ward Metallic Glass Dataset - Top 5 systems per arity with rounded corners"
fig.layout.title = dict(text=title, x=0.5, y=0.85, font_size=18)
fig.layout.update(height=500)
fig.show()
pmv.io.save_and_compress_svg(fig, "chem-sys-treemap-top-5")


# %% Custom color mapping with additional customizations
# Create a base treemap to customize
formulas = [
    "Fe2O3",  # binary
    "Fe4O6",  # same as Fe2O3 when group_by="reduced_formula"
    "FeO",  # different formula but same system when group_by="chem_sys"
    "Li2O",  # binary
    "LiFeO2",  # ternary
    "Li3FeO3",  # ternary (same system as LiFeO2)
    "Al2O3",  # binary
    "MgO",  # binary
    "SiO2",  # binary
]

fig = pmv.chem_sys_treemap(formulas)
# Create a custom color map for specific chemical systems
color_map = {
    "Fe-O": "red",
    "Li-O": "blue",
    "Fe-Li-O": "purple",
    "Al-O": "green",
    "Mg-O": "orange",
    "O-Si": "yellow",
}
# Initialize the colors array with None values
colors = [color_map.get(label) for label in fig.data[0].labels]
# Add multiple customizations: custom colors, text display, and rounded corners
fig.update_traces(
    marker=dict(colors=colors, cornerradius=5),
    textinfo="label+value+percent entry",
    root_color="lightgrey",
)
fig.layout.title = dict(
    text="Treemap with Custom Color Mapping, Text Display and Root Color",
    x=0.5,
    y=0.85,
    font_size=18,
)
fig.show()


# %% Comprehensive example with multiple customizations
fig = pmv.chem_sys_treemap(
    formulas,
    color_discrete_sequence=px.colors.qualitative.Pastel,
)
# many customizations: patterns, hover info, rounded corners, and custom text
patterns = {"unary": "|", "binary": "/", "ternary": "x"}
fig.update_traces(
    marker=dict(cornerradius=10),
    marker_pattern_shape=[
        next((val for key, val in patterns.items() if key in parent), "")
        for parent in fig.data[0].parents
    ],
    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: "
    "%{percentRoot:.1%} of total<extra></extra>",
    textinfo="label+value",
)
fig.layout.title = dict(
    text="Comprehensive Customization Example", x=0.5, y=0.85, font_size=18
)
fig.show()
