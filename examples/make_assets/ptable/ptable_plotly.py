# %%
import os

import numpy as np
import yaml
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import ElemCountMode, Key


df_expt_gap = load_dataset("matbench_expt_gap")
module_dir = os.path.dirname(__file__)


# %% Plotly interactive periodic table heatmap
fig = pmv.ptable_heatmap_plotly(
    pmv.df_ptable[Key.atomic_mass],
    hover_props=[Key.atomic_mass, Key.atomic_number],
    hover_data="density = " + pmv.df_ptable[Key.density].astype(str) + " g/cm^3",
    show_values=False,
)
fig.layout.title = dict(text="<b>Atomic mass heatmap</b>", x=0.4, y=0.94, font_size=20)

fig.show()
pmv.io.save_and_compress_svg(fig, "ptable-heatmap-plotly-more-hover-data")


# %%
fig = pmv.ptable_heatmap_plotly(df_expt_gap[Key.composition], heat_mode="percent")
title = "Elements in Matbench Experimental Bandgap"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.4, y=0.94, font_size=20)
fig.show()
pmv.io.save_and_compress_svg(fig, "ptable-heatmap-plotly-percent-labels")


# %%
fig = pmv.ptable_heatmap_plotly(
    df_expt_gap[Key.composition], log=True, colorscale="viridis"
)
title = "Elements in Matbench Experimental Bandgap (log scale)"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.45, y=0.94, font_size=20)
fig.show()
pmv.io.save_and_compress_svg(fig, "ptable-heatmap-plotly-log")


# %% ex 1: Electronegativity Heatmap with Custom Hover Data
fig = pmv.ptable_heatmap_plotly(
    pmv.df_ptable[Key.electronegativity],
    colorscale="Viridis",
    hover_props=["atomic_mass", "melting_point"],
    hover_data={
        el: f"Fun fact about {el}!" for el in pmv.df_ptable[Key.electronegativity].index
    },
    font_colors=["white", "black"],
    color_bar=dict(title="Electronegativity", orientation="v"),
    font_size=11,
)
fig.show()


# %% ex 2: Log-scale Abundance with Excluded Elements
fig = pmv.ptable_heatmap_plotly(
    pmv.df_ptable[Key.specific_heat_capacity].dropna(),
    # colorscale="YlOrRd",
    log=True,
    font_colors=["black"],
    exclude_elements=["H", "He", "C", "O"],
    heat_mode="value",
    fmt=".2",
    color_bar=dict(title="Specific heat"),
    gap=3,
    # border=False
)
fig.show()


# %% ex 3: Fictional Data with Percent Mode and Custom Color Scale
rand_data = {
    elem: np.random.default_rng().random() * 100 for elem in pmv.df_ptable.index
}
custom_colorscale = [
    (0, "rgb(0,0,255)"),
    (0.25, "rgb(0,255,255)"),
    (0.5, "rgb(0,255,0)"),
    (0.75, "rgb(255,255,0)"),
    (1, "rgb(255,0,0)"),
]

fig = pmv.ptable_heatmap_plotly(
    rand_data,
    colorscale=custom_colorscale,
    heat_mode="percent",
    fmt=".2f",
    font_colors=["black"],
    bg_color="#f0f0f0",
)
fig.show()


# %% ex 4: Multi-element Compositions with Fraction Mode
fig = pmv.ptable_heatmap_plotly(
    df_expt_gap[Key.composition][:30],
    count_mode=ElemCountMode.fractional_composition,
    heat_mode="fraction",
    colorscale="plasma",
    show_values=True,
    fmt=".3f",
    hover_props={"electronegativity": "EN", "atomic_radius": "Radius (pm)"},
)
fig.show()


# %% ex 5: Atomic Radius with Custom Hover and Label Mapping
fig = pmv.ptable_heatmap_plotly(
    pmv.df_ptable[Key.atomic_radius],
    colorscale="RdYlBu",
    hover_data={
        elem: f"{radius} pm"
        for elem, radius in pmv.df_ptable[Key.atomic_radius].items()
    },
    font_colors=["black"],
    color_bar=dict(title="Atomic Radius (pm)"),
)
fig.show()


# %% ex 6: valence electrons in VASP PBE 64 pseudo-potentials
with open(f"{module_dir}/vasp-pbe-64-n-val-elecs.yml") as file:
    elem_to_n_val_elecs = yaml.safe_load(file)

elem_to_n_val_elecs = {  # convert Potcar symbol to element symbol
    key.split("_")[0]: val for key, val in elem_to_n_val_elecs.items()
}

fig = pmv.ptable_heatmap_plotly(elem_to_n_val_elecs, fmt=".0f")
title = (
    "Number of valence electrons in VASP PBE 64 pseudo-potentials<br>"
    "(Materials Project input set)"
)
fig.layout.title.update(text=title, x=0.4, y=0.92)
fig.show()

pmv.io.save_and_compress_svg(
    fig, "ptable-heatmap-plotly-vasp-psp-n-valence-electrons.pdf"
)
