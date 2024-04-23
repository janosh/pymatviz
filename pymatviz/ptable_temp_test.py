# %%
import random

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio
from matminer.datasets import load_dataset
from pymatgen.core.periodic_table import Element
from tqdm import tqdm

from pymatviz.io import save_and_compress_svg
from pymatviz.ptable import (
    ptable_heatmap_plotly,
    ptable_hists,
    ptable_lines,
    ptable_scatters,
    ptable_splits,
)
from pymatviz.utils import df_ptable


# %% Configure matplotlib and load test data
plt.rc("font", size=14)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("axes", titlesize=16, titleweight="bold")
plt.rc("figure", dpi=200, titlesize=20, titleweight="bold")
plt.rcParams["figure.constrained_layout.use"] = True

px.defaults.template = "pymatviz_white"
pio.templates.default = "pymatviz_white"


df_steels = load_dataset("matbench_steels")
df_expt_gap = load_dataset("matbench_expt_gap")
df_phonons = load_dataset("matbench_phonons")

df_phonons[["spg_symbol", "spg_num"]] = [
    struct.get_space_group_info() for struct in tqdm(df_phonons.structure)
]


# %% Plotly interactive periodic table heatmap
fig = ptable_heatmap_plotly(
    df_ptable.atomic_mass,
    hover_props=["atomic_mass", "atomic_number"],
    hover_data="density = " + df_ptable.density.astype(str) + " g/cm^3",
    show_values=False,
)
fig.update_layout(
    title=dict(text="<b>Atomic mass heatmap</b>", x=0.4, y=0.94, font_size=20)
)
fig.show()
save_and_compress_svg(fig, "ptable-heatmap-plotly-more-hover-data")

fig = ptable_heatmap_plotly(df_expt_gap.composition, heat_mode="percent")
title = "Elements in Matbench Experimental Bandgap"
fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.4, y=0.94, font_size=20))
fig.show()
save_and_compress_svg(fig, "ptable-heatmap-plotly-percent-labels")

fig = ptable_heatmap_plotly(df_expt_gap.composition, log=True, colorscale="viridis")
title = "Elements in Matbench Experimental Bandgap (log scale)"
fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.4, y=0.94, font_size=20))
fig.show()
save_and_compress_svg(fig, "ptable-heatmap-plotly-log")


# %% Histograms laid out in as a periodic table
# Generate random parity data with y \approx x with some noise
data_dict = {
    elem.symbol: np.random.randn(100) + np.random.randn(100) for elem in Element
}
fig = ptable_hists(
    data_dict, colormap="coolwarm", cbar_title="Periodic Table Histograms"
)
save_and_compress_svg(fig, "ptable-hists")


# %% Scatter plots laid out as a periodic table
data_dict = {
    elem.symbol: [
        np.random.randint(0, 20, 10),
        np.random.randint(0, 20, 10),
        # np.random.randint(0, 20, 10),  # TODO:
    ]
    for elem in Element
}

fig = ptable_scatters(
    data_dict,
    # colormap="coolwarm",
    # cbar_title="Periodic Table Scatter Plots",
    child_args=dict(marker="o", linestyle=""),
)
save_and_compress_svg(fig, "ptable-scatters")


# %% Line plots laid out as a periodic table
data_dict = {
    elem.symbol: [
        np.linspace(0, 10, 10),
        np.sin(2 * np.pi * np.linspace(0, 10, 10)) + np.random.normal(0, 0.2, 10),
    ]
    for elem in Element
}

fig = ptable_lines(
    data_dict,
)
save_and_compress_svg(fig, "ptable-lines")


# %% Evenly-split tile plots laid out as a periodic table
data_dict = {
    elem.symbol: [
        random.randint(0, 10),
        random.randint(10, 20),
        # random.randint(20, 30),
    ]
    for elem in Element
}

fig = ptable_splits(
    data=data_dict,
    colormap="coolwarm",
    start_angle=135,
    cbar_title="Periodic Table Evenly-Split Tiles Plots",
)
save_and_compress_svg(fig, "ptable-splits")
