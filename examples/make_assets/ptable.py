# %%
import random

import numpy as np
from matminer.datasets import load_dataset
from pymatgen.core.periodic_table import Element

from pymatviz.enums import Key
from pymatviz.io import save_and_compress_svg
from pymatviz.ptable import (
    PTableData,
    count_elements,
    ptable_heatmap,
    ptable_heatmap_plotly,
    ptable_heatmap_ratio,
    ptable_heatmap_splits,
    ptable_hists,
    ptable_lines,
    ptable_scatters,
)
from pymatviz.utils import df_ptable


# %%
np_rng = np.random.default_rng(seed=0)
df_expt_gap = load_dataset("matbench_expt_gap")
df_steels = load_dataset("matbench_steels")


# %% Elemental Plots
# DEBUG: the color mapping seems incorrect for log mode
fig = ptable_heatmap(count_elements(df_expt_gap[Key.composition]), log=True)
title = (
    f"Elements in Matbench Experimental Band Gap ({len(df_expt_gap):,} compositions)"
)
fig.suptitle(title, y=0.96, fontsize=16, fontweight="bold")
save_and_compress_svg(fig, "ptable-heatmap")


# %%
# DEBUG: 0 is not shown in cbar for some reason
fig = ptable_heatmap(df_ptable.atomic_mass, cbar_range=(0, 300))
fig.suptitle("Atomic Mass Heatmap", y=0.96, fontsize=16, fontweight="bold")
save_and_compress_svg(fig, "ptable-heatmap-atomic-mass")


# %%
# DEBUG: color mapping is incorrect (similar to ptable-heatmap)
data = PTableData(count_elements(df_expt_gap[Key.composition]))
data.drop_elements(["O"])

fig = ptable_heatmap(data, values_show_mode="percent")
title = "Elements in Matbench Experimental Band Gap (percent)"
fig.suptitle(title, y=0.96, fontsize=16, fontweight="bold")
save_and_compress_svg(fig, "ptable-heatmap-percent")


# %%
# TODO: migrate this
ax = ptable_heatmap_ratio(
    df_expt_gap[Key.composition], df_steels[Key.composition], log=True
)
title = "Element ratios in Matbench Experimental Band Gap vs Matbench Steel"
ax.set_title(title, y=0.96, fontsize=16, fontweight="bold")
save_and_compress_svg(ax, "ptable-heatmap-ratio")


# %% Plotly interactive periodic table heatmap
fig = ptable_heatmap_plotly(
    df_ptable.atomic_mass,
    hover_props=["atomic_mass", "atomic_number"],
    hover_data="density = " + df_ptable[Key.density].astype(str) + " g/cm^3",
    show_values=False,
)
fig.layout.title = dict(text="<b>Atomic mass heatmap</b>", x=0.4, y=0.94, font_size=20)

fig.show()
save_and_compress_svg(fig, "ptable-heatmap-plotly-more-hover-data")


# %%
fig = ptable_heatmap_plotly(df_expt_gap[Key.composition], heat_mode="percent")
title = "Elements in Matbench Experimental Bandgap"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.4, y=0.94, font_size=20)
fig.show()
save_and_compress_svg(fig, "ptable-heatmap-plotly-percent-labels")


# %%
fig = ptable_heatmap_plotly(
    df_expt_gap[Key.composition], log=True, colorscale="viridis"
)
title = "Elements in Matbench Experimental Bandgap (log scale)"
fig.layout.title = dict(text=f"<b>{title}</b>", x=0.4, y=0.94, font_size=20)
fig.show()
save_and_compress_svg(fig, "ptable-heatmap-plotly-log")


# %% Histograms laid out in as a periodic table
# Generate random parity data with y \approx x with some noise
data_dict = {
    elem.symbol: np_rng.standard_normal(100) + np_rng.standard_normal(100)
    for elem in Element
}
fig = ptable_hists(
    data_dict,
    colormap="coolwarm",
    cbar_title="Periodic Table Histograms",
    cbar_axis="x",
    color_elem_strategy="background",
    add_elem_type_legend=True,
    # x_range=(0, None),
)
save_and_compress_svg(fig, "ptable-hists")


# %% Scatter plots laid out as a periodic table
data_dict = {  # random parity data with y = x + noise
    elem.symbol: [
        np.arange(10) + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 1, 10),
    ]
    for elem in Element
}

fig = ptable_scatters(
    data_dict,
    colormap="coolwarm",
    cbar_title="Periodic Table Scatter Plots",
    child_kwargs=dict(marker="o", linestyle="", s=10),
    symbol_pos=(0.5, 1.2),
    symbol_kwargs=dict(fontsize=14),
)
save_and_compress_svg(fig, "ptable-scatters-parity")


# %% 2nd ptable_scatters example
data_dict = {  # random parabola data with y = x^2 + noise
    elem.symbol: [
        np.arange(10),
        (np.arange(10) - 4) ** 2 + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 10, 10),
    ]
    for elem in Element
}

fig = ptable_scatters(
    data_dict,
    colormap="inferno",
    cbar_title="Periodic Table Scatter Plots",
    child_kwargs=dict(marker="o", linestyle="", s=8, alpha=1),
    symbol_pos=(0.5, 1.2),
    symbol_kwargs=dict(fontsize=14),
    color_elem_strategy="off",
)
save_and_compress_svg(fig, "ptable-scatters-parabola")


# %% Line plots laid out as a periodic table
data_dict = {
    elem.symbol: [
        np.linspace(0, 10, 10),
        np.sin(2 * np.pi * np.linspace(0, 10, 10)) + np_rng.normal(0, 0.2, 10),
    ]
    for elem in Element
}

fig = ptable_lines(
    data_dict,
    symbol_pos=(0.5, 1.2),
    symbol_kwargs=dict(fontsize=14),
)
save_and_compress_svg(fig, "ptable-lines")


# %% Evenly-split tile plots laid out as a periodic table
for n_splits in (2, 3, 4):
    data_dict = {
        elem.symbol: [
            random.randint(10 * n_splits, 20 * (n_splits + 1)) for _ in range(n_splits)
        ]
        for elem in Element
    }

    fig = ptable_heatmap_splits(
        data=data_dict,
        colormap="coolwarm",
        start_angle=135 if n_splits % 2 == 0 else 90,
        cbar_title="Periodic Table Evenly-Split Heatmap Plots",
        hide_f_block=True,
    )
    fig.show()
    save_and_compress_svg(fig, f"ptable-heatmap-splits-{n_splits}")
