# %%
import numpy as np
from matminer.datasets import load_dataset
from pymatgen.core.periodic_table import Element

import pymatviz as pmv
from pymatviz.enums import Key


# %%
np_rng = np.random.default_rng(seed=0)
df_expt_gap = load_dataset("matbench_expt_gap")
df_steels = load_dataset("matbench_steels")


# %% Elemental Plots
ax = pmv.ptable_heatmap(
    pmv.count_elements(df_expt_gap[Key.composition]),
    log=True,
)
title = (
    f"Elements in Matbench Experimental Band Gap ({len(df_expt_gap):,} compositions)"
)
ax.set_title(title, x=0.75, y=2.5, fontsize=18, fontweight="bold")
pmv.io.save_and_compress_svg(ax, "ptable-heatmap")


# %%
fig = pmv.ptable_heatmap(pmv.df_ptable[Key.atomic_mass], return_type="figure")
fig.suptitle("Atomic Mass Heatmap", y=0.96, fontsize=20, fontweight="bold")
pmv.io.save_and_compress_svg(fig, "ptable-heatmap-atomic-mass")


# %%
# Filter out near-zero entries
ptable_data = pmv.count_elements(df_expt_gap[Key.composition])
ptable_data = ptable_data[ptable_data > 0.01]

fig = pmv.ptable_heatmap(
    ptable_data,
    value_show_mode="percent",
    exclude_elements=["O"],
    return_type="figure",
)
title = "Elements in Matbench Experimental Band Gap (percent)"
fig.suptitle(title, y=0.96, fontsize=20, fontweight="bold")
pmv.io.save_and_compress_svg(fig, "ptable-heatmap-percent")


# %%
fig = pmv.ptable_heatmap_ratio(
    df_expt_gap[Key.composition], df_steels[Key.composition], log=True, value_fmt=".4g"
)
title = "Element ratios in Matbench Experimental Band Gap vs Matbench Steel"
fig.suptitle(title, y=0.96, fontsize=16, fontweight="bold")
pmv.io.save_and_compress_svg(fig, "ptable-heatmap-ratio")


# %% Histograms laid out in as a periodic table
# Generate random parity data with y \approx x with some noise
data_dict = {
    elem.symbol: np_rng.standard_normal(100) + np_rng.standard_normal(100)
    for elem in Element
}
fig = pmv.ptable_hists(
    data_dict,
    colormap="coolwarm",
    cbar_title="Periodic Table Histograms",
    cbar_axis="x",
    color_elem_strategy="background",
    add_elem_type_legend=True,
    # x_range=(0, None),
)
pmv.io.save_and_compress_svg(fig, "ptable-hists")


# %% Scatter plots laid out as a periodic table
data_dict = {  # random parity data with y = x + noise
    elem.symbol: [
        np.arange(10) + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 1, 10),
    ]
    for elem in Element
}

fig = pmv.ptable_scatters(
    data_dict,
    colormap="coolwarm",
    cbar_title="Periodic Table Scatter Plots",
    child_kwargs=dict(marker="o", linestyle="", s=10),
    symbol_pos=(0.5, 1.2),
    symbol_kwargs=dict(fontsize=14),
)
pmv.io.save_and_compress_svg(fig, "ptable-scatters-parity")


# %% 2nd pmv.ptable_scatters example
data_dict = {  # random parabola data with y = x^2 + noise
    elem.symbol: [
        np.arange(10),
        (np.arange(10) - 4) ** 2 + np_rng.normal(0, 1, 10),
        np.arange(10) + np_rng.normal(0, 10, 10),
    ]
    for elem in Element
}

fig = pmv.ptable_scatters(
    data_dict,
    colormap="inferno",
    cbar_title="Periodic Table Scatter Plots",
    child_kwargs=dict(marker="o", linestyle="", s=8, alpha=1),
    symbol_pos=(0.5, 1.2),
    symbol_kwargs=dict(fontsize=14),
    color_elem_strategy="off",
)
pmv.io.save_and_compress_svg(fig, "ptable-scatters-parabola")


# %% Line plots laid out as a periodic table
data_dict = {
    elem.symbol: [
        np.linspace(0, 10, 10),
        np.sin(2 * np.pi * np.linspace(0, 10, 10)) + np_rng.normal(0, 0.2, 10),
    ]
    for elem in Element
}

fig = pmv.ptable_lines(
    data_dict,
    symbol_pos=(0.5, 1.2),
    symbol_kwargs=dict(fontsize=14),
)
pmv.io.save_and_compress_svg(fig, "ptable-lines")


# %% Evenly-split tile plots laid out as a periodic table
rng = np.random.default_rng()
for n_splits in (2, 3, 4):
    data_dict = {
        elem.symbol: rng.integers(10 * n_splits, 20 * (n_splits + 1), size=n_splits)
        for elem in Element
    }

    fig = pmv.ptable_heatmap_splits(
        data=data_dict,
        colormap="coolwarm",
        start_angle=135 if n_splits % 2 == 0 else 90,
        cbar_title="Periodic Table Evenly-Split Heatmap Plots",
        hide_f_block=True,
    )
    fig.show()
    pmv.io.save_and_compress_svg(fig, f"ptable-heatmap-splits-{n_splits}")
