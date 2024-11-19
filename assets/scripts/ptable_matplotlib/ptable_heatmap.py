# %%
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


# %%
df_expt_gap = load_dataset("matbench_expt_gap")


# %% Elemental Plots
ax = pmv.ptable_heatmap(
    pmv.count_elements(df_expt_gap[Key.composition]),
    log=True,
    return_type="axes",  # TODO: change to return Figure after 2025-07-01
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
