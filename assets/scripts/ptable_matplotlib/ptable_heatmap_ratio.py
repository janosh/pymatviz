# %%
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


# %%
df_expt_gap = load_dataset("matbench_expt_gap")
df_steels = load_dataset("matbench_steels")


# %%
fig = pmv.ptable_heatmap_ratio(
    df_expt_gap[Key.composition], df_steels[Key.composition], log=True, value_fmt=".4g"
)
title = "Element ratios in Matbench Experimental Band Gap vs Matbench Steel"
fig.suptitle(title, y=0.96, fontsize=16, fontweight="bold")
pmv.io.save_and_compress_svg(fig, "ptable-heatmap-ratio")
