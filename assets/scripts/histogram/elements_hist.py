# %%
from matminer.datasets import load_dataset

import pymatviz as pmv
from pymatviz.enums import Key


# %% Histogram Plots
df_expt_gap = load_dataset("matbench_expt_gap")

ax = pmv.elements_hist(
    df_expt_gap[Key.composition], keep_top=15, v_offset=200, rotation=0, fontsize=12
)
pmv.io.save_and_compress_svg(ax, "elements-hist")
