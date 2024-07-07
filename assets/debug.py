from matminer.datasets import load_dataset

from pymatviz.enums import Key
from pymatviz.io import save_and_compress_svg
from pymatviz.ptable import ptable_heatmap_ratio


df_expt_gap = load_dataset("matbench_expt_gap")
df_steels = load_dataset("matbench_steels")


fig = ptable_heatmap_ratio(
    df_expt_gap[Key.composition], df_steels[Key.composition], log=True, values_fmt=".4g"
)
title = "Element ratios in Matbench Experimental Band Gap vs Matbench Steel"
fig.suptitle(title, y=0.96, fontsize=20, fontweight="bold")
save_and_compress_svg(fig, "debug")
