# Periodic Table Heatmap

from matminer.datasets import load_dataset

from pymatviz.ptable import ptable_heatmap_ratio
from pymatviz.utils import save_and_compress_svg


df_expt_gap = load_dataset("matbench_expt_gap")
df_steels = load_dataset("matbench_steels")

ax = ptable_heatmap_ratio(df_expt_gap.composition, df_steels.composition, log=True)
title = "Element ratios in Matbench Experimental Band Gap vs Matbench Steel"
ax.set_title(title, y=0.96)
save_and_compress_svg(ax, "ptable-heatmap-ratio")
