# Periodic Table Heatmap

from matminer.datasets import load_dataset

from pymatviz.ptable import ptable_heatmap
from pymatviz.utils import save_and_compress_svg


df_expt_gap = load_dataset("matbench_expt_gap")

ax = ptable_heatmap(
    df_expt_gap.composition, heat_mode="percent", exclude_elements=["O"]
)
title = "Elements in Matbench Experimental Band Gap (percent)"
ax.set_title(title, fontsize=16, fontweight="bold")
save_and_compress_svg(ax, "ptable-heatmap-percent")
