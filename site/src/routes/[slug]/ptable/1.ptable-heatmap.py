# Periodic Table Heatmap

from matminer.datasets import load_dataset

from pymatviz.ptable import ptable_heatmap
from pymatviz.utils import save_and_compress_svg


df_expt_gap = load_dataset("matbench_expt_gap")

ax = ptable_heatmap(df_expt_gap.composition, log=True)
title = (
    f"Elements in Matbench Experimental Band Gap ({len(df_expt_gap):,} compositions)"
)
ax.set_title(title, fontsize=16, fontweight="bold")
save_and_compress_svg(ax, "ptable-heatmap")
