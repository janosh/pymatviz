# Periodic Table Heatmap

from matminer.datasets import load_dataset

from pymatviz.ptable import ptable_heatmap
from pymatviz.utils import df_ptable, save_and_compress_svg


df_expt_gap = load_dataset("matbench_expt_gap")

ax = ptable_heatmap(df_ptable.atomic_mass)
ax.set_title("Atomic Mass Heatmap", fontsize=16, fontweight="bold")
save_and_compress_svg(ax, "ptable-heatmap")
