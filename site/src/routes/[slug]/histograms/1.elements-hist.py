# Periodic Table Heatmap

from matminer.datasets import load_dataset

from pymatviz.histograms import elements_hist
from pymatviz.io import save_and_compress_svg


df_expt_gap = load_dataset("matbench_expt_gap")

ax = elements_hist(df_expt_gap.composition, keep_top=15, v_offset=1)
save_and_compress_svg(ax, "hist-elemental-prevalence")
