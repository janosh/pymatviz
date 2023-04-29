# Periodic Table Heatmap

from matminer.datasets import load_dataset

from pymatviz.histograms import hist_elemental_prevalence
from pymatviz.utils import save_and_compress_svg


df_expt_gap = load_dataset("matbench_expt_gap")

ax = hist_elemental_prevalence(df_expt_gap.composition, keep_top=15, v_offset=1)
save_and_compress_svg(ax, "hist-elemental-prevalence")
