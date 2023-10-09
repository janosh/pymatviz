# Periodic Table Heatmap

from pymatviz.io import save_and_compress_svg
from pymatviz.ptable import ptable_heatmap
from pymatviz.utils import df_ptable


ax = ptable_heatmap(df_ptable.atomic_mass)
ax.set_title("Atomic Mass Heatmap", fontsize=16, fontweight="bold")
save_and_compress_svg(ax, "ptable-heatmap-atomic-mass")
