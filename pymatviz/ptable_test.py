import matplotlib.pyplot as plt

from pymatviz.ptable import ptable_heatmap
from pymatviz.utils import df_ptable


ax = ptable_heatmap(
    df_ptable.atomic_mass,
    colormap="coolwarm",
    values_show_mode="value",
    values_color="AUTO",
)
# ax.set_title("Atomic Mass Heatmap", y=0.96, fontsize=16, fontweight="bold")

plt.savefig("ptable_test.png")
