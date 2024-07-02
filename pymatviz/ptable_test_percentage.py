import matplotlib.pyplot as plt

from pymatviz.ptable import ptable_heatmap
from pymatviz.utils import df_ptable


ax = ptable_heatmap(
    df_ptable.atomic_mass,
    colormap="coolwarm",
    values_show_mode="percent",
)
plt.savefig("ptable_test_percentage.png")
