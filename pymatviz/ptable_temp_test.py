"""Temporary asset generator, to be merged and removed."""

# %%
import random

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core.periodic_table import Element

from pymatviz.io import save_and_compress_svg
from pymatviz.ptable import ptable_lines, ptable_scatters, ptable_splits


# %% Configure matplotlib and load test data
plt.rc("font", size=14)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("axes", titlesize=16, titleweight="bold")
plt.rc("figure", dpi=200, titlesize=20, titleweight="bold")
plt.rcParams["figure.constrained_layout.use"] = True


# %% Scatter plots laid out as a periodic table
data_dict = {
    elem.symbol: [
        np.random.randint(0, 20, 10),
        np.random.randint(0, 20, 10),
        # np.random.randint(0, 20, 10),  # TODO:
    ]
    for elem in Element
}

fig = ptable_scatters(
    data_dict,
    # colormap="coolwarm",
    # cbar_title="Periodic Table Scatter Plots",
    child_args=dict(marker="o", linestyle=""),
)
save_and_compress_svg(fig, "ptable-scatters")


# %% Line plots laid out as a periodic table
data_dict = {
    elem.symbol: [
        np.linspace(0, 10, 10),
        np.sin(2 * np.pi * np.linspace(0, 10, 10)) + np.random.normal(0, 0.2, 10),
    ]
    for elem in Element
}

fig = ptable_lines(
    data_dict,
)
save_and_compress_svg(fig, "ptable-lines")


# %% Evenly-split tile plots laid out as a periodic table
data_dict = {
    elem.symbol: [
        random.randint(0, 10),
        random.randint(10, 20),
        # random.randint(20, 30),
    ]
    for elem in Element
}

fig = ptable_splits(
    data=data_dict,
    colormap="coolwarm",
    start_angle=135,
    cbar_title="Periodic Table Evenly-Split Tiles Plots",
)
save_and_compress_svg(fig, "ptable-splits")
