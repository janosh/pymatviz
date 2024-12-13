"""Plot MLIP pair repulsion curves in a periodic table layout.

Thanks to Tamas Stenczel who first did this type of PES smoothness and physicality
analysis in https://github.com/stenczelt/MACE-MP-work for the MACE-MP paper
https://arxiv.org/abs/2401.00096
"""

# %%
import json
import lzma
import os

import numpy as np
from pymatgen.core import Element

import pymatviz as pmv


pmv.set_plotly_template("pymatviz_dark")
module_dir = os.path.dirname(__file__)
__date__ = "2024-03-31"


# %% plot homo-nuclear and heteronuclear pair repulsion curves
model = "medium"  # "small"
lzma_path = f"{module_dir}/homo-nuclear-mace-{model}.json.xz"
with lzma.open(lzma_path, mode="rt") as file:
    homo_nuc_diatomics = json.load(file)

# Convert data to format needed for plotting
# Each element in diatomic_curves should be a tuple of (x_values, y_values)
diatomic_curves: dict[str, tuple[list[float], list[float]]] = {}
distances = homo_nuc_diatomics.pop("distances", locals().get("distances"))

for symbol in homo_nuc_diatomics:
    energies = np.asarray(homo_nuc_diatomics[symbol])
    # Get element symbol from the key (format is "Z-Z" where Z is atomic number)
    elem_z = int(symbol.split("-")[0])
    elem_symbol = Element.from_Z(elem_z).symbol

    # Shift energies so the energy at infinite separation (last point) is 0
    shifted_energies = energies - energies[-1]

    diatomic_curves[elem_symbol] = distances, shifted_energies


fig = pmv.ptable_scatter_plotly(
    diatomic_curves,
    mode="lines",
    x_axis_kwargs=dict(range=[0, 6]),
    y_axis_kwargs=dict(range=[-8, 15]),
    scale=1.2,
)

fig.layout.title.update(text=f"MACE {model.title()} Diatomic Curves", x=0.4, y=0.8)
fig.show()
pmv.io.save_and_compress_svg(fig, f"homo-nuclear-mace-{model}")


# %% count number of elements with energies below E_TOO_LOW
E_TOO_LOW = -20
for model in ("small", "medium"):
    lzma_path = f"{module_dir}/homo-nuclear-mace-{model}.json.xz"
    with lzma.open(lzma_path, mode="rt") as file:
        homo_nuc_diatomics = json.load(file)

    x_dists = homo_nuc_diatomics.pop("distances")
    min_energies = {
        Element.from_Z(int(key.split("-")[0])).symbol: min(y_vals)
        for key, y_vals in homo_nuc_diatomics.items()
    }
    n_lt_10 = sum(val < E_TOO_LOW for val in min_energies.values())
    print(f"diatomic curves for {model=} that dip below {E_TOO_LOW=} eV: {n_lt_10=}")
