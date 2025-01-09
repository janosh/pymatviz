"""Plot MLIP pair repulsion curves in a periodic table layout."""

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
model_name = ("mace-small", "mace-medium", "mace-mpa-0-medium")[-1]
lzma_path = f"{module_dir}/homo-nuclear-{model_name}.json.xz"
with lzma.open(lzma_path, mode="rt") as file:
    homo_nuc_diatomics = json.load(file)

# Convert data to format needed for plotting
# Each element in diatomic_curves should be a tuple of (x_values, y_values)
diatomic_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
distances = homo_nuc_diatomics.pop("distances", locals().get("distances"))

for symbol in homo_nuc_diatomics:
    energies = np.asarray(homo_nuc_diatomics[symbol])
    # Get element symbol from the key (format is "Z-Z" where Z is atomic number)
    elem_symbol = Element(symbol.split("-")[0]).symbol

    # Shift energies so the energy at infinite separation (last point) is 0
    shifted_energies = energies - energies[-1]

    diatomic_curves[elem_symbol] = {model_name: (distances, shifted_energies)}


fig = pmv.ptable_scatter_plotly(
    diatomic_curves,
    mode="lines",
    x_axis_kwargs=dict(range=[0, 6]),
    y_axis_kwargs=dict(range=[-8, 15]),
    scale=1.5,
)

fig.layout.title.update(text=f"{model_name.title()} Diatomic Curves", x=0.4, y=0.8)
fig.show()
# pmv.io.save_and_compress_svg(fig, f"homo-nuclear-{model}")


# %% count number of elements with energies below E_TOO_LOW
E_TOO_LOW = -20
for model_name in ("mace-small", "mace-medium"):
    lzma_path = f"{module_dir}/homo-nuclear-{model_name}.json.xz"
    with lzma.open(lzma_path, mode="rt") as file:
        homo_nuc_diatomics = json.load(file)

    x_dists = homo_nuc_diatomics.pop("distances")
    min_energies = {
        Element.from_Z(int(key.split("-")[0])).symbol: min(y_vals)
        for key, y_vals in homo_nuc_diatomics.items()
    }
    n_lt_10 = sum(val < E_TOO_LOW for val in min_energies.values())
    print(
        f"diatomic curves for {model_name} that dip below {E_TOO_LOW=} eV: {n_lt_10=}"
    )
