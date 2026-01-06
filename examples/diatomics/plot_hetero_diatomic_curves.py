"""Plot MLIP pair repulsion curves in a periodic table layout and as 3D lines with
elements stacked in the z-direction.
"""

# %%
import json
import lzma
import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.core import Element

import pymatviz as pmv


pmv.set_plotly_template("pymatviz_dark")
module_dir = os.path.dirname(__file__)
__date__ = "2024-03-31"

model_name, z1 = "mace-mp-0-small", 5
elem1 = Element.from_Z(z1)
lzma_path = f"{module_dir}/hetero-nuclear-diatomics-{z1}-{model_name}.json.xz"
with lzma.open(lzma_path, mode="rt") as file:
    hetero_nuc_diatomics = json.load(file)

x_range, y_range = [0, 6], [-8, 15]


# %% plot homo-nuclear and heteronuclear pair repulsion curves
# Convert data to format needed for plotting
# Each element in diatomic_curves should be a tuple of (x_values, y_values)
diatomic_curves: dict[str, tuple[list[float], list[float]]] = {}
distances = hetero_nuc_diatomics.pop("distances", locals().get("distances"))

for elem_pair in hetero_nuc_diatomics:
    energies = np.asarray(hetero_nuc_diatomics[elem_pair])
    # Get element symbol from the key (format is "Z-Z" where Z is atomic number)
    elem2 = elem_pair.split("-")[1]

    # Shift energies so the energy at infinite separation (last point) is 0
    shifted_energies = energies - energies[-1]

    diatomic_curves[elem2] = distances, shifted_energies


# %%
fig = pmv.ptable_scatter_plotly(
    diatomic_curves,
    mode="lines",
    x_axis_kwargs=dict(range=x_range),
    y_axis_kwargs=dict(range=y_range),
    scale=1.2,
)

title = f"<b>{model_name.title()}</b> Heteronuclear Diatomic Curves for <b>{elem1.long_name}</b>"  # noqa: E501
fig.layout.title.update(text=title, x=0.4, y=0.8)
fig.show()
pmv.io.save_and_compress_svg(fig, f"hetero-nuclear-{model_name}-{elem1}")


# %%
fig = go.Figure()
# Sort elements by atomic number for consistent z-ordering
sorted_elements = sorted(diatomic_curves, key=lambda symbol: Element(symbol).Z)

# Find global min/max energy for consistent coloring
min_energies: dict[str, float] = {}
filtered_distances: dict[str, np.ndarray] = {}
filtered_energies: dict[str, np.ndarray] = {}

# First pass: collect min energies and filter data
for elem2 in sorted_elements:
    distances, energies = map(np.asarray, diatomic_curves[elem2])
    mask = distances >= 0.5  # type: ignore[unsupported-operator]
    filtered_distances[elem2] = distances[mask]
    filtered_energies[elem2] = energies[mask]
    min_energies[elem2] = min(energies[mask])

min_energy_global = min(min_energies.values())

# Create a trace for each element
for idx, elem2 in enumerate(sorted_elements):
    distances = filtered_distances[elem2]
    energies = filtered_energies[elem2]
    z_pos = Element(elem2).Z  # Use atomic number for z-position

    # Create a constant z array for the line
    z_vals = [z_pos] * len(distances)

    # Normalize the minimum energy for this element to get color
    min_energy = min_energies[elem2]
    # Use log scale for better color distribution
    normalized_energy = np.log(-min_energy + 1) / np.log(-min_energy_global + 1)
    line_color = px.colors.sample_colorscale("Reds", [normalized_energy])[0]

    fig.add_scatter3d(
        x=distances,
        y=energies,
        z=z_vals,
        name=f"{elem1}-{elem2} (min={min_energy:.1f} eV)",
        mode="lines",
        line=dict(width=4, color=line_color),
        showlegend=True,
    )

    # Create 4-fold staggered pattern for element labels
    x_offset = (idx % 4) * 0.3  # 4 positions, spaced by 0.3 Å

    fig.add_scatter3d(
        x=[distances[-1] - x_offset],  # Last x point with staggered offset
        y=[energies[-1] + 0.1],  # Last y point
        z=[z_pos],
        mode="text",
        text=[elem2],
        textfont=dict(size=20, color=line_color),
        showlegend=False,
    )


title = f"<b>{model_name.title()}</b> Heteronuclear Diatomic Curves for <b>{elem1.long_name}</b>"  # noqa: E501
fig.layout.title = dict(text=title, x=0.5, y=0.98)
fig.layout.scene = dict(
    xaxis_title="Distance (Å)",
    yaxis_title="Energy (eV)",
    zaxis_title="Atomic Number (Z)",
    camera=dict(
        eye=dict(x=1.3, y=1.3, z=0),
        up=dict(x=0, y=1, z=0),
    ),
    aspectratio=dict(x=1, y=1, z=3),  # Make plot wider by adjusting aspect ratio
    xaxis=dict(range=x_range),
    yaxis=dict(range=y_range),
)
fig.layout.update(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))

fig.show()
pmv.io.save_and_compress_svg(fig, f"hetero-nuclear-{model_name}-{elem1}-lines-3d")
