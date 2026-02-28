"""VSCode interactive demo for pymatviz widgets."""
# /// script
# dependencies = [
#     "pymatgen>=2024.1.1",
#     "ase>=3.22.0",
#     "phonopy>=2.20.0",
# ]
# ///

__date__ = "2025-07-16"


# %%
import itertools
import json
import os
from typing import Final

import numpy as np
from ase.build import bulk, molecule
from ipywidgets import GridBox, Layout
from monty.io import zopen
from monty.json import MontyDecoder
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Composition, Lattice, Structure

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES
from pymatviz.widgets.matterviz import MatterVizWidget


np_rng = np.random.default_rng(seed=0)

# === Convex Hull from PhaseDiagram ===


# %% [markdown]
# ### Convex Hull Widget
# Build a small Li-Fe-O phase diagram and visualize stable/unstable entries on the hull.


# %% Convex Hull — compute stability from PhaseDiagram entries
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram  # noqa: E402


phase_diag = PhaseDiagram(
    [
        PDEntry(Composition("Li"), -1.9),
        PDEntry(Composition("Fe"), -4.2),
        PDEntry(Composition("O"), -3.0),
        PDEntry(Composition("Li2O"), -14.3),
        PDEntry(Composition("Fe2O3"), -25.5),
        PDEntry(Composition("LiFeO2"), -18.0),
        PDEntry(Composition("FeO"), -8.5),
    ]
)

convex_hull_widget = pmv.ConvexHullWidget(
    entries=phase_diag,
    style="height: 500px;",
)
convex_hull_widget.display()


# === 3D Structure + Brillouin Zone ===


# %% [markdown]
# ### Structure Widget
# Render wurtzite GaN as an interactive crystal structure with bonds.


# %% Structure Widget — wurtzite GaN (hexagonal, more interesting than cubic)
struct = Structure(
    lattice=Lattice.hexagonal(3.19, 5.19),
    species=["Ga", "Ga", "N", "N"],
    coords=[
        [1 / 3, 2 / 3, 0],
        [2 / 3, 1 / 3, 0.5],
        [1 / 3, 2 / 3, 0.375],
        [2 / 3, 1 / 3, 0.875],
    ],
)

structure_widget = pmv.StructureWidget(
    structure=struct, show_bonds=True, style="height: 400px;"
)
structure_widget.display()


# %% [markdown]
# ### Brillouin Zone Widget
# Show the reciprocal-space Brillouin zone corresponding to the GaN structure above.


# %% Brillouin Zone — hexagonal BZ from GaN
bz_widget = pmv.BrillouinZoneWidget(
    structure=struct, show_vectors=True, style="height: 400px;"
)
bz_widget.display()


# === XRD Pattern ===


# %% [markdown]
# ### XRD Widget
# Compute an XRD pattern for rutile TiO2 and render peak intensities versus 2-theta.


# %% XRD Pattern — rutile TiO2 (tetragonal, richer peak pattern than cubic Si)
from pymatgen.analysis.diffraction.xrd import XRDCalculator  # noqa: E402


tio2_struct = Structure(
    Lattice.tetragonal(4.594, 2.959),
    ["Ti", "Ti", "O", "O", "O", "O"],
    [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
        [0.305, 0.305, 0],
        [0.695, 0.695, 0],
        [0.195, 0.805, 0.5],
        [0.805, 0.195, 0.5],
    ],
)
xrd_pattern = XRDCalculator().get_pattern(tio2_struct)

xrd_widget = pmv.XrdWidget(patterns=xrd_pattern, style="height: 350px;")
xrd_widget.display()


# === Trajectory with Force Vectors ===


# %% [markdown]
# ### Trajectory Widget
# Simulate a short perturbed Fe trajectory and display structure and scalar properties.


# %% Trajectory Widget — expanding lattice with energy/force properties
trajectory = []
base_struct = Structure(
    lattice=Lattice.cubic(3.0),
    species=("Fe", "Fe"),
    coords=((0, 0, 0), (0.5, 0.5, 0.5)),
)

for idx in range(n_steps := 20):
    struct_frame = base_struct.perturb(distance=0.2).copy()
    energy = n_steps / 2 - idx * np_rng.random()
    np.fill_diagonal(dist_max := struct_frame.distance_matrix, np.inf)
    min_dist = dist_max.min()
    f_max = 1 / min_dist
    step_data = {"structure": struct_frame, "energy": energy, "force_max": f_max}
    trajectory.append(step_data)

trajectory_widget = pmv.TrajectoryWidget(
    trajectory=trajectory,
    display_mode="structure+scatter",
    show_force_vectors=True,
    style="height: 600px;",
)
trajectory_widget.display()


# === Plot Widgets ===


# %% [markdown]
# ### Scatter Plot Widget
# Dual-axis comparison of `sin(x)` and `cos(x)` on shared x-values.


# %% ScatterPlot Widget — dual-axis trigonometric curves
scatter_series = [
    {
        "label": "sin(x)",
        "x": np.linspace(0, 6.0, 60).tolist(),
        "y": np.sin(np.linspace(0, 6.0, 60)).tolist(),
    },
    {
        "label": "cos(x)",
        "x": np.linspace(0, 6.0, 60).tolist(),
        "y": np.cos(np.linspace(0, 6.0, 60)).tolist(),
        "y_axis": "y2",
    },
]
scatter_plot_widget = pmv.ScatterPlotWidget(
    series=scatter_series,
    x_axis={"label": "x"},
    y_axis={"label": "sin(x)"},
    y2_axis={"label": "cos(x)", "color": "#ff7f0e"},
    display={"x_grid": True, "y_grid": True},
    legend={"position": "top-right"},
    style="height: 420px;",
)
MatterVizWidget.display(scatter_plot_widget)


# %% [markdown]
# ### Bar Plot Widget
# Grouped bars compare two model score series over the same sample index axis.


# %% BarPlot Widget — grouped comparison bars
bar_series = [
    {"label": "Model A", "x": [0, 1, 2], "y": [4.2, 5.1, 4.8]},
    {"label": "Model B", "x": [0, 1, 2], "y": [3.9, 4.6, 5.2]},
]
bar_plot_widget = pmv.BarPlotWidget(
    series=bar_series,
    mode="grouped",
    x_axis={"label": "Sample index"},
    y_axis={"label": "Score"},
    display={"y_grid": True},
    style="height: 360px;",
)
MatterVizWidget.display(bar_plot_widget)


# %% [markdown]
# ### Histogram Widget
# Overlaid histograms summarize the value distributions of the two scatter series.


# %% Histogram Widget — distribution overlay for scatter data
histogram_series = [
    {
        "label": scatter_series[0]["label"],
        "x": scatter_series[0]["x"],
        "y": scatter_series[0]["y"],
    },
    {
        "label": scatter_series[1]["label"],
        "x": scatter_series[1]["x"],
        "y": scatter_series[1]["y"],
    },
]
histogram_widget = pmv.HistogramWidget(
    series=histogram_series,
    bins=20,
    mode="overlay",
    x_axis={"label": "Value"},
    y_axis={"label": "Count"},
    style="height: 360px;",
)
MatterVizWidget.display(histogram_widget)


# === Band Structure + DOS ===


# %% [markdown]
# ### Band Structure Widget
# Load realistic phonon band structure data from test fixtures.


# %% Band Structure Widget — dict-based demo
phonon_fixture_path = f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-pbe.json.xz"
with zopen(phonon_fixture_path, mode="rt") as file:
    phonon_doc = json.loads(file.read(), cls=MontyDecoder)

band_data = phonon_doc.phonon_bandstructure

bands_widget = pmv.BandStructureWidget(band_structure=band_data, style="height: 400px;")
bands_widget.display()


# %% [markdown]
# ### DOS Widget
# Use matching phonon DOS data from the same fixture document.


# %% DOS Widget — dict-based demo
dos_data = phonon_doc.phonon_dos

dos_widget = pmv.DosWidget(dos=dos_data, style="height: 400px;")
dos_widget.display()


# %% [markdown]
# ### Bands + DOS Widget
# Combine bands and DOS into a coordinated side-by-side view.


# %% Combined Bands + DOS
bands_dos_widget = pmv.BandsAndDosWidget(
    band_structure=band_data, dos=dos_data, style="height: 500px;"
)
bands_dos_widget.display()


# === Composition Grid ===


# %% [markdown]
# ### Composition Widgets Grid
# Compare multiple compositions across pie, bar, and bubble display modes.


# %% Composition Widget -- grid of compositions x modes
comps = (
    "Fe2 O3",
    Composition("Li P O4"),
    dict(Co=20, Cr=20, Fe=20, Mn=20, Ni=20),
    dict(Ti=20, Zr=20, Nb=20, Mo=20, V=20),
)
modes = ("pie", "bar", "bubble")
size = 100
children = [
    pmv.CompositionWidget(
        composition=comp,
        mode=mode,
        style=f"width: {(1 + (mode == 'bar')) * size}px; height: {size}px;",
    )
    for comp, mode in itertools.product(comps, modes)
]
layout = Layout(
    grid_template_columns=f"repeat({len(modes)}, auto)",
    grid_gap="2em 4em",
    padding="2em",
)
GridBox(children=children, layout=layout)


# === Remote Trajectory Files ===


# %% [markdown]
# ### Remote Trajectory Widget
# Load and visualize an online XYZ trajectory with force vectors and bond rendering.


# %% Render remote ASE trajectory file
matterviz_traj_dir_url: Final = (
    "https://github.com/janosh/matterviz/raw/6288721042/src/site/trajectories"
)

file_name = "Cr0.25Fe0.25Co0.25Ni0.25-mace-omat-qha.xyz.gz"
ase_traj_widget = pmv.TrajectoryWidget(
    data_url=f"{matterviz_traj_dir_url}/{file_name}",
    display_mode="structure+scatter",
    show_force_vectors=True,
    force_vector_scale=0.5,
    force_vector_color="#ff4444",
    show_bonds=True,
    bonding_strategy="nearest_neighbor",
    style="height: 600px;",
)
ase_traj_widget.display()


# %% [markdown]
# ### Local HDF5 Trajectory Widget
# Download once, cache locally, and render an HDF5 trajectory file from disk.


# %% Render local flame HDF5 trajectory file
file_name = "flame-gold-cluster-55-atoms.h5"
tmp_dir = f"{os.path.dirname(__file__)}/tmp"
local_path = f"{tmp_dir}/{file_name}"
if not os.path.isfile(local_path):
    import urllib.request

    os.makedirs(tmp_dir, exist_ok=True)
    urllib.request.urlretrieve(  # noqa: S310
        f"{matterviz_traj_dir_url}/{file_name}", local_path
    )

traj_widget = pmv.TrajectoryWidget(
    data_url=local_path,
    display_mode="structure+scatter",
    show_force_vectors=False,
)
traj_widget.display()


# === MIME Type Auto-display ===


# %% [markdown]
# ### ASE Atoms MIME Rendering
# Demonstrate automatic widget rendering for an ASE bulk structure object.


# %% ASE Atoms — auto-rendered via MIME type recognition
ase_atoms = bulk("Al", "fcc", a=4.05)
ase_atoms *= (2, 2, 2)
ase_atoms  # noqa: B018


# %% [markdown]
# ### ASE Molecule MIME Rendering
# Demonstrate automatic widget rendering for a simple ASE molecular object.


# %% ASE molecule
ase_molecule = molecule("H2O")
ase_molecule.center(vacuum=3.0)
ase_molecule  # noqa: B018


# %% [markdown]
# ### PhonopyAtoms MIME Rendering
# Demonstrate automatic widget rendering for a PhonopyAtoms crystal object.


# %% PhonopyAtoms — auto-rendered via MIME type recognition
phonopy_atoms = PhonopyAtoms(
    symbols=["Na", "Cl"],
    positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
)
phonopy_atoms  # noqa: B018
