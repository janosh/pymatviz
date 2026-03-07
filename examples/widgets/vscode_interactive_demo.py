"""VSCode interactive demo for pymatviz widgets."""
# /// script
# dependencies = [
#     "pymatgen>=2024.1.1",
#     "ase>=3.22.0",
#     "phonopy>=2.20.0",
# ]
# ///

__date__ = "2026-03-04"


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


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TEST_FILES = f"{REPO_ROOT}/tests/files"


np_rng = np.random.default_rng(seed=0)

# === Convex Hull from PhaseDiagram ===


# %% [markdown]
# ### Convex Hull Widget
# Build a small Li-Fe-O phase diagram and visualize stable/unstable entries on the hull.


# %% Convex Hull — compute stability from PhaseDiagram entries
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram  # noqa: E402


# Stable phases on the convex hull
stable_phases = {
    "Li": -1.9,
    "Fe": -4.2,
    "O": -3.0,
    "Li2O": -15.8,
    "FeO": -13.0,
    "Fe2O3": -33.0,
    "Fe3O4": -46.0,
    "LiFeO2": -25.0,
    "Li5FeO4": -58.0,
    "LiFe5O8": -92.0,
    "Li2Fe2O4": -52.0,
    "LiFeO3": -31.0,
    "Li2FeO3": -37.0,
    "LiFe2O4": -46.0,
    "Li3FeO3": -42.0,
    "Fe2O5": -40.0,
}
entries = [PDEntry(Composition(c), e) for c, e in stable_phases.items()]
_hull = PhaseDiagram(entries)

# Sweep Li_a Fe_b O_c compositions and place each above the hull with
# log-normal delta (median ~0.04 eV/atom, clamped to <=0.25 eV/atom for realism).
# Deterministically produces 265 unstable entries (281 total with stable_phases).
_seen = set(stable_phases)
for li_count, fe_count, o_count in itertools.product(range(7), range(7), range(1, 8)):
    if li_count == 0 and fe_count == 0:
        continue
    comp = Composition({"Li": li_count, "Fe": fe_count, "O": o_count})
    if comp.reduced_formula in _seen:
        continue
    _seen.add(comp.reduced_formula)
    per_atom_delta = min(np_rng.lognormal(mean=-3.2, sigma=1.0), 0.25)
    entries.append(
        PDEntry(comp, _hull.get_hull_energy(comp) + per_atom_delta * comp.num_atoms)
    )
phase_diag = PhaseDiagram(entries)

convex_hull_widget = pmv.ConvexHullWidget(
    entries=phase_diag,
    style="height: 500px;",
)
convex_hull_widget.show()


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
structure_widget.show()


# %% [markdown]
# ### Multi-Vector Comparison
# Compare per-atom forces from two methods (e.g. DFT vs MLFF) on the same structure.
# Each vector set gets a distinct color and can be toggled independently.


# %% Multi-Vector — DFT vs MLFF force comparison on GaN
dft_forces = [
    [0.15, -0.08, 0.03],
    [-0.12, 0.18, -0.06],
    [0.03, 0.06, -0.22],
    [-0.09, -0.03, 0.15],
]
mlff_forces = [
    [0.13, -0.07, 0.04],
    [-0.11, 0.16, -0.05],
    [0.02, 0.07, -0.20],
    [-0.08, -0.04, 0.14],
]
struct_multi_vec = struct.copy(
    site_properties={"force_DFT": dft_forces, "force_MLFF": mlff_forces}
)

multi_vec_widget = pmv.StructureWidget(
    structure=struct_multi_vec,
    show_bonds=True,
    vector_configs={
        "force_DFT": {"color": "#e74c3c"},
        "force_MLFF": {"color": "#3498db"},
    },
    style="height: 400px;",
)
multi_vec_widget.show()


# %% [markdown]
# ### Brillouin Zone Widget
# Show the reciprocal-space Brillouin zone corresponding to the GaN structure above.


# %% Brillouin Zone — hexagonal BZ from GaN
bz_widget = pmv.BrillouinZoneWidget(
    structure=struct, show_vectors=True, style="height: 400px;"
)
bz_widget.show()


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
xrd_widget.show()


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
    style="height: 600px;",
)
trajectory_widget.show()


# === Plot Widgets ===


# %% [markdown]
# ### Scatter Plot Widget
# Dual-axis comparison of `sin(x)` and `cos(x)` on shared x-values.


# %% ScatterPlot Widget — dual-axis trigonometric curves
scatter_series = [
    dict(label="sin(x)", x=np.linspace(0, 6.0, 60), y=np.sin(np.linspace(0, 6.0, 60))),
    dict(
        label="cos(x)",
        x=np.linspace(0, 6.0, 60),
        y=np.cos(np.linspace(0, 6.0, 60)),
        y_axis="y2",
    ),
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
scatter_plot_widget.show()


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
bar_plot_widget.show()


# %% [markdown]
# ### Histogram Widget
# Overlaid histograms summarize the value distributions of the two scatter series.


# %% Histogram Widget — distribution overlay for scatter data
histogram_series = [
    {key: s[key] for key in ("label", "x", "y")} for s in scatter_series
]
histogram_widget = pmv.HistogramWidget(
    series=histogram_series,
    bins=20,
    mode="overlay",
    x_axis={"label": "Value"},
    y_axis={"label": "Count"},
    style="height: 360px;",
)
histogram_widget.show()


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
bands_widget.show()


# %% [markdown]
# ### DOS Widget
# Use matching phonon DOS data from the same fixture document.


# %% DOS Widget — dict-based demo
dos_data = phonon_doc.phonon_dos

dos_widget = pmv.DosWidget(dos=dos_data, style="height: 400px;")
dos_widget.show()


# %% [markdown]
# ### Bands + DOS Widget
# Combine bands and DOS into a coordinated side-by-side view.


# %% Combined Bands + DOS
bands_dos_widget = pmv.BandsAndDosWidget(
    band_structure=band_data, dos=dos_data, style="height: 500px;"
)
bands_dos_widget.show()


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
composition_pie_widget = pmv.CompositionWidget(
    composition=comps[0],
    mode="pie",
    style=f"width: {size}px; height: {size}px;",
)
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
    vector_scale=0.5,
    vector_color="#ff4444",
    show_bonds=True,
    bonding_strategy="nearest_neighbor",
    style="height: 600px;",
)
ase_traj_widget.show()


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
)
traj_widget.show()


# === New Widget Types ===


# %% [markdown]
# ### Isosurface Rendering (CHGCAR/CUBE)
# Load a VASP CHGCAR file and render charge density isosurfaces with custom settings.

matterviz_iso_dir_url: Final = (
    "https://github.com/janosh/matterviz/raw/550d96d2/src/site/isosurfaces"
)


# %% Isosurface — Si charge density from CHGCAR
iso_widget = pmv.StructureWidget(
    data_url=f"{matterviz_iso_dir_url}/Si-CHGCAR.gz",
    isosurface_settings={
        "isovalue": 0.05,
        "opacity": 0.6,
        "positive_color": "#3b82f6",
        "show_negative": False,
    },
    style="height: 500px;",
)
iso_widget.show()


# %% [markdown]
# ### Molecular Orbital Isosurface (.cube)
# Render caffeine HOMO orbital lobes from a Gaussian .cube file.


# %% Isosurface — caffeine HOMO
orbital_widget = pmv.StructureWidget(
    data_url=f"{matterviz_iso_dir_url}/caffeine-HOMO.cube.gz",
    isosurface_settings={
        "isovalue": 0.02,
        "opacity": 0.7,
        "positive_color": "#3b82f6",
        "negative_color": "#ef4444",
        "show_negative": True,
    },
    show_bonds=True,
    style="height: 500px;",
)
orbital_widget.show()


# %% [markdown]
# ### Periodic Table Widget
# Element heatmap showing atomic masses with interactive hover.


# %% Periodic Table — atomic mass heatmap
mass_data = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.01,
    "N": 14.01,
    "O": 16.00,
    "F": 19.00,
    "Ne": 20.18,
    "Na": 22.99,
    "Mg": 24.31,
    "Al": 26.98,
    "Si": 28.09,
    "P": 30.97,
    "S": 32.07,
    "Cl": 35.45,
    "Ar": 39.95,
    "K": 39.10,
    "Ca": 40.08,
    "Fe": 55.85,
    "Co": 58.93,
    "Ni": 58.69,
    "Cu": 63.55,
    "Zn": 65.38,
}
ptable_widget = pmv.PeriodicTableWidget(
    heatmap_values=mass_data,
    color_scale="interpolateViridis",
    style="height: 400px;",
)
ptable_widget.show()


# %% [markdown]
# ### 3D Scatter Plot Widget
# Random 3D point cloud colored by z-value.


# %% ScatterPlot3D — random 3D data
scatter_3d_widget = pmv.ScatterPlot3DWidget(
    series=[
        {
            "x": np_rng.normal(0, 1, 50).tolist(),
            "y": np_rng.normal(0, 1, 50).tolist(),
            "z": np_rng.normal(0, 1, 50).tolist(),
            "label": "Random points",
        }
    ],
    x_axis={"label": "x"},
    y_axis={"label": "y"},
    z_axis={"label": "z"},
    style="height: 500px;",
)
scatter_3d_widget.show()


# %% [markdown]
# ### Heatmap Matrix Widget
# Element pair interaction strengths as a labeled grid.


# %% HeatmapMatrix — element pair interactions
elements = ["Fe", "O", "Li", "Mn"]
heatmap_widget = pmv.HeatmapMatrixWidget(
    x_items=elements,
    y_items=elements,
    values=[
        [1.0, 0.8, 0.3, 0.6],
        [0.8, 1.0, 0.2, 0.5],
        [0.3, 0.2, 1.0, 0.4],
        [0.6, 0.5, 0.4, 1.0],
    ],
    color_scale="interpolateBlues",
    tile_size="80px",
    style="height: 400px;",
)
heatmap_widget.show()


# %% [markdown]
# ### Space Group Bar Plot Widget
# Distribution of space groups in a dataset.


# %% SpacegroupBarPlot — space group frequencies
spacegroup_widget = pmv.SpacegroupBarPlotWidget(
    data=[225, 225, 225, 166, 166, 62, 62, 62, 62, 139, 139, 12, 14, 14, 14, 14],
    style="height: 350px;",
)
spacegroup_widget.show()


# %% [markdown]
# ### Chemical Potential Diagram Widget
# Stability regions in Li-Fe-O chemical potential space.


# %% ChemPotDiagram — Li-Fe-O system
chem_pot_widget = pmv.ChemPotDiagramWidget(
    entries=[
        {"name": "Li", "energy": -1.9, "composition": {"Li": 1}},
        {"name": "Fe", "energy": -8.3, "composition": {"Fe": 1}},
        {"name": "O2", "energy": -4.9, "composition": {"O": 1}},
        {"name": "Li2O", "energy": -14.3, "composition": {"Li": 2, "O": 1}},
        {"name": "Fe2O3", "energy": -25.0, "composition": {"Fe": 2, "O": 3}},
        {"name": "LiFeO2", "energy": -17.5, "composition": {"Li": 1, "Fe": 1, "O": 2}},
    ],
    style="height: 500px;",
)
chem_pot_widget.show()


# %% [markdown]
# ### RDF Plot Widget
# Radial distribution function computed on-the-fly from the GaN structure.


# %% RdfPlot — GaN pair distribution
rdf_widget = pmv.RdfPlotWidget(
    structures=struct.as_dict(),
    cutoff=10,
    n_bins=80,
    style="height: 400px;",
)
rdf_widget.show()


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


# === Export all widgets to image files ===


# %% [markdown]
# ### Export All Widgets
# Save every widget above to PNG, SVG, and PDF.
# Canvas-based (WebGL) widgets only support PNG/PDF; SVG export is skipped for those.


# %% Export widgets to all supported formats
from pymatviz.widgets.matterviz import MatterVizWidget  # noqa: E402, I001

EXPORT_DIR = f"{os.path.dirname(__file__)}/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Canvas-based (WebGL) widgets — PNG and PDF only (SVG not supported)
canvas_widgets: dict[str, MatterVizWidget] = {
    "structure": structure_widget,
    "brillouin_zone": bz_widget,
    "trajectory": trajectory_widget,
    "ase_trajectory": ase_traj_widget,
    "trajectory_md": traj_widget,
    "isosurface": iso_widget,
    "orbital": orbital_widget,
    "scatter_plot_3d": scatter_3d_widget,
}

# SVG-based widgets — PNG, SVG, and PDF
svg_widgets: dict[str, MatterVizWidget] = {
    "convex_hull": convex_hull_widget,
    "xrd": xrd_widget,
    "scatter_plot": scatter_plot_widget,
    "bar_plot": bar_plot_widget,
    "histogram": histogram_widget,
    "band_structure": bands_widget,
    "dos": dos_widget,
    "bands_and_dos": bands_dos_widget,
    "periodic_table": ptable_widget,
    "heatmap_matrix": heatmap_widget,
    "spacegroup_bar": spacegroup_widget,
    "chem_pot_diagram": chem_pot_widget,
    "rdf_plot": rdf_widget,
    "composition_pie": composition_pie_widget,
}

for widgets, fmts in [
    (canvas_widgets, ("png", "pdf")),
    (svg_widgets, ("png", "svg", "pdf")),
]:
    for name, widget in widgets.items():
        for fmt in fmts:
            path = f"{EXPORT_DIR}/{name}.{fmt}"
            try:
                widget.to_img(filename=path, fmt=fmt)  # type: ignore[arg-type]
                print(f"  saved {path}")
            except (
                TimeoutError,
                RuntimeError,
                ImportError,
                OSError,
                ValueError,
            ) as exc:
                print(f"  SKIP {path}: {exc}")

print(f"\nExport complete. Files in {EXPORT_DIR}/")
