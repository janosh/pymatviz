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
import os
from typing import Final

import numpy as np
from ase.build import bulk, molecule
from IPython.display import display
from ipywidgets import GridBox, Layout
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Composition, Lattice, Structure

import pymatviz as pmv


np_rng = np.random.default_rng(seed=0)

# === Convex Hull from PhaseDiagram ===


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
display(convex_hull_widget)


# === 3D Structure + Brillouin Zone ===


# %% Structure Widget — BCC iron
struct = Structure(
    lattice=Lattice.cubic(3),
    species=("Fe", "Fe"),
    coords=((0, 0, 0), (0.5, 0.5, 0.5)),
)

structure_widget = pmv.StructureWidget(
    structure=struct, show_bonds=True, style="height: 400px;"
)
display(structure_widget)


# %% Brillouin Zone — from the same structure
bz_widget = pmv.BrillouinZoneWidget(
    structure=struct, show_vectors=True, style="height: 400px;"
)
display(bz_widget)


# === XRD Pattern ===


# %% XRD Pattern — computed from a silicon structure
si_struct = Structure(
    Lattice.cubic(5.431),
    ["Si", "Si"],
    [[0, 0, 0], [0.25, 0.25, 0.25]],
)
from pymatgen.analysis.diffraction.xrd import XRDCalculator  # noqa: E402


xrd_pattern = XRDCalculator().get_pattern(si_struct)

xrd_widget = pmv.XrdWidget(patterns=xrd_pattern, style="height: 350px;")
display(xrd_widget)


# === Trajectory with Force Vectors ===


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
display(trajectory_widget)


# === Band Structure + DOS ===


# %% Band Structure Widget — dict-based demo
band_data = {
    "@module": "pymatgen.electronic_structure.bandstructure",
    "@class": "BandStructureSymmLine",
    "bands": {
        "1": [[0.5 * np.sin(k / 10) + idx for k in range(50)] for idx in range(4)]
    },
    "efermi": 0.0,
    "kpoints": [[k / 50, 0, 0] for k in range(50)],
    "labels_dict": {"\\Gamma": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0]},
    "lattice_rec": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
}

bands_widget = pmv.BandStructureWidget(band_structure=band_data, style="height: 400px;")
display(bands_widget)


# %% DOS Widget — dict-based demo
dos_data = {
    "@module": "pymatgen.electronic_structure.dos",
    "@class": "Dos",
    "energies": list(np.linspace(-5, 5, 200)),
    "densities": {"1": list(np.exp(-0.5 * np.linspace(-5, 5, 200) ** 2))},
    "efermi": 0.0,
}

dos_widget = pmv.DosWidget(dos=dos_data, style="height: 400px;")
display(dos_widget)


# %% Combined Bands + DOS
bands_dos_widget = pmv.BandsAndDosWidget(
    band_structure=band_data, dos=dos_data, style="height: 500px;"
)
display(bands_dos_widget)


# === Composition Grid ===


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
display(ase_traj_widget)


# %% Render local flame HDF5 trajectory file
file_name = "flame-gold-cluster-55-atoms.h5"
if not os.path.isfile(f"tmp/{file_name}"):
    import urllib.request

    os.makedirs("tmp", exist_ok=True)
    urllib.request.urlretrieve(  # noqa: S310
        f"{matterviz_traj_dir_url}/{file_name}", f"tmp/{file_name}"
    )

traj_widget = pmv.TrajectoryWidget(
    data_url=f"{os.path.dirname(__file__)}/tmp/{file_name}",
    display_mode="structure+scatter",
    show_force_vectors=False,
)
display(traj_widget)


# === MIME Type Auto-display ===


# %% ASE Atoms — auto-rendered via MIME type recognition
ase_atoms = bulk("Al", "fcc", a=4.05)
ase_atoms *= (2, 2, 2)
display(ase_atoms)


# %% ASE molecule
ase_molecule = molecule("H2O")
ase_molecule.center(vacuum=3.0)
display(ase_molecule)


# %% PhonopyAtoms — auto-rendered via MIME type recognition
phonopy_atoms = PhonopyAtoms(
    symbols=["Na", "Cl"],
    positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
)
display(phonopy_atoms)
