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


# Test Structure Widget
struct = Structure(
    lattice=Lattice.cubic(3), species=("Fe", "Fe"), coords=((0, 0, 0), (0.5, 0.5, 0.5))
)

structure_widget = pmv.StructureWidget(structure=struct)
display(structure_widget)


# %% Test pymatgen Structure MIME type recognition (should render as StructureWidget)
display(struct)


# %% Test Trajectory Widget with simple trajectory of expanding lattice
trajectory = []
for idx in range(5):
    scale = 3.0 + idx * 0.1
    struct_frame = Structure(
        lattice=Lattice.cubic(scale),
        species=("Fe", "Fe"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )
    trajectory.append(struct_frame)

trajectory_widget = pmv.TrajectoryWidget(trajectory=trajectory)
display(trajectory_widget)


# %% Test Trajectory Widget with simple trajectory of expanding lattice
trajectory = []
base_struct = Structure(
    lattice=Lattice.cubic(3.0),
    species=("Fe", "Fe"),
    coords=((0, 0, 0), (0.5, 0.5, 0.5)),
)

for idx in range(n_steps := 20):
    struct_frame = base_struct.perturb(distance=0.2).copy()
    # calc dist between atoms
    energy = n_steps / 2 - idx * np_rng.random()
    np.fill_diagonal(dist_max := struct_frame.distance_matrix, np.inf)
    min_dist = dist_max.min()
    f_max = 1 / min_dist
    step_data = {"structure": struct_frame, "energy": energy, "force_max": f_max}
    trajectory.append(step_data)

trajectory_widget = pmv.TrajectoryWidget(trajectory=trajectory)
display(trajectory_widget)


# %% Test ASE Atoms MIME type display
ase_atoms = bulk("Al", "fcc", a=4.05)
ase_atoms *= (2, 2, 2)  # Create a 2x2x2 supercell
display(ase_atoms)


# %% Test ASE molecule MIME type display
ase_molecule = molecule("H2O")
ase_molecule.center(vacuum=3.0)
display(ase_molecule)


# %% Test phonopy atoms MIME type display
lattice = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
symbols = ["Na", "Cl"]

phonopy_atoms = PhonopyAtoms(symbols=symbols, positions=positions, cell=lattice)
display(phonopy_atoms)


# %% Render local flame HDF5 trajectory file
matterviz_traj_dir_url: Final = (
    "https://github.com/janosh/matterviz/raw/33aa595dc/src/site/trajectories"
)
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


# %% Render remote ASE trajectory file
githack_traj_dir_url: Final = (
    "https://raw.githack.com/janosh/matterviz/33aa595dc/src/site/trajectories"
)
file_name = "Cr0.25Fe0.25Co0.25Ni0.25-mace-omat-qha.xyz.gz"
ase_traj_widget = pmv.TrajectoryWidget(
    data_url=f"{githack_traj_dir_url}/{file_name}",
    display_mode="structure+scatter",
    show_force_vectors=True,
    force_vector_scale=0.5,
    force_vector_color="#ff4444",
    show_bonds=True,
    bonding_strategy="nearest_neighbor",
    style="height: 600px;",
)
display(ase_traj_widget)


# %% Render remote flame HDF5 trajectory file
gold_cluster_traj = pmv.TrajectoryWidget(
    data_url=f"{githack_traj_dir_url}/flame-gold-cluster-55-atoms.h5",
    display_mode="structure+scatter",
    show_force_vectors=False,
    style="height: 600px;",
)
display(gold_cluster_traj)


# %% Test Composition Widget
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
