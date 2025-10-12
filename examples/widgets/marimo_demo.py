"""marimo demo for pymatviz widgets.

Run with `marimo edit examples/widgets/marimo_demo.py --no-sandbox --watch`.
"""
# /// script
# dependencies = [
#     "pymatgen>=2024.1.1",
#     "ase>=3.22.0",
#     "phonopy>=2.20.0",
# ]
# ///

import marimo


# ruff: noqa: B018, ANN001, N803, ANN202
# mypy: ignore-errors


__generated_with = "0.14.10"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from typing import Final

    import marimo as mo
    from ase.build import bulk, molecule
    from ipywidgets import GridBox, Layout
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Composition, Lattice, Structure

    import pymatviz as pmv

    # Test Structure Widget
    struct = Structure(
        lattice=Lattice.cubic(3),
        species=("Fe", "Fe"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )

    structure_widget = pmv.StructureWidget(structure=struct)
    structure_widget
    return (
        Composition,
        Final,
        GridBox,
        Lattice,
        Layout,
        PhonopyAtoms,
        Structure,
        bulk,
        mo,
        molecule,
        os,
        pmv,
        struct,
    )


@app.cell
def _(struct):
    # Test pymatgen Structure MIME type recognition (should render as StructureWidget)
    struct


@app.cell
def _(Lattice, Structure, pmv):
    # Test Trajectory Widget with simple trajectory of expanding lattice

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
    trajectory_widget
    return (trajectory_widget,)


@app.cell
def _(bulk):
    # Test ASE Atoms MIME type display

    ase_atoms = bulk("Al", "fcc", a=4.05)
    ase_atoms *= (2, 2, 2)  # Create a 2x2x2 supercell
    ase_atoms
    return (ase_atoms,)


@app.cell
def _(molecule):
    # Test ASE molecule MIME type display

    ase_molecule = molecule("H2O")
    ase_molecule.center(vacuum=3.0)
    ase_molecule
    return (ase_molecule,)


@app.cell
def _(PhonopyAtoms):
    # Test phonopy atoms MIME type display

    lattice = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
    positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    symbols = ["Na", "Cl"]

    phonopy_atoms = PhonopyAtoms(symbols=symbols, positions=positions, cell=lattice)
    phonopy_atoms


@app.cell
def _(Final, os, pmv):
    matterviz_traj_dir_url: Final = (
        "https://github.com/janosh/matterviz/raw/6288721042/src/site/trajectories"
    )
    _file_name = "flame-gold-cluster-55-atoms.h5"
    if not os.path.isfile(f"tmp/{_file_name}"):
        import urllib.request

        os.makedirs("tmp", exist_ok=True)
        urllib.request.urlretrieve(  # noqa: S310
            f"{matterviz_traj_dir_url}/{_file_name}", f"tmp/{_file_name}"
        )
    _gold_cluster_traj = pmv.TrajectoryWidget(
        data_url=f"tmp/{_file_name}",
        display_mode="structure+scatter",
        show_force_vectors=False,
    )
    _gold_cluster_traj
    return (matterviz_traj_dir_url,)


@app.cell
def _(matterviz_traj_dir_url, pmv):
    _file_name = "Cr0.25Fe0.25Co0.25Ni0.25-mace-omat-qha.xyz.gz"
    ase_traj_widget = pmv.TrajectoryWidget(
        data_url=f"{matterviz_traj_dir_url}/{_file_name}",
        display_mode="structure+scatter",
        show_force_vectors=True,
        force_vector_scale=0.5,
        force_vector_color="#ff4444",
        show_bonds=True,
        bonding_strategy="nearest_neighbor",
        style="height: 600px;",
    )
    ase_traj_widget


@app.cell
def _(matterviz_traj_dir_url, pmv):
    _gold_cluster_traj = pmv.TrajectoryWidget(
        data_url=f"{matterviz_traj_dir_url}/flame-gold-cluster-55-atoms.h5",
        display_mode="structure+scatter",
        show_force_vectors=False,
        style="height: 600px;",
    )
    _gold_cluster_traj


@app.cell
def _(Lattice, Structure, mo):
    """Dynamic trajectory growing in real time."""
    from time import sleep

    # Create trajectory with expanding lattice and properties
    dynamic_trajectory = []
    _step_idx = 0
    for _step_idx in range(_n_steps := 10):
        sleep(0.5)
        _scale = 3.0 + _step_idx * 0.1
        _struct = Structure(
            lattice=Lattice.cubic(_scale),
            species=("Fe", "Fe"),
            coords=((0, 0, 0), (0.5, 0.5, 0.5)),
        )

        # Add properties to demonstrate the new functionality
        _trajectory_step = {
            "structure": _struct,
            "energy": -1.23 - _step_idx**0.5 * 0.01,
            "step": _step_idx,
            "lattice_parameter": _scale,
        }
        dynamic_trajectory.append(_trajectory_step)

    # Display status
    mo.md(f"**Auto-growing trajectory demo** - Current steps: {_step_idx}/{_n_steps}")

    return dynamic_trajectory


@app.cell
def _(dynamic_trajectory, pmv):
    """Create the auto-growing trajectory widget for marimo demo."""
    # Create initial trajectory with one step

    # Create trajectory widget
    dynamic_trajectory_widget = pmv.TrajectoryWidget(
        trajectory=dynamic_trajectory,
        display_mode="structure+scatter",
        show_controls=True,
        style="height: 600px;",
    )
    dynamic_trajectory_widget
    return dynamic_trajectory_widget


@app.cell
def _(Composition, mo, pmv):
    # Test Composition Widget

    comps = (
        "Fe2 O3",
        Composition("Li P O4"),
        dict(Co=20, Cr=20, Fe=20, Mn=20, Ni=20),
        dict(Ti=20, Zr=20, Nb=20, Mo=20, V=20),
    )
    modes = ("pie", "bar", "bubble")
    size = 100
    # Build a grid.
    h_stacks = [
        mo.hstack(
            [
                pmv.CompositionWidget(
                    composition=comp,
                    mode=mode,
                    style=f"width: {(1 + (mode == 'bar')) * size}px; height: {size}px;",
                )
                for mode in modes
            ]
        )
        for comp in comps
    ]
    mo.vstack(h_stacks, align="center", gap=2)


if __name__ == "__main__":
    app.run()
