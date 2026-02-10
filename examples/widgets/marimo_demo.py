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


__generated_with = "0.14.10"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    from typing import Final

    import marimo as mo
    import numpy as np
    from ase.build import bulk, molecule
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Composition, Lattice, Structure

    import pymatviz as pmv

    np_rng = np.random.default_rng(seed=0)
    return (
        Composition,
        Final,
        Lattice,
        PhonopyAtoms,
        Structure,
        bulk,
        mo,
        molecule,
        np,
        np_rng,
        os,
        pmv,
    )


# === Convex Hull from PhaseDiagram ===


@app.cell
def _(Composition, pmv):
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

    _phase_diag = PhaseDiagram(
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
    pmv.ConvexHullWidget(entries=_phase_diag, style="height: 500px;")


# === 3D Structure + Brillouin Zone ===


@app.cell
def _(Lattice, Structure, pmv):
    _struct = Structure(
        lattice=Lattice.hexagonal(3.19, 5.19),
        species=["Ga", "Ga", "N", "N"],
        coords=[
            [1 / 3, 2 / 3, 0],
            [2 / 3, 1 / 3, 0.5],
            [1 / 3, 2 / 3, 0.375],
            [2 / 3, 1 / 3, 0.875],
        ],
    )
    pmv.StructureWidget(structure=_struct, show_bonds=True, style="height: 400px;")
    return (_struct,)


@app.cell
def _(_struct, pmv):
    pmv.BrillouinZoneWidget(
        structure=_struct, show_vectors=True, style="height: 400px;"
    )


# === XRD Pattern ===


@app.cell
def _(Lattice, Structure, pmv):
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    _tio2_struct = Structure(
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
    pmv.XrdWidget(
        patterns=XRDCalculator().get_pattern(_tio2_struct), style="height: 350px;"
    )


# === Trajectory with Force Vectors ===


@app.cell
def _(Lattice, Structure, np, np_rng, pmv):
    _trajectory = []
    _base_struct = Structure(
        lattice=Lattice.cubic(3.0),
        species=("Fe", "Fe"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )
    for _idx in range(_n_steps := 20):
        _frame = _base_struct.perturb(distance=0.2).copy()
        _energy = _n_steps / 2 - _idx * np_rng.random()
        np.fill_diagonal(_dist := _frame.distance_matrix, np.inf)
        _trajectory.append(
            {"structure": _frame, "energy": _energy, "force_max": 1 / _dist.min()}
        )

    pmv.TrajectoryWidget(
        trajectory=_trajectory,
        display_mode="structure+scatter",
        show_force_vectors=True,
        style="height: 600px;",
    )


# === Band Structure + DOS ===


@app.cell
def _(np, pmv):
    _band_data = {
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
    pmv.BandStructureWidget(band_structure=_band_data, style="height: 400px;")
    return (_band_data,)


@app.cell
def _(np, pmv):
    _dos_data = {
        "@module": "pymatgen.electronic_structure.dos",
        "@class": "Dos",
        "energies": np.linspace(-5, 5, 200).tolist(),
        "densities": {"1": np.exp(-0.5 * np.linspace(-5, 5, 200) ** 2).tolist()},
        "efermi": 0.0,
    }
    pmv.DosWidget(dos=_dos_data, style="height: 400px;")
    return (_dos_data,)


@app.cell
def _(_band_data, _dos_data, pmv):
    pmv.BandsAndDosWidget(
        band_structure=_band_data, dos=_dos_data, style="height: 500px;"
    )


# === Composition Grid ===


@app.cell
def _(Composition, mo, pmv):
    _comps = (
        "Fe2 O3",
        Composition("Li P O4"),
        dict(Co=20, Cr=20, Fe=20, Mn=20, Ni=20),
        dict(Ti=20, Zr=20, Nb=20, Mo=20, V=20),
    )
    _modes = ("pie", "bar", "bubble")
    _size = 100
    _h_stacks = [
        mo.hstack(
            [
                pmv.CompositionWidget(
                    composition=comp,
                    mode=mode,
                    style=f"width: {(1 + (mode == 'bar')) * _size}px;"
                    f" height: {_size}px;",
                )
                for mode in _modes
            ]
        )
        for comp in _comps
    ]
    mo.vstack(_h_stacks, align="center", gap=2)


# === Remote Trajectory Files ===


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

    pmv.TrajectoryWidget(
        data_url=f"tmp/{_file_name}",
        display_mode="structure+scatter",
        show_force_vectors=False,
    )
    return (matterviz_traj_dir_url,)


@app.cell
def _(matterviz_traj_dir_url, pmv):
    _file_name = "Cr0.25Fe0.25Co0.25Ni0.25-mace-omat-qha.xyz.gz"
    pmv.TrajectoryWidget(
        data_url=f"{matterviz_traj_dir_url}/{_file_name}",
        display_mode="structure+scatter",
        show_force_vectors=True,
        force_vector_scale=0.5,
        force_vector_color="#ff4444",
        show_bonds=True,
        bonding_strategy="nearest_neighbor",
        style="height: 600px;",
    )


# === MIME Type Auto-display ===


@app.cell
def _(bulk):
    _ase_atoms = bulk("Al", "fcc", a=4.05)
    _ase_atoms *= (2, 2, 2)
    _ase_atoms


@app.cell
def _(molecule):
    _ase_molecule = molecule("H2O")
    _ase_molecule.center(vacuum=3.0)
    _ase_molecule


@app.cell
def _(PhonopyAtoms):
    PhonopyAtoms(
        symbols=["Na", "Cl"],
        positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
    )


if __name__ == "__main__":
    app.run()
