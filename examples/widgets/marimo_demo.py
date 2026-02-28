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
def _(mo):
    mo.md(
        "### Convex Hull Widget\n"
        "Build a compact Li-Fe-O phase diagram and visualize its convex hull."
    )


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
def _(mo):
    mo.md(
        "### Structure Widget\n"
        "Render a wurtzite GaN crystal with bonds in an interactive 3D view."
    )


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
def _(mo):
    mo.md(
        "### Brillouin Zone Widget\n"
        "Show the reciprocal-space Brillouin zone for the structure above."
    )


@app.cell
def _(_struct, pmv):
    pmv.BrillouinZoneWidget(
        structure=_struct, show_vectors=True, style="height: 400px;"
    )


# === XRD Pattern ===


@app.cell
def _(mo):
    mo.md("### XRD Widget\nCompute and display an XRD pattern for rutile TiO2.")


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
def _(mo):
    mo.md(
        "### Trajectory Widget\n"
        "Generate a short perturbed Fe trajectory and plot structure with "
        "scalar metadata."
    )


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


# === Plot Widgets ===


@app.cell
def _(mo):
    mo.md(
        "### Scatter Plot Widget\n"
        "Dual-axis comparison of `sin(x)` and `cos(x)` with shared x-values."
    )


@app.cell
def _(np, pmv):
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
    pmv.ScatterPlotWidget(
        series=scatter_series,
        x_axis={"label": "x"},
        y_axis={"label": "sin(x)"},
        y2_axis={"label": "cos(x)", "color": "#ff7f0e"},
        display={"x_grid": True, "y_grid": True},
        legend={"position": "top-right"},
        style="height: 420px;",
    )
    return (scatter_series,)


@app.cell
def _(mo):
    mo.md(
        "### Bar Plot Widget\n"
        "Grouped bars compare two model scores across the same sample indices."
    )


@app.cell
def _(pmv):
    bar_series = [
        {"label": "Model A", "x": [0, 1, 2], "y": [4.2, 5.1, 4.8]},
        {"label": "Model B", "x": [0, 1, 2], "y": [3.9, 4.6, 5.2]},
    ]
    pmv.BarPlotWidget(
        series=bar_series,
        mode="grouped",
        x_axis={"label": "Sample index"},
        y_axis={"label": "Score"},
        display={"y_grid": True},
        style="height: 360px;",
    )


@app.cell
def _(mo):
    mo.md(
        "### Histogram Widget\n"
        "Overlaid histograms show the distributions of the two trigonometric series."
    )


@app.cell
def _(scatter_series, pmv):
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
    pmv.HistogramWidget(
        series=histogram_series,
        bins=20,
        mode="overlay",
        x_axis={"label": "Value"},
        y_axis={"label": "Count"},
        style="height: 360px;",
    )


# === Band Structure + DOS ===


@app.cell
def _(mo):
    mo.md(
        "### Band Structure Widget\n"
        "Load realistic phonon band structure data from test fixtures."
    )


@app.cell
def _(pmv):
    import json

    from monty.io import zopen
    from monty.json import MontyDecoder

    from pymatviz.utils.testing import TEST_FILES

    _phonon_fixture_path = f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-pbe.json.xz"
    with zopen(_phonon_fixture_path, mode="rt") as file:
        _phonon_doc = json.loads(file.read(), cls=MontyDecoder)

    _band_data = _phonon_doc.phonon_bandstructure
    pmv.BandStructureWidget(band_structure=_band_data, style="height: 400px;")
    return _band_data, _phonon_doc


@app.cell
def _(mo):
    mo.md("### DOS Widget\nUse matching phonon DOS data from the same fixture.")


@app.cell
def _(_phonon_doc, pmv):
    _dos_data = _phonon_doc.phonon_dos
    pmv.DosWidget(dos=_dos_data, style="height: 400px;")
    return (_dos_data,)


@app.cell
def _(mo):
    mo.md(
        "### Bands + DOS Widget\n"
        "Combine bands and DOS into a coordinated electronic-structure view."
    )


@app.cell
def _(_band_data, _dos_data, pmv):
    pmv.BandsAndDosWidget(
        band_structure=_band_data, dos=_dos_data, style="height: 500px;"
    )


# === Composition Grid ===


@app.cell
def _(mo):
    mo.md(
        "### Composition Widgets Grid\n"
        "Compare several compositions across pie, bar, and bubble modes."
    )


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
def _(mo):
    mo.md(
        "### Local Trajectory From Download\n"
        "Download a trajectory file once, cache it, and visualize it from "
        "local storage."
    )


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
def _(mo):
    mo.md(
        "### Remote Trajectory URL\n"
        "Render a trajectory directly from a remote URL with force vectors and bonds."
    )


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
def _(mo):
    mo.md(
        "### ASE Atoms MIME Rendering\n"
        "Display an ASE bulk structure via pymatviz MIME auto-rendering."
    )


@app.cell
def _(bulk):
    _ase_atoms = bulk("Al", "fcc", a=4.05)
    _ase_atoms *= (2, 2, 2)
    _ase_atoms


@app.cell
def _(mo):
    mo.md(
        "### ASE Molecule MIME Rendering\n"
        "Display an ASE molecule via pymatviz MIME auto-rendering."
    )


@app.cell
def _(molecule):
    _ase_molecule = molecule("H2O")
    _ase_molecule.center(vacuum=3.0)
    _ase_molecule


@app.cell
def _(mo):
    mo.md(
        "### PhonopyAtoms MIME Rendering\n"
        "Display a PhonopyAtoms structure via pymatviz MIME auto-rendering."
    )


@app.cell
def _(PhonopyAtoms):
    PhonopyAtoms(
        symbols=["Na", "Cl"],
        positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
    )


if __name__ == "__main__":
    app.run()
