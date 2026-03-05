"""marimo demo for pymatviz widgets.

Run with `marimo edit examples/widgets/marimo_demo.py --no-sandbox --watch`.
"""

# ruff: noqa: ANN001, ANN202, B018, N803
# marimo cells are generated as `def _(...):` with injected args — these rules
# conflict with marimo's code generation pattern and cannot be fixed per-cell.
# /// script
# dependencies = [
#     "pymatgen>=2024.1.1",
#     "ase>=3.22.0",
#     "phonopy>=2.20.0",
# ]
# ///

import marimo


__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import itertools
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
        itertools,
        mo,
        molecule,
        np,
        np_rng,
        os,
        pmv,
    )


@app.cell
def _(mo):
    mo.md("""
    ### Convex Hull Widget
    Build a compact Li-Fe-O phase diagram and visualize its convex hull.
    """)


@app.cell
def _(Composition, itertools, np_rng, pmv):
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

    _stable_phases = {
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
    _entries = [PDEntry(Composition(c), e) for c, e in _stable_phases.items()]
    _hull = PhaseDiagram(_entries)

    _seen = set(_stable_phases)
    for _li, _fe, _o in itertools.product(range(7), range(7), range(1, 8)):
        if _li == 0 and _fe == 0:
            continue
        _comp = Composition({"Li": _li, "Fe": _fe, "O": _o})
        if _comp.reduced_formula in _seen:
            continue
        _seen.add(_comp.reduced_formula)
        _delta = min(np_rng.lognormal(mean=-3.2, sigma=1.0), 0.25)
        _entries.append(
            PDEntry(_comp, _hull.get_hull_energy(_comp) + _delta * _comp.num_atoms)
        )
    _phase_diag = PhaseDiagram(_entries)

    pmv.ConvexHullWidget(entries=_phase_diag, style="height: 500px;")


@app.cell
def _(mo):
    mo.md("""
    ### Structure Widget
    Render a wurtzite GaN crystal with bonds in an interactive 3D view.
    """)


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
    mo.md("""
    ### Brillouin Zone Widget
    Show the reciprocal-space Brillouin zone for the structure above.
    """)


@app.cell
def _(_struct, pmv):
    pmv.BrillouinZoneWidget(
        structure=_struct, show_vectors=True, style="height: 400px;"
    )


@app.cell
def _(mo):
    mo.md("""
    ### XRD Widget
    Compute and display an XRD pattern for rutile TiO2.
    """)


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


@app.cell
def _(mo):
    mo.md("""
    ### Trajectory Widget
    Generate a short perturbed Fe trajectory and plot structure with
    scalar metadata.
    """)


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
        show_vectors=True,
        style="height: 600px;",
    )


@app.cell
def _(mo):
    mo.md("""
    ### Scatter Plot Widget
    Dual-axis comparison of `sin(x)` and `cos(x)` with shared x-values.
    """)


@app.cell
def _(np, pmv):
    x_vals = np.linspace(0, 6.0, 60)
    scatter_series = [
        dict(label="sin(x)", x=x_vals, y=np.sin(x_vals)),
        dict(label="cos(x)", x=x_vals, y=np.cos(x_vals), y_axis="y2"),
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
    mo.md("""
    ### Bar Plot Widget
    Grouped bars compare two model scores across the same sample indices.
    """)


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
    mo.md("""
    ### Histogram Widget
    Overlaid histograms show the distributions of the two trigonometric series.
    """)


@app.cell
def _(pmv, scatter_series):
    histogram_series = [
        {key: s[key] for key in ("label", "x", "y")} for s in scatter_series
    ]
    pmv.HistogramWidget(
        series=histogram_series,
        bins=20,
        mode="overlay",
        x_axis={"label": "Value"},
        y_axis={"label": "Count"},
        style="height: 360px;",
    )


@app.cell
def _(mo):
    mo.md("""
    ### Band Structure Widget
    Load realistic phonon band structure data from test fixtures.
    """)


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
    return (_band_data, _phonon_doc)


@app.cell
def _(mo):
    mo.md("""
    ### DOS Widget
    Use matching phonon DOS data from the same fixture.
    """)


@app.cell
def _(_phonon_doc, pmv):
    _dos_data = _phonon_doc.phonon_dos
    pmv.DosWidget(dos=_dos_data, style="height: 400px;")
    return (_dos_data,)


@app.cell
def _(mo):
    mo.md("""
    ### Bands + DOS Widget
    Combine bands and DOS into a coordinated electronic-structure view.
    """)


@app.cell
def _(_band_data, _dos_data, pmv):
    pmv.BandsAndDosWidget(
        band_structure=_band_data, dos=_dos_data, style="height: 500px;"
    )


@app.cell
def _(mo):
    mo.md("""
    ### Composition Widgets Grid
    Compare several compositions across pie, bar, and bubble modes.
    """)


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


@app.cell
def _(mo):
    mo.md("""
    ### Local Trajectory From Download
    Download a trajectory file once, cache it, and visualize it from
    local storage.
    """)


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
        show_vectors=False,
    )
    return (matterviz_traj_dir_url,)


@app.cell
def _(mo):
    mo.md("""
    ### Remote Trajectory URL
    Render a trajectory directly from a remote URL with force vectors and bonds.
    """)


@app.cell
def _(matterviz_traj_dir_url, pmv):
    _file_name = "Cr0.25Fe0.25Co0.25Ni0.25-mace-omat-qha.xyz.gz"
    pmv.TrajectoryWidget(
        data_url=f"{matterviz_traj_dir_url}/{_file_name}",
        display_mode="structure+scatter",
        show_vectors=True,
        vector_scale=0.5,
        vector_color="#ff4444",
        show_bonds=True,
        bonding_strategy="nearest_neighbor",
        style="height: 600px;",
    )


@app.cell
def _(Final):
    matterviz_iso_dir_url: Final = (
        "https://github.com/janosh/matterviz/raw/550d96d2/src/site/isosurfaces"
    )
    return (matterviz_iso_dir_url,)


@app.cell
def _(mo):
    mo.md("""
    ### Isosurface Rendering (CHGCAR)
    Load a VASP CHGCAR and render charge density isosurfaces.
    """)


@app.cell
def _(matterviz_iso_dir_url, pmv):
    pmv.StructureWidget(
        data_url=f"{matterviz_iso_dir_url}/Si-CHGCAR.gz",
        isosurface_settings={
            "isovalue": 0.05,
            "opacity": 0.6,
            "positive_color": "#3b82f6",
            "show_negative": False,
        },
        style="height: 500px;",
    )


@app.cell
def _(mo):
    mo.md("""
    ### Molecular Orbital Isosurface
    Caffeine HOMO orbital lobes from a Gaussian .cube file.
    """)


@app.cell
def _(matterviz_iso_dir_url, pmv):
    pmv.StructureWidget(
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


@app.cell
def _(mo):
    mo.md("""
    ### Periodic Table Widget
    Element heatmap showing atomic masses.
    """)


@app.cell
def _(pmv):
    pmv.PeriodicTableWidget(
        heatmap_values={
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
            "Fe": 55.85,
        },
        color_scale="interpolateViridis",
        style="height: 400px;",
    )


@app.cell
def _(mo):
    mo.md("""
    ### 3D Scatter Plot
    Random 3D point cloud.
    """)


@app.cell
def _(np_rng, pmv):
    pmv.ScatterPlot3DWidget(
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


@app.cell
def _(mo):
    mo.md("""
    ### Heatmap Matrix
    Element pair interaction strengths.
    """)


@app.cell
def _(pmv):
    _elems = ["Fe", "O", "Li", "Mn"]
    pmv.HeatmapMatrixWidget(
        x_items=_elems,
        y_items=_elems,
        values=[
            [1.0, 0.8, 0.3, 0.6],
            [0.8, 1.0, 0.2, 0.5],
            [0.3, 0.2, 1.0, 0.4],
            [0.6, 0.5, 0.4, 1.0],
        ],
        color_scale="interpolateBlues",
        style="height: 400px;",
    )


@app.cell
def _(mo):
    mo.md("""
    ### Space Group Bar Plot
    Distribution of space groups in a dataset.
    """)


@app.cell
def _(pmv):
    pmv.SpacegroupBarPlotWidget(
        data=[225, 225, 225, 166, 166, 62, 62, 62, 62, 139, 139, 12, 14, 14, 14, 14],
        style="height: 350px;",
    )


@app.cell
def _(mo):
    mo.md("""
    ### Chemical Potential Diagram
    Stability regions in Li-Fe-O chemical potential space.
    """)


@app.cell
def _(pmv):
    pmv.ChemPotDiagramWidget(
        entries=[
            {"name": "Li", "energy": -1.9, "composition": {"Li": 1}},
            {"name": "Fe", "energy": -8.3, "composition": {"Fe": 1}},
            {"name": "O2", "energy": -4.9, "composition": {"O": 1}},
            {"name": "Li2O", "energy": -14.3, "composition": {"Li": 2, "O": 1}},
            {"name": "Fe2O3", "energy": -25.0, "composition": {"Fe": 2, "O": 3}},
            {
                "name": "LiFeO2",
                "energy": -17.5,
                "composition": {"Li": 1, "Fe": 1, "O": 2},
            },
        ],
        style="height: 500px;",
    )


@app.cell
def _(mo):
    mo.md("""
    ### RDF Plot Widget
    Radial distribution function computed on-the-fly from the GaN structure.
    """)


@app.cell
def _(_struct, pmv):
    pmv.RdfPlotWidget(
        structures=_struct.as_dict(),
        cutoff=10,
        n_bins=80,
        style="height: 400px;",
    )


@app.cell
def _(mo):
    mo.md("""
    ### ASE Atoms MIME Rendering
    Display an ASE bulk structure via pymatviz MIME auto-rendering.
    """)


@app.cell
def _(bulk):
    _ase_atoms = bulk("Al", "fcc", a=4.05)
    _ase_atoms *= (2, 2, 2)
    _ase_atoms


@app.cell
def _(mo):
    mo.md("""
    ### ASE Molecule MIME Rendering
    Display an ASE molecule via pymatviz MIME auto-rendering.
    """)


@app.cell
def _(molecule):
    _ase_molecule = molecule("H2O")
    _ase_molecule.center(vacuum=3.0)
    _ase_molecule


@app.cell
def _(mo):
    mo.md("""
    ### PhonopyAtoms MIME Rendering
    Display a PhonopyAtoms structure via pymatviz MIME auto-rendering.
    """)


@app.cell
def _(PhonopyAtoms):
    PhonopyAtoms(
        symbols=["Na", "Cl"],
        positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
    )


if __name__ == "__main__":
    app.run()
