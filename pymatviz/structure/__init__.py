"""2D and 3D plots of Structures."""

from __future__ import annotations

from typing import Final

from pymatgen.core import IStructure, Lattice
from pymatgen.core.periodic_table import DummySpecies

from pymatviz.structure.plotly import structure_2d, structure_3d


"""Disordered structures for testing and examples."""

fe3co4_disordered = IStructure(
    lattice=Lattice.cubic(5),
    species=[{"Fe": 0.75, "C": 0.25}, "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)
batio3_disordered = IStructure(
    lattice=Lattice.cubic(4.0),
    species=[
        "Ba",  # Pure Ba site
        {"Ti": 0.6, "Zr": 0.4},  # Disordered Ti/Zr site
        "O",
        "O",
        "O",  # Pure O sites
    ],
    coords=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
)

# Li-ion cathode with vacancy disorder
lico2_disordered = IStructure(
    lattice=Lattice.hexagonal(2.82, 14.05),
    species=[
        {"Li": 0.75, DummySpecies("X"): 0.25},  # Li site with vacancies (X = vacancy)
        "Co",
        "O",
        "O",
    ],
    coords=[(0, 0, 0), (0, 0, 0.5), (0, 0, 0.25), (0, 0, 0.75)],
)

# High-entropy alloy with ternary disorder (3 species on one site)
hea_disordered = IStructure(
    lattice=Lattice.cubic(3.6),
    species=[
        {"Fe": 0.5, "Ni": 0.3, "Cr": 0.2},  # Ternary disordered site
        "Al",  # Pure Al site
    ],
    coords=[(0, 0, 0), (0.5, 0.5, 0.5)],
)

# Complex multisite disorder - solid solution with multiple disordered sites
solid_solution = IStructure(
    lattice=Lattice.cubic(5.0),
    species=[
        {"Ca": 0.6, "Sr": 0.4},  # Disordered A-site
        {"Ti": 0.8, "Zr": 0.15, "Hf": 0.05},  # Ternary disordered B-site
        "O",  # Pure O sites
        "O",
        "O",
    ],
    coords=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
)

disordered_demo_structures: Final[dict[str, IStructure]] = {
    "BaTi₀.₆Zr₀.₄O₃": batio3_disordered,
    "Li₀.₇₅CoO₂ (with vacancies)": lico2_disordered,
    "Fe₀.₅Ni₀.₃Cr₀.₂Al": hea_disordered,
    "Ca₀.₆Sr₀.₄Ti₀.₈Zr₀.₁₅Hf₀.₀₅O₃": solid_solution,
}
