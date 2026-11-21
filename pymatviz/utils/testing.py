"""Testing related utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pymatgen.core import IStructure, Lattice
from pymatgen.core.periodic_table import DummySpecies

from pymatviz.utils import ROOT


if TYPE_CHECKING:
    from phonopy import Phonopy
    from pymatgen.core import Structure


TEST_FILES: str = f"{ROOT}/tests/files"

# === Disordered structures for tests and examples ===

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


def formula_spg_str(struct: Structure) -> str:
    """Get a string with formula and space group number for a structure.

    Args:
        struct: A pymatgen Structure object.

    Returns:
        String in format "Formula (spg_number)", e.g. "NaCl (225)".
    """
    return f"{struct.formula} ({struct.get_space_group_info()[1]})"


def load_phonopy_nacl() -> Phonopy:
    """Load phonopy NaCl object from test files.

    Returns a Phonopy class instance of NaCl 2x2x2 without symmetrizing fc2.
    This function can be used in both tests and examples to get a consistent
    phonopy object for demonstrations.

    Returns:
        Phonopy: Loaded phonopy object with force constants.

    Raises:
        ImportError: If phonopy is not installed.

    Example:
        >>> phonopy_nacl = load_phonopy_nacl()
        >>> phonopy_nacl.run_mesh([10, 10, 10])
        >>> phonopy_nacl.run_total_dos()
        >>> dos = phonopy_nacl.total_dos
    """
    import phonopy

    return phonopy.load(
        f"{TEST_FILES}/phonons/NaCl/phonopy_disp.yaml.xz",
        force_sets_filename=f"{TEST_FILES}/phonons/NaCl/force_sets.dat",
        symmetrize_fc=False,
        produce_fc=True,
    )
