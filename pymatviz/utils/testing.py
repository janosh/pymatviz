"""Testing related utils."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import matscikit as mk
import numpy as np

from pymatviz.utils import ROOT


if TYPE_CHECKING:
    from phonopy import Phonopy


TEST_FILES: str = f"{ROOT}/tests/files"

# === Disordered structures for tests and examples ===


def _hexagonal(a: float, c: float) -> np.ndarray:
    """Row-vector lattice of a hexagonal cell (pymatgen's Lattice.hexagonal)."""
    return np.array([[a, 0, 0], [-a / 2, a * 3**0.5 / 2, 0], [0, 0, c]])


fe3co4_disordered = mk.Structure(
    np.eye(3) * 5,
    [{"Fe": 0.75, "C": 0.25}, "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5]],
    fractional=True,
)
batio3_disordered = mk.Structure(
    np.eye(3) * 4.0,
    [
        "Ba",  # Pure Ba site
        {"Ti": 0.6, "Zr": 0.4},  # Disordered Ti/Zr site
        "O",
        "O",
        "O",  # Pure O sites
    ],
    [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
    fractional=True,
)

# Li-ion cathode with vacancy disorder
lico2_disordered = mk.Structure(
    _hexagonal(2.82, 14.05),
    [
        {"Li": 0.75},  # Li site with 25% vacancies
        "Co",
        "O",
        "O",
    ],
    [(0, 0, 0), (0, 0, 0.5), (0, 0, 0.25), (0, 0, 0.75)],
    fractional=True,
)

# High-entropy alloy with ternary disorder (3 species on one site)
hea_disordered = mk.Structure(
    np.eye(3) * 3.6,
    [
        {"Fe": 0.5, "Ni": 0.3, "Cr": 0.2},  # Ternary disordered site
        "Al",  # Pure Al site
    ],
    [(0, 0, 0), (0.5, 0.5, 0.5)],
    fractional=True,
)

# Complex multisite disorder - solid solution with multiple disordered sites
solid_solution = mk.Structure(
    np.eye(3) * 5.0,
    [
        {"Ca": 0.6, "Sr": 0.4},  # Disordered A-site
        {"Ti": 0.8, "Zr": 0.15, "Hf": 0.05},  # Ternary disordered B-site
        "O",  # Pure O sites
        "O",
        "O",
    ],
    [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
    fractional=True,
)

disordered_demo_structures: Final[dict[str, mk.Structure]] = {
    "BaTi₀.₆Zr₀.₄O₃": batio3_disordered,
    "Li₀.₇₅CoO₂ (with vacancies)": lico2_disordered,
    "Fe₀.₅Ni₀.₃Cr₀.₂Al": hea_disordered,
    "Ca₀.₆Sr₀.₄Ti₀.₈Zr₀.₁₅Hf₀.₀₅O₃": solid_solution,
}


def formula_spg_str(struct: Any) -> str:
    """Get a string with formula and space group number for a structure.

    Args:
        struct: An ordered structure (matscikit, pymatgen, ASE, phonopy).

    Returns:
        String in format "Formula (spg_number)", e.g. "Na4 Cl4 (225)".
    """
    from pymatviz.structures import to_structure

    struct = to_structure(struct)
    return f"{struct.composition.formula} ({mk.symmetry_dataset(struct).number})"


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
