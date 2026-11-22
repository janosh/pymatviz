"""Tests for structure input detection, conversion, site helpers and space groups."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import matscikit as mk
import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from pymatgen.core import (
    DummySpecies,
    Element,
    IMolecule,
    IStructure,
    Lattice,
    Molecule,
    Species,
    Structure,
)

from pymatviz.structures import (
    is_ase_atoms,
    is_phonopy_atoms,
    normalize_spacegroups,
    normalize_structures,
    site_elements,
    site_symbols,
    structure_dict,
    to_structure,
)
from tests.conftest import SI2_STRUCT, SI_ATOMS, SI_STRUCTS


if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any


SI_ISTRUCT = IStructure.from_sites(SI_STRUCTS[0])
H2O = Molecule(["H", "H", "O"], [[0, 0, 0], [0, 0, 1.5], [0, 0, 0.75]])
H2O_IMOL = IMolecule.from_sites(H2O)


class DerivedAtoms(Atoms):
    """Exercise adapter-free conversion of caller-defined ASE subclasses."""


@pytest.mark.parametrize(
    ("input_raw", "expected_keys"),
    [
        (SI_STRUCTS[0], ["Si2"]),
        (SI_ISTRUCT, ["Si2"]),
        (SI_ATOMS[0], ["Si2"]),
        (to_structure(SI_STRUCTS[0]), ["Si2"]),
        (H2O_IMOL, ["H2 O1"]),
        ([SI_ATOMS[0], H2O], ["1 Si2", "2 H2 O1"]),  # 1-based index and formula
        ({"si": SI_ATOMS[1], "h2o": H2O}, ["si", "h2o"]),
        (pd.Series([SI_ISTRUCT], index=["s"]), ["s"]),
    ],
    ids=[
        "structure",
        "istructure",
        "atoms",
        "matscikit",
        "imolecule",
        "list",
        "dict",
        "series",
    ],
)
def test_normalize_structures(input_raw: Any, expected_keys: list[Hashable]) -> None:
    """Single inputs key by formula, mappings keep keys, all become matscikit."""
    struct_dict = normalize_structures(input_raw)
    assert list(struct_dict) == expected_keys
    assert all(isinstance(struct, mk.Structure) for struct in struct_dict.values())


@pytest.mark.parametrize("source", [SI_STRUCTS[0], SI_ATOMS[0]], ids=["pmg", "ase"])
def test_to_structure_geometry(source: Any) -> None:
    """Pymatgen and ASE inputs keep lattice, positions, species and pbc."""
    struct = to_structure(source)
    np.testing.assert_allclose(struct.lattice, SI_STRUCTS[0].lattice.matrix, atol=1e-12)
    np.testing.assert_allclose(struct.positions, SI_STRUCTS[0].cart_coords, atol=1e-12)
    assert struct.symbols == ["Si", "Si"]
    assert struct.pbc == (True, True, True)


def test_to_structure_species_and_molecules() -> None:
    """Oxidation states sum per element, dummy species become vacancies, molecules
    turn off periodic boundaries, matscikit Structures pass through.
    """
    site_species = [
        {Species("Fe", 2): 0.25, Species("Fe", 3): 0.5, DummySpecies("X"): 0.25},
        Element("O"),
    ]
    pmg_struct = IStructure(Lattice.cubic(3), site_species, [[0, 0, 0], [0.5] * 3])
    struct = to_structure(pmg_struct)
    assert struct.site_occupancies == [{"Fe": 0.75}, {"O": 1.0}]
    assert not struct.is_ordered
    assert to_structure(struct) is struct

    mol = to_structure(H2O)
    assert mol.pbc == (False, False, False)
    np.testing.assert_allclose(mol.positions, H2O.cart_coords)

    with pytest.raises(TypeError, match="expected a matscikit or pymatgen"):
        to_structure(42)
    with pytest.raises(ValueError, match="unknown structure file format"):
        to_structure("Si")  # strings are file paths
    vacancy_only = IStructure(Lattice.cubic(3), [DummySpecies("X")], [[0, 0, 0]])
    with pytest.raises(ValueError, match="only vacancies"):
        to_structure(vacancy_only)


def test_structure_dict() -> None:
    """Pymatgen objects keep their as_dict, others become pymatgen-style dicts,
    molecules without lattice or fractional coordinates.
    """
    assert structure_dict(SI2_STRUCT) == SI2_STRUCT.as_dict()  # keeps Si4+
    periodic = structure_dict(SI_ATOMS[0])
    assert periodic["lattice"]["pbc"] == [True, True, True]
    assert [site["species"] for site in periodic["sites"]] == [
        [{"element": "Si", "occu": 1}]
    ] * 2
    mol = structure_dict(to_structure(H2O))
    assert mol["@class"] == "Molecule"
    assert "lattice" not in mol
    assert all("abc" not in site and "xyz" in site for site in mol["sites"])


def test_site_elements_and_symbols() -> None:
    """Element sets and majority symbols of ordered and disordered sites."""
    from pymatviz.utils.testing import hea_disordered

    assert site_elements(hea_disordered) == [{"Fe", "Ni", "Cr"}, {"Al"}]
    assert site_symbols(hea_disordered) == ["Fe", "Al"]


@pytest.mark.parametrize(
    ("invalid_input", "err_cls", "error_match"),
    [
        ("not a structure", TypeError, "Input must be a matscikit or pymatgen"),
        (12345, TypeError, "Input must be a matscikit or pymatgen"),
        ([SI_STRUCTS[0], "invalid"], ValueError, "unknown structure file format"),
        ([], ValueError, "Cannot plot empty set of structures"),
        ({}, ValueError, "Cannot plot empty set of structures"),
    ],
)
def test_normalize_structures_errors(
    invalid_input: Any, err_cls: type[Exception], error_match: str
) -> None:
    """Reject unsupported and empty inputs."""
    with pytest.raises(err_cls, match=error_match):
        normalize_structures(invalid_input)


class MockAseAtoms:
    """Mock object whose type identity matches ASE Atoms."""

    __module__ = "ase.atoms"
    __qualname__ = "Atoms"


class MockMsonAtoms:
    """Mock object whose type identity matches pymatgen MSONAtoms."""

    __module__ = "pymatgen.io.ase"
    __qualname__ = "MSONAtoms"


class NotAse:
    """Mock object whose type identity should not match ASE Atoms."""

    __module__ = "some.other.module"
    __qualname__ = "SomeClass"


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (MockAseAtoms(), True),
        (MockMsonAtoms(), True),
        (SI_STRUCTS[0], False),
        (NotAse(), False),
        ("string", False),
        (None, False),
    ],
)
def test_is_ase_atoms(obj: object, expected: bool) -> None:
    assert is_ase_atoms(obj) == expected


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        pytest.param("phonopy_atoms", True, id="phonopy_atoms"),
        # Supercell/Primitive subclass PhonopyAtoms and must count as structures too
        pytest.param("phonopy_supercell", True, id="phonopy_supercell"),
        (SI_STRUCTS[0], False),
        (None, False),
    ],
)
def test_is_phonopy_atoms(obj: object, expected: bool) -> None:
    if obj in ("phonopy_atoms", "phonopy_supercell"):
        pytest.importorskip("phonopy")
        from phonopy.structure.atoms import PhonopyAtoms
        from phonopy.structure.cells import Supercell

        unit_cell = PhonopyAtoms(symbols=["Si"], positions=[[0] * 3], cell=np.eye(3))
        is_cell = obj == "phonopy_atoms"
        obj = unit_cell if is_cell else Supercell(unit_cell, np.diag([2, 1, 1]))
    assert is_phonopy_atoms(obj) is expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([1, 2, 3, 225], [1, 2, 3, 225]),
        ([225.0, 1.0, 62.0], [225, 1, 62]),
        (pd.Series([225, 1, 62], dtype="Float64"), [225, 1, 62]),
        (pd.Series([1, 225, 167]), [1, 225, 167]),
        (["Fm-3m", "P1", "Pnma"], [225, 1, 62]),  # Hermann-Mauguin symbols -> numbers
        # full, spaced and spaceless symbols from the lookup table, setting suffixes
        # (R-3m:H), repeated
        (
            ["P 1 2_1/c 1", "P4/m2_1/nc", "P2_1/c", "R-3m:H", "P2_1/c", "F m -3 m"],
            [14, 128, 14, 166, 14, 225],
        ),
    ],
)
@pytest.mark.parametrize("as_series", [False, True])
def test_normalize_spacegroups(
    data: list | pd.Series, expected: list, as_series: bool
) -> None:
    """Numbers and symbols map to ints, keeping a Series' index and name."""
    expected_series = pd.Series(expected)
    if as_series:
        index = pd.Index([f"sample-{idx}" for idx in range(len(data))], name="sample")
        data = pd.Series(list(data), index=index, name="spacegroup")
        expected_series = pd.Series(expected, index=index, name="spacegroup")
    pd.testing.assert_series_equal(normalize_spacegroups(data), expected_series)


@pytest.mark.parametrize(
    "structures",
    [
        SI_STRUCTS,
        SI_ATOMS,
        tuple(DerivedAtoms(atoms) for atoms in SI_ATOMS),
        tuple(map(to_structure, SI_STRUCTS)),
    ],
    ids=["pymatgen", "ase", "ase_subclass", "matscikit"],
)
@pytest.mark.parametrize("as_series", [False, True])
def test_normalize_spacegroups_with_structures(
    structures: tuple, as_series: bool
) -> None:
    """Preserve identifiers when extracting symmetry from either structure format."""
    from moyopy import MoyoDataset
    from moyopy.interface import MoyoAdapter

    expected = pd.Series(
        [MoyoDataset(MoyoAdapter.from_py_obj(struct)).number for struct in SI_STRUCTS]
    )
    data: tuple | pd.Series = structures
    if as_series:
        index = pd.Index(["sample-a", "sample-b"], name="sample")
        data = pd.Series(list(structures), index=index, name="spacegroup")
        expected.index, expected.name = index, "spacegroup"
    pd.testing.assert_series_equal(normalize_spacegroups(data), expected)


def test_normalize_spacegroups_disordered() -> None:
    """Disordered structures get the space groups of pymatgen's SpacegroupAnalyzer."""
    from matscikit.interop import structure_to_pymatgen_dict
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from pymatviz.utils.testing import disordered_demo_structures

    structs = list(disordered_demo_structures.values())
    expected = [
        SpacegroupAnalyzer(
            Structure.from_dict(structure_to_pymatgen_dict(struct))
        ).get_space_group_number()
        for struct in structs
    ]
    assert normalize_spacegroups(structs).tolist() == expected == [221, 191, 221, 221]


def test_core_features_without_pymatgen_or_ase() -> None:
    """Base imports, symmetry, RDF, coordination, chemical environments, compositions,
    XRD, phonon DOS and widgets need neither pymatgen nor ASE.
    """
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["ase"] = sys.modules["pymatgen"] = sys.modules["moyopy.interface"] = None
import matscikit as mk
import numpy as np
import pymatviz as pmv
from pymatviz.structures import normalize_spacegroups

structure = mk.Structure(np.eye(3) * 3, ["Si"], [[0, 0, 0]])
assert normalize_spacegroups([structure]).tolist() == [221]
assert pmv.spacegroup_sunburst([structure]).data[0]["children"][0]["label"] == "221"
pmv.structure(structure=structure)
pmv.rdf_plot(structure, mode="full")
pmv.coordination_bar_plot(structure, strategy=3.0)
pmv.coordination_bar_plot(structure, strategy="crystal_nn")
for method in ("crystal_nn", "chemenv"):
    pmv.chem_env_hierarchy(structure, method=method)
pmv.phonon_dos(mk.PhononDos([0, 1, 2], [0, 1, 0]))
pmv.xrd_plot(structure)
pmv.ptable_heatmap(pmv.count_elements(["Fe2O3", mk.Composition("NaCl")]))
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("invalid_val", [0, -1, 231, 500, 2.5, float("nan"), pd.NA])
@pytest.mark.parametrize("as_nullable_series", [False, True])
def test_normalize_spacegroups_invalid_number_raises(
    invalid_val: Any, as_nullable_series: bool
) -> None:
    """Reject out-of-range, fractional, and missing space-group numbers."""
    data = [1, invalid_val, 225]
    if as_nullable_series:
        data = pd.Series(data, dtype="Float64")
    elif not pd.isna(invalid_val):  # object dtype must not truncate 2.5 to 2
        data = pd.Series(data, dtype=object)
    match = "missing values" if pd.isna(invalid_val) else r"must be in \[1, 230\]"
    with pytest.raises(ValueError, match=match):
        normalize_spacegroups(data)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ([], "Cannot normalize empty spacegroup data"),
        (["Fm-3m", "InvalidSymbol", "P1"], "InvalidSymbol"),
        (["P2_1/c", "Fm-3m", "P1_bad"], "P1_bad"),
    ],
)
def test_normalize_spacegroups_raises(data: list[str], match: str) -> None:
    """Reject empty input and unknown Hermann-Mauguin symbols."""
    with pytest.raises(ValueError, match=match):
        normalize_spacegroups(data)
