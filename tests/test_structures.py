"""Tests for structure input detection, conversion, site helpers and space groups."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from pymatgen.core import (
    Composition,
    DummySpecies,
    Element,
    IMolecule,
    IStructure,
    Lattice,
    Molecule,
    PeriodicSite,
    Species,
)

from pymatviz.structures import (
    get_site_elements,
    get_site_species,
    get_site_symbol,
    is_ase_atoms,
    is_phonopy_atoms,
    normalize_spacegroups,
    normalize_structures,
)
from tests.conftest import SI_ATOMS, SI_STRUCTS


if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any


SI_ISTRUCT = IStructure.from_sites(SI_STRUCTS[0])
H2O = Molecule(["H", "H", "O"], [[0, 0, 0], [0, 0, 1.5], [0, 0, 0.75]])
H2O_IMOL = IMolecule.from_sites(H2O)


class DerivedAtoms(Atoms):
    """Exercise adapter-free conversion of caller-defined ASE subclasses."""


@pytest.mark.parametrize(
    ("input_raw", "expected"),
    [
        (SI_STRUCTS[0], {SI_STRUCTS[0].formula: SI_STRUCTS[0]}),
        (SI_ISTRUCT, {SI_ISTRUCT.formula: SI_ISTRUCT}),
        (SI_ATOMS[0], {SI_STRUCTS[0].formula: SI_STRUCTS[0]}),
        (H2O_IMOL, {H2O.formula: H2O_IMOL}),
        (  # sequences key by 1-based index and formula
            [SI_ATOMS[0], H2O],
            {f"1 {SI_STRUCTS[0].formula}": SI_STRUCTS[0], f"2 {H2O.formula}": H2O},
        ),
        ({"si": SI_ATOMS[1], "h2o": H2O}, {"si": SI_STRUCTS[1], "h2o": H2O}),
        (pd.Series([SI_ISTRUCT], index=["s"]), {"s": SI_ISTRUCT}),
    ],
    ids=["structure", "istructure", "atoms", "imolecule", "list", "dict", "series"],
)
def test_normalize_structures(input_raw: Any, expected: dict[Hashable, Any]) -> None:
    """Single inputs key by formula, mappings keep keys, ASE Atoms become pymatgen."""
    assert normalize_structures(input_raw) == expected


@pytest.mark.parametrize(
    ("invalid_input", "err_cls", "error_match"),
    [
        ("not a structure", TypeError, "Input must be a pymatgen Structure"),
        (12345, TypeError, "Input must be a pymatgen Structure"),
        ([SI_STRUCTS[0], "invalid"], TypeError, "Item must be a pymatgen Structure"),
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
        # full, spaced and spaceless symbols from the lookup table (pymatgen rejects
        # "P4/m2_1/nc") plus looser spellings only pymatgen parses, repeated
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
    [SI_STRUCTS, SI_ATOMS, tuple(DerivedAtoms(atoms) for atoms in SI_ATOMS)],
    ids=["pymatgen", "ase", "ase_subclass"],
)
@pytest.mark.parametrize("as_series", [False, True])
def test_normalize_spacegroups_with_structures(
    structures: tuple, as_series: bool
) -> None:
    """Preserve identifiers when extracting symmetry from either structure format."""
    from moyopy import MoyoDataset
    from moyopy.interface import MoyoAdapter

    expected = pd.Series(
        [MoyoDataset(MoyoAdapter.from_py_obj(struct)).number for struct in structures]
    )
    data: tuple | pd.Series = structures
    if as_series:
        index = pd.Index(["sample-a", "sample-b"], name="sample")
        data = pd.Series(list(structures), index=index, name="spacegroup")
        expected.index, expected.name = index, "spacegroup"
    pd.testing.assert_series_equal(normalize_spacegroups(data), expected)


def test_symmetry_and_widgets_without_ase() -> None:
    """Base imports, structure symmetry and notebook widgets need no ASE adapter."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["ase"] = None
sys.modules["moyopy.interface"] = None
import pymatviz as pmv
from pymatgen.core import IStructure, Lattice, Structure
from pymatviz.structures import normalize_spacegroups

structure = Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])
immutable = IStructure.from_sites(structure)
assert normalize_spacegroups([structure, immutable]).tolist() == [221, 221]
assert pmv.spacegroup_sunburst([immutable]).data[0]["children"][0]["label"] == "221"
pmv.structure(structure=structure.as_dict())
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


@pytest.mark.parametrize(
    ("species_input", "expected_type", "expected_elements", "expected_symbol"),
    [
        ("Si", Element, {"Si"}, "Si"),
        (Element("O"), Element, {"O"}, "O"),
        (Species("Fe", 2), Species, {"Fe"}, "Fe"),
        (Composition({"Fe": 0.75, "Co": 0.25}), Composition, {"Fe", "Co"}, "Fe"),
        (DummySpecies("X"), DummySpecies, {"X"}, "X"),
        (
            Composition({"Li": 0.75, DummySpecies("X"): 0.25}),
            Composition,
            {"Li", "X"},
            "Li",
        ),
    ],
)
def test_get_site_species_elements_symbol(
    species_input: Any,
    expected_type: type,
    expected_elements: set[str],
    expected_symbol: str,
) -> None:
    """Species, element set and majority symbol for ordered and disordered sites."""
    site = PeriodicSite(species_input, [0, 0, 0], Lattice.cubic(3.0))
    assert isinstance(get_site_species(site), expected_type)
    assert get_site_elements(site) == expected_elements
    assert get_site_symbol(site) == expected_symbol
