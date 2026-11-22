"""Tests for structure input detection, conversion, site helpers and space groups."""

from __future__ import annotations

from typing import TYPE_CHECKING

import kemio
import numpy as np
import pandas as pd
import pytest

from pymatviz.data import TEST_FILES, disordered_demo_structures, hea_disordered
from pymatviz.structures import (
    normalize_spacegroups,
    normalize_structures,
    site_elements,
    site_symbols,
    structure_dict,
    to_structure,
)
from tests.conftest import SI2_STRUCT, SI_STRUCTS


if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any


H2O = kemio.Structure(
    None, ["H", "H", "O"], [[0, 0, 0], [0, 0, 1.5], [0, 0, 0.75]], pbc=(False,) * 3
)


@pytest.mark.parametrize(
    ("input_raw", "expected_keys"),
    [
        (SI_STRUCTS[0], ["Si2"]),
        (H2O, ["H2 O1"]),
        ([SI_STRUCTS[0], H2O], ["1 Si2", "2 H2 O1"]),  # 1-based index and formula
        ({"si": SI_STRUCTS[1], "h2o": H2O}, ["si", "h2o"]),
        (pd.Series([SI_STRUCTS[0]], index=["s"]), ["s"]),
        ([f"{TEST_FILES}/structures/mp-12712.json"], ["1 Zr18 Pd24"]),  # file path
        (f"{TEST_FILES}/structures/mp-12712.json", ["Zr18 Pd24"]),  # single path
        (structure_dict(SI2_STRUCT), ["Si2"]),  # single pymatgen-format dict
    ],
    ids=[
        *("structure", "molecule", "list", "dict", "series", "path", "single_path"),
        "structure_dict",
    ],
)
def test_normalize_structures(input_raw: Any, expected_keys: list[Hashable]) -> None:
    """Single inputs key by formula, mappings keep keys, all become kemio."""
    struct_dict = normalize_structures(input_raw)
    assert list(struct_dict) == expected_keys
    assert all(isinstance(struct, kemio.Structure) for struct in struct_dict.values())


def test_to_structure_and_site_helpers() -> None:
    """Kemio Structures pass through, vacant sites keep their index, disordered
    sites give element sets and majority symbols, molecules become pymatgen-style
    dicts without lattice or fractional coordinates.
    """
    assert to_structure(SI2_STRUCT) is SI2_STRUCT
    with pytest.raises(TypeError, match="expected a kemio or pymatgen"):
        to_structure(42)
    with pytest.raises(FileNotFoundError, match="no structure file at 'Si'"):
        to_structure("Si")  # strings are file paths

    vacant = kemio.Structure(
        np.eye(3) * 3, [{}, "Fe"], [[0, 0, 0], [0.5] * 3], fractional=True
    )
    assert vacant.site_occupancies == [{}, {"Fe": 1.0}]
    assert site_symbols(vacant) == ["X", "Fe"]
    assert structure_dict(vacant)["sites"][0]["label"] == "X0+"
    assert site_elements(hea_disordered) == [{"Fe", "Ni", "Cr"}, {"Al"}]
    assert site_symbols(hea_disordered) == ["Fe", "Al"]

    periodic = structure_dict(SI2_STRUCT)
    assert periodic["lattice"]["pbc"] == [True, True, True]
    site_species = [site["species"] for site in periodic["sites"]]
    assert site_species == [[{"element": "Si", "occu": 1}]] * 2
    mol = structure_dict(H2O)
    assert mol["@class"] == "Molecule"
    assert "lattice" not in mol
    assert all("abc" not in site and "xyz" in site for site in mol["sites"])


@pytest.mark.parametrize(
    ("invalid_input", "err_cls", "error_match"),
    [
        ("not a structure", FileNotFoundError, "no structure file at 'not a struct"),
        (12345, TypeError, "Input must be a kemio or pymatgen"),
        (
            [SI_STRUCTS[0], "invalid"],
            FileNotFoundError,
            "no structure file at 'invalid'",
        ),
        ([], ValueError, "Cannot plot empty set of structures"),
        ({}, ValueError, "Cannot plot empty set of structures"),
        (
            pd.Series(SI_STRUCTS, index=["a", "a"]),
            ValueError,
            r"labels must be unique, got duplicates \['a'\]",
        ),
    ],
)
def test_normalize_structures_errors(
    invalid_input: Any, err_cls: type[Exception], error_match: str
) -> None:
    """Reject unsupported and empty inputs."""
    with pytest.raises(err_cls, match=error_match):
        normalize_structures(invalid_input)


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
        (SI_STRUCTS, [12, 129]),  # symmetry detected from structures
        # disordered structures are told apart by species and occupancy
        (list(disordered_demo_structures.values()), [221, 191, 221, 221]),
    ],
)
@pytest.mark.parametrize("as_series", [False, True])
def test_normalize_spacegroups(
    data: list | tuple | pd.Series, expected: list, as_series: bool
) -> None:
    """Numbers, symbols and structures map to ints, keeping a Series' index and name."""
    expected_series = pd.Series(expected)
    if as_series:
        index = pd.Index([f"sample-{idx}" for idx in range(len(data))], name="sample")
        data = pd.Series(list(data), index=index, name="spacegroup")
        expected_series = pd.Series(expected, index=index, name="spacegroup")
    pd.testing.assert_series_equal(normalize_spacegroups(data), expected_series)


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
    ("data", "err_cls", "match"),
    [
        ([], ValueError, "Cannot normalize empty spacegroup data"),
        (["Fm-3m", "InvalidSymbol", "P1"], ValueError, r"symbols: \['InvalidSymbol'\]"),
        (["P2_1/c", "Fm-3m", "P1_bad"], ValueError, r"symbols: \['P1_bad'\]"),
        ([225, "Fm-3m"], TypeError, r"one kind\), got \['int', 'str'\]"),
        (["Fm-3m", 225], TypeError, r"one kind\), got \['int', 'str'\]"),
        ([[225]], TypeError, r"one kind\), got \['list'\]"),
        ([SI2_STRUCT, 225], TypeError, r"one kind\), got \['Structure', 'int'\]"),
        # bools are no space group numbers although True == 1
        ([True, True], TypeError, r"one kind\), got \['bool'\]"),
    ],
)
def test_normalize_spacegroups_raises(
    data: list[Any], err_cls: type[Exception], match: str
) -> None:
    """Reject empty input, unknown symbols and mixed or unsupported item types."""
    with pytest.raises(err_cls, match=match):
        normalize_spacegroups(data)
