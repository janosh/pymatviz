from __future__ import annotations

import copy
import re
import subprocess
import sys
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from ase import Atoms
from pymatgen.core import (
    Composition,
    IMolecule,
    IStructure,
    Lattice,
    Molecule,
    Structure,
)

import pymatviz as pmv
from pymatviz import process_data as pmv_pd
from pymatviz.enums import ElemCountMode
from tests.conftest import SI_ATOMS, SI_STRUCTS, y_pred, y_true


if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Any

    from pymatviz.typing import FormulaGroupBy


FE_P_O_COMPS = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3


@pytest.mark.parametrize(
    "inputs",
    [["Fe2 O3"] * 5 + ["Fe4 P4 O16"] * 3, FE_P_O_COMPS],
    ids=["strings", "compositions"],
)
@pytest.mark.parametrize(
    ("count_mode", "counts"),
    [
        (ElemCountMode.composition, {"Fe": 22, "O": 63, "P": 12}),
        (ElemCountMode.fractional_composition, {"Fe": 2.5, "O": 5, "P": 0.5}),
        (ElemCountMode.reduced_composition, {"Fe": 13, "O": 27, "P": 3}),
        (ElemCountMode.occurrence, {"Fe": 8, "O": 8, "P": 3}),
    ],
)
def test_count_elements(
    inputs: list[str] | list[Composition],
    count_mode: ElemCountMode,
    counts: dict[str, float],
) -> None:
    series = pmv_pd.count_elements(inputs, count_mode=count_mode)
    expected = pd.Series(counts, index=pmv.df_ptable.index, name="count")
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


@pytest.mark.parametrize("index_type", [int, float, str])
def test_count_elements_by_atomic_nums(index_type: type) -> None:
    """Accept integer atomic numbers without changing counts."""
    series_in = pd.Series(1, index=list(map(index_type, range(1, 119))))
    el_cts = pmv_pd.count_elements(series_in)
    expected = pd.Series(1, index=pmv.df_ptable.index, name="count")

    pd.testing.assert_series_equal(expected, el_cts)


@pytest.mark.parametrize(
    ("values", "kwargs", "match"),
    [
        ({"Fe": 1, "Zz": 2}, {}, r"Unexpected element symbol\(s\): Zz"),
        ({1.5: 4}, {}, "Atomic numbers must be integers"),
        ({1.5: 4, "2": 3}, {}, "Atomic numbers must be integers"),
        *[
            (values, {}, "assumed to represent atomic numbers")
            for start, stop in [(-1, 10), (100, 200)]
            for values in (
                dict.fromkeys(range(start, stop), 0),
                {str(idx): 0 for idx in range(start, stop)},
            )
        ],
        (
            ["Fe2O3", "Fe4P4O16"],
            {"exclude_elements": ["Fe", "P", "Zz"]},
            re.escape("Unexpected symbol(s) Zz in excluded_elements=['Fe', 'P', 'Zz']"),
        ),
    ],
)
def test_count_elements_raises(values: Any, kwargs: dict[str, Any], match: str) -> None:
    """Reject unknown symbols, non-integral or out-of-range atomic numbers."""
    with pytest.raises(ValueError, match=match):
        pmv_pd.count_elements(values, **kwargs)


@pytest.mark.parametrize(
    ("values", "kwargs", "counts", "dropped"),
    [
        (
            ["Fe2O3", Composition("Fe4P4O16"), "LiCoO2", Composition("NaCl")],
            {},
            {"Fe": 6, "O": 21, "P": 4, "Li": 1, "Co": 1, "Na": 1, "Cl": 1},
            [],
        ),
        (FE_P_O_COMPS, {"exclude_elements": ["Fe", "P"]}, {"O": 63}, ["Fe", "P"]),
        (FE_P_O_COMPS, {"exclude_elements": "Fe"}, {"O": 63, "P": 12}, ["Fe"]),
    ],
    ids=["mixed-input", "exclude-list", "exclude-str"],
)
@pytest.mark.parametrize("fill_value", [None, 0])
def test_count_elements_options(
    values: list[str | Composition],
    kwargs: dict[str, Any],
    counts: dict[str, int],
    dropped: list[str],
    fill_value: float | None,
) -> None:
    """Count mixed inputs, drop excluded elements and fill missing ones."""
    series = pmv_pd.count_elements(values, fill_value=fill_value, **kwargs)
    expected = pd.Series(counts, index=pmv.df_ptable.index.drop(dropped), name="count")
    if fill_value is not None:
        expected = expected.fillna(fill_value)
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            ["Fe2O3", "Fe4O6", "FeO", "Li2O", "LiFeO2"],
            [("binary", "Fe-O", 3), ("binary", "Li-O", 1), ("ternary", "Fe-Li-O", 1)],
        ),
        (  # sorted by arity before name
            ["Li2O", "Fe2O3", "NaCl", "C", "Li-Fe-O"],
            [
                ("unary", "C", 1),
                ("binary", "Cl-Na", 1),
                ("binary", "Fe-O", 1),
                ("binary", "Li-O", 1),
                ("ternary", "Fe-Li-O", 1),
            ],
        ),
        (
            ["Fe2O3", Composition("Fe4O6"), "Fe-O", Composition("FeO")],
            [("binary", "Fe-O", 4)],
        ),
    ],
    ids=["formulas", "arity-order", "mixed-input"],
)
def test_count_formulas_chem_sys(
    data: list[str | Composition], expected: list[tuple[str, str, int]]
) -> None:
    """Group formulas, chemical systems and compositions by sorted chemical system."""
    df_out = pmv_pd.count_formulas(data)
    assert list(df_out.columns) == ["arity_name", "chem_sys", "count"]
    assert list(df_out.itertuples(index=False, name=None)) == expected


@pytest.mark.parametrize(
    ("data", "group_by", "error_match"),
    [
        ([], "chem_sys", "Empty input: data sequence is empty"),
        (["Fe2O3", "NotAFormula"], "chem_sys", "Invalid formula"),
        (["Fe-Zz"], "chem_sys", "Invalid elements in system"),
        (["Fe2O3"], "invalid", "Invalid group_by="),
    ],
)
def test_count_formulas_raises(data: list, group_by: str, error_match: str) -> None:
    """Reject empty data, invalid formulas/elements and unknown group_by."""
    with pytest.raises(ValueError, match=error_match):
        pmv_pd.count_formulas(data, group_by=group_by)  # ty: ignore[invalid-argument-type]


def test_count_formulas_composition_objects() -> None:
    data = [
        Composition("Fe2O3"),
        Composition("Fe4O6"),
        Composition("FeO"),
        Composition("Li2O"),
    ]
    df_out = pmv_pd.count_formulas(data, group_by="reduced_formula")

    assert len(df_out) == 3
    assert df_out["count"].sum() == 4

    fe_o_counts = df_out[df_out["formula"].str.contains("Fe")]
    assert len(fe_o_counts) == 2
    assert fe_o_counts[fe_o_counts["formula"] == "Fe2O3"]["count"].iloc[0] == 2


@pytest.mark.parametrize(
    ("group_by", "expected_formulas", "expected_counts"),
    [
        (
            "formula",
            ["Fe2O3", "Fe4O6", "FeO"],
            [1, 1, 1],
        ),
        (
            "reduced_formula",
            ["Fe2O3", "FeO"],
            [2, 1],
        ),
        (
            "chem_sys",
            ["Fe-O"],
            [3],
        ),
    ],
)
def test_count_formulas_grouping_modes(
    group_by: FormulaGroupBy, expected_formulas: list[str], expected_counts: list[int]
) -> None:
    data = ["Fe2O3", "Fe4O6", "FeO"]
    df_out = pmv_pd.count_formulas(data, group_by=group_by)

    if group_by == "chem_sys":
        assert list(df_out["chem_sys"]) == expected_formulas
    else:
        assert list(df_out["formula"]) == expected_formulas
    assert list(df_out["count"]) == expected_counts


PMG_FORMULA_0 = SI_STRUCTS[0].formula
SI_ISTRUCTURE_0 = IStructure.from_sites(SI_STRUCTS[0])


class DerivedAtoms(Atoms):
    """Exercise adapter-free conversion of caller-defined ASE subclasses."""


H2O_MOL = Molecule(["H", "H", "O"], [[0, 0, 0], [0, 0, 1.5], [0, 0, 0.75]])
CO2_MOL = Molecule(["C", "O", "O"], [[0, 0, 0], [0, 0, 1.2], [0, 0, -1.2]])
H2O_IMOL = IMolecule(["H", "H", "O"], [[0, 0, 0], [0, 0, 1.5], [0, 0, 0.75]])
CO2_IMOL = IMolecule(["C", "O", "O"], [[0, 0, 0], [0, 0, 1.2], [0, 0, -1.2]])


def _formula(obj: Structure | IStructure | Molecule | IMolecule) -> str:
    return obj.formula if hasattr(obj, "formula") else obj.composition.formula


def _seq_expected(
    *items: Structure | IStructure | Molecule | IMolecule,
) -> dict[str, Structure | IStructure | Molecule | IMolecule]:
    return {f"{idx} {_formula(item)}": item for idx, item in enumerate(items, start=1)}


PMG_EXPECTED_DICT = _seq_expected(*SI_STRUCTS)

_normalize_structures_cases = [
    ("single_pmg_structure", SI_STRUCTS[0], {PMG_FORMULA_0: SI_STRUCTS[0]}),
    ("single_istructure", SI_ISTRUCTURE_0, {PMG_FORMULA_0: SI_ISTRUCTURE_0}),
    ("list_of_pmg_structures", SI_STRUCTS, PMG_EXPECTED_DICT),
    (
        "dict_of_pmg_structures",
        {"s0_key": SI_STRUCTS[0], "s1_key": SI_STRUCTS[1]},
        {"s0_key": SI_STRUCTS[0], "s1_key": SI_STRUCTS[1]},
    ),
    ("single_ase_atoms", SI_ATOMS[0], {PMG_FORMULA_0: SI_STRUCTS[0]}),
    ("list_of_ase_atoms", SI_ATOMS, PMG_EXPECTED_DICT),
    (
        "dict_of_ase_atoms",
        {"a0_key": SI_ATOMS[0], "a1_key": SI_ATOMS[1]},
        {"a0_key": SI_STRUCTS[0], "a1_key": SI_STRUCTS[1]},
    ),
    ("mixed_list_pmg_and_ase", [SI_STRUCTS[0], SI_ATOMS[1]], PMG_EXPECTED_DICT),
    (
        "mixed_dict_pmg_and_ase",
        {"pmg_key": SI_STRUCTS[0], "ase_key": SI_ATOMS[1]},
        {"pmg_key": SI_STRUCTS[0], "ase_key": SI_STRUCTS[1]},
    ),
    ("single_molecule", H2O_MOL, {H2O_MOL.composition.formula: H2O_MOL}),
    ("single_imolecule", H2O_IMOL, {H2O_IMOL.composition.formula: H2O_IMOL}),
    ("list_molecules", [H2O_MOL, CO2_MOL], _seq_expected(H2O_MOL, CO2_MOL)),
    ("list_imolecules", [H2O_IMOL, CO2_IMOL], _seq_expected(H2O_IMOL, CO2_IMOL)),
    (
        "dict_molecules",
        {"h2o": H2O_MOL, "co2": CO2_MOL},
        {"h2o": H2O_MOL, "co2": CO2_MOL},
    ),
    (
        "dict_imolecules",
        {"h2o": H2O_IMOL, "co2": CO2_IMOL},
        {"h2o": H2O_IMOL, "co2": CO2_IMOL},
    ),
    (
        "mixed_struct_mol",
        [SI_STRUCTS[0], H2O_MOL],
        _seq_expected(SI_STRUCTS[0], H2O_MOL),
    ),
    (
        "mixed_dict_struct_mol",
        {"s": SI_STRUCTS[0], "m": H2O_MOL},
        {"s": SI_STRUCTS[0], "m": H2O_MOL},
    ),
    (
        "mixed_istruct_imol",
        [SI_ISTRUCTURE_0, H2O_IMOL],
        _seq_expected(SI_ISTRUCTURE_0, H2O_IMOL),
    ),
    (
        "series_structures",
        pd.Series(SI_STRUCTS, index=["s1", "s2"]),
        {"s1": SI_STRUCTS[0], "s2": SI_STRUCTS[1]},
    ),
    (
        "series_molecules",
        pd.Series([H2O_MOL, CO2_MOL], index=["h2o", "co2"]),
        {"h2o": H2O_MOL, "co2": CO2_MOL},
    ),
    (
        "series_struct_mol",
        pd.Series([SI_STRUCTS[0], H2O_MOL], index=["s", "m"]),
        {"s": SI_STRUCTS[0], "m": H2O_MOL},
    ),
]


@pytest.mark.parametrize(
    ("input_raw", "expected_output_dict"),
    [
        pytest.param(input_raw, expected_output_dict, id=case_name)
        for case_name, input_raw, expected_output_dict in _normalize_structures_cases
    ],
)
def test_normalize_structures(
    input_raw: Any,
    expected_output_dict: dict[Hashable, Structure | IStructure | Molecule | IMolecule],
) -> None:
    result_dict = pmv_pd.normalize_structures(input_raw)

    assert result_dict == expected_output_dict


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
        pmv_pd.normalize_structures(invalid_input)


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
        (Structure([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ["Fe"], [[0, 0, 0]]), False),
        (NotAse(), False),
        ("string", False),
        (123, False),
        ([1, 2, 3], False),
        ({"key": "value"}, False),
        (None, False),
    ],
)
def test_is_ase_atoms(obj: object, expected: bool) -> None:
    assert pmv_pd.is_ase_atoms(obj) == expected


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        pytest.param("phonopy_atoms", True, id="phonopy_atoms"),
        # Supercell/Primitive subclass PhonopyAtoms and must count as structures too
        pytest.param("phonopy_supercell", True, id="phonopy_supercell"),
        (Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]]), False),
        ("string", False),
        (123, False),
        (None, False),
    ],
)
def test_is_phonopy_atoms(obj: object, expected: bool) -> None:
    if obj in ("phonopy_atoms", "phonopy_supercell"):
        pytest.importorskip("phonopy")
        from phonopy.structure.atoms import PhonopyAtoms
        from phonopy.structure.cells import Supercell

        unit_cell = PhonopyAtoms(
            symbols=["Si"],
            positions=[[0, 0, 0]],
            cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
        supercell_matrix = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        obj = (
            unit_cell
            if obj == "phonopy_atoms"
            else Supercell(unit_cell, supercell_matrix)
        )
    assert pmv_pd.is_phonopy_atoms(obj) is expected


def test_df_to_arrays() -> None:
    df_regr = pd.DataFrame([y_true, y_pred]).T
    x1, y1 = pmv_pd.df_to_arrays(None, y_true, y_pred)
    x_col, y_col = df_regr.columns[:2]
    x2, y2 = pmv_pd.df_to_arrays(df_regr, x_col, y_col)
    assert x1 == pytest.approx(x2)
    assert y1 == pytest.approx(y2)
    assert x1 == pytest.approx(y_true)
    assert y1 == pytest.approx(y_pred)

    with pytest.raises(TypeError, match="df should be pandas DataFrame or None"):
        pmv_pd.df_to_arrays("foo", y_true, y_pred)  # ty: ignore[invalid-argument-type]

    bad_col_name = "not-real-col-name"
    with pytest.raises(KeyError) as exc:
        pmv_pd.df_to_arrays(df_regr, bad_col_name, df_regr.columns[0])

    assert "not-real-col-name" in str(exc.value)

    args = pmv_pd.df_to_arrays(42, "foo", "bar", strict=False)  # ty: ignore[invalid-argument-type]
    assert args == ["foo", "bar"]


@pytest.mark.parametrize(
    (
        "bin_by_cols",
        "group_by_cols",
        "n_bins",
        "expected_n_bins",
        "verbose",
        "density_col",
        "expected_n_rows",
    ),
    [
        (["A"], [], 2, [2], True, "", 2),
        (["A", "B"], [], 2, [2, 2], True, "kde_bin_counts", 4),
        (["A", "B"], [], [2, 3], [2, 3], False, "kde_bin_counts", 6),
        (["A"], ["B"], 2, [2], False, "", 30),
    ],
)
def test_bin_df_cols(
    bin_by_cols: list[str],
    group_by_cols: list[str],
    n_bins: int | list[int],
    expected_n_bins: list[int],
    verbose: bool,
    density_col: str,
    expected_n_rows: int,
    df_float: pd.DataFrame,
) -> None:
    idx_col = "index"
    # don't move this below df_float.copy() line
    df_float.index.name = idx_col

    # keep copy of original DataFrame to assert it is not modified
    # not using df.copy(deep=True) here for extra sensitivity, doc str says
    # not as deep as deepcopy
    df_float_orig = copy.deepcopy(df_float)

    bin_counts_col = "bin_counts"
    df_binned = pmv_pd.bin_df_cols(
        df_float,
        bin_by_cols,
        group_by_cols=group_by_cols,
        n_bins=n_bins,
        verbose=verbose,
        bin_counts_col=bin_counts_col,
        density_col=density_col,
    )

    assert len(df_binned) == expected_n_rows, f"{len(df_binned)=} {expected_n_rows=}"
    assert len(df_binned) <= len(df_float), f"{len(df_binned)=} {len(df_float)=}"
    assert df_binned.index.name == idx_col

    # ensure binned DataFrame has a minimum set of expected columns
    expected_cols = {bin_counts_col, *df_float, *(f"{col}_bins" for col in bin_by_cols)}
    assert {*df_binned} >= expected_cols, (
        f"{set(df_binned)=}\n{expected_cols=},\n{bin_by_cols=}\n{group_by_cols=}"
    )

    # validate the number of unique bins for each binned column
    for col, n_bins_expec in zip(bin_by_cols, expected_n_bins, strict=True):
        assert df_binned[f"{col}_bins"].nunique() == n_bins_expec

    # ensure original DataFrame is not modified
    pd.testing.assert_frame_equal(df_float, df_float_orig)

    # Check that the index values of df_binned are a subset of df_float
    assert set(df_binned.index).issubset(set(df_float.index))

    # Check that bin_counts column exists and contains only integers
    assert bin_counts_col in df_binned
    assert df_binned[bin_counts_col].dtype in [int, "int64"]

    # If density column is specified, it must exist and be scaled to the number
    # of data points (regression: was scaled by the number of binned columns)
    if density_col:
        assert df_binned[density_col].sum() == pytest.approx(len(df_float))
    else:
        assert density_col not in df_binned


def test_bin_df_cols_raises() -> None:
    df_dummy = pd.DataFrame({"col1": [1, 2, 3, 4], "col2": [2, 3, 4, 5]})
    bin_by_cols = ["col1", "col2"]

    # test error when passing n_bins as list but list has wrong length
    with pytest.raises(
        ValueError, match=re.escape("len(bin_by_cols)=2 != len(n_bins)=1")
    ):
        pmv_pd.bin_df_cols(df_dummy, bin_by_cols, n_bins=[2])


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([1, 2, 3, 225], [1, 2, 3, 225]),
        ([225.0, 1.0, 62.0], [225, 1, 62]),
        (pd.Series([225, 1, 62], dtype="Float64"), [225, 1, 62]),
        (pd.Series([1, 225, 167]), [1, 225, 167]),
        (["Fm-3m", "P1", "Pnma"], [225, 1, 62]),  # Hermann-Mauguin symbols -> numbers
    ],
)
@pytest.mark.parametrize("as_series", [False, True])
def test_normalize_spacegroups(
    data: list | pd.Series, expected: list, as_series: bool
) -> None:
    """Test normalize_spacegroups with various input types."""
    expected_series = pd.Series(expected)
    if as_series:
        index = pd.Index([f"sample-{idx}" for idx in range(len(data))], name="sample")
        data = pd.Series(list(data), index=index, name="spacegroup")
        expected_series = pd.Series(expected, index=index, name="spacegroup")
    pd.testing.assert_series_equal(pmv_pd.normalize_spacegroups(data), expected_series)


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
    pd.testing.assert_series_equal(pmv_pd.normalize_spacegroups(data), expected)


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
from pymatviz.process_data import normalize_spacegroups
from pymatviz.structure.helpers import get_subplot_title

structure = Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])
immutable = IStructure.from_sites(structure)
assert normalize_spacegroups([structure, immutable]).tolist() == [221, 221]
assert get_subplot_title(immutable, 0, 1, None)["text"] == "1. Si1 (spg=221)"
assert "221" in pmv.spacegroup_sunburst([structure]).data[0].labels
pmv.StructureWidget(structure=structure.as_dict())
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
    match = "missing values" if pd.isna(invalid_val) else r"must be in \[1, 230\]"
    with pytest.raises(ValueError, match=match):
        pmv_pd.normalize_spacegroups(data)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ([], "Cannot normalize empty spacegroup data"),
        (["Fm-3m", "InvalidSymbol", "P1"], "InvalidSymbol"),
    ],
)
def test_normalize_spacegroups_raises(data: list[str], match: str) -> None:
    """Reject empty input and unknown Hermann-Mauguin symbols."""
    with pytest.raises(ValueError, match=match):
        pmv_pd.normalize_spacegroups(data)


@pytest.mark.parametrize(
    ("labels_with_counts", "expected_labels"),
    [
        (True, ["A: 4", "B: 1", "C: 3"]),
        ("percent", ["A: 100.0%", "B: 25.0%", "C: 75.0%"]),
        (False, ["A", "B", "C"]),
    ],
)
def test_sankey_flow_data(
    labels_with_counts: bool | str, expected_labels: list[str]
) -> None:
    """Deduplicate shared nodes, sum their source+target counts and index links."""
    df_flows = pd.DataFrame({"src": ["A", "A", "A", "B"], "tgt": ["C", "C", "A", "C"]})
    flow_data = pmv_pd.sankey_flow_data(
        df_flows, ["src", "tgt"], labels_with_counts=labels_with_counts
    )
    assert flow_data == {
        "source": ["A", "A", "B"],
        "target": ["C", "A", "C"],
        "value": [2, 1, 1],
        "labels": expected_labels,
        "source_indices": [0, 0, 1],
        "target_indices": [2, 0, 2],
    }


def test_sankey_flow_data_invalid_cols_raises() -> None:
    """Test that invalid columns raise ValueError."""
    df_test = pd.DataFrame({"A": ["x"], "B": ["p"]})
    with pytest.raises(ValueError, match="should specify exactly two columns"):
        pmv_pd.sankey_flow_data(df_test, ["A"])
