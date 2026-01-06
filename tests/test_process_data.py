from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING

import pandas as pd
import pytest
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


@pytest.mark.parametrize(
    ("count_mode", "counts"),
    [
        (ElemCountMode.composition, {"Fe": 22, "O": 63, "P": 12}),
        (ElemCountMode.fractional_composition, {"Fe": 2.5, "O": 5, "P": 0.5}),
        (ElemCountMode.reduced_composition, {"Fe": 13, "O": 27, "P": 3}),
        (ElemCountMode.occurrence, {"Fe": 8, "O": 8, "P": 3}),
    ],
)
def test_count_elements(count_mode: ElemCountMode, counts: dict[str, float]) -> None:
    series = pmv_pd.count_elements(
        ["Fe2 O3"] * 5 + ["Fe4 P4 O16"] * 3, count_mode=count_mode
    )
    expected = pd.Series(counts, index=pmv.df_ptable.index, name="count")
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_by_atomic_nums() -> None:
    series_in = pd.Series(1, index=range(1, 119))
    el_cts = pmv_pd.count_elements(series_in)
    expected = pd.Series(1, index=pmv.df_ptable.index, name="count")

    pd.testing.assert_series_equal(expected, el_cts)


@pytest.mark.parametrize("range_limits", [(-1, 10), (100, 200)])
def test_count_elements_bad_atomic_nums(range_limits: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="assumed to represent atomic numbers"):
        pmv_pd.count_elements(dict.fromkeys(range(*range_limits), 0))

    with pytest.raises(ValueError, match="assumed to represent atomic numbers"):
        # string and integer keys for atomic numbers should be handled equally
        pmv_pd.count_elements({str(idx): 0 for idx in range(*range_limits)})


@pytest.mark.parametrize(
    ("count_mode", "expected_counts"),
    [
        (ElemCountMode.composition, {"Fe": 22, "O": 63, "P": 12}),
        (ElemCountMode.fractional_composition, {"Fe": 2.5, "O": 5, "P": 0.5}),
        (ElemCountMode.reduced_composition, {"Fe": 13, "O": 27, "P": 3}),
        (ElemCountMode.occurrence, {"Fe": 8, "O": 8, "P": 3}),
    ],
)
def test_count_elements_composition_objects(
    count_mode: ElemCountMode, expected_counts: dict[str, float]
) -> None:
    """Test count_elements with Composition objects and various count modes."""
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = pmv_pd.count_elements(compositions, count_mode=count_mode)
    expected = pd.Series(expected_counts, index=pmv.df_ptable.index, name="count")
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_mixed_input() -> None:
    mixed_input = ["Fe2O3", Composition("Fe4P4O16"), "LiCoO2", Composition("NaCl")]
    series = pmv_pd.count_elements(mixed_input, count_mode=ElemCountMode.composition)  # type: ignore[arg-type]
    expected = pd.Series(
        {"Fe": 6, "O": 21, "P": 4, "Li": 1, "Co": 1, "Na": 1, "Cl": 1},
        index=pmv.df_ptable.index,
        name="count",
    )
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_exclude_elements() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = pmv_pd.count_elements(
        compositions, count_mode=ElemCountMode.composition, exclude_elements=["Fe", "P"]
    )
    expected = pd.Series(
        {"O": 63}, index=pmv.df_ptable.index.drop(["Fe", "P"]), name="count"
    )
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_elements_exclude_invalid_elements() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    exclude_elements = ["Fe", "P", "Zz"]
    with pytest.raises(
        ValueError,
        match=re.escape(f"Unexpected symbol(s) Zz in {exclude_elements=}"),
    ):
        pmv_pd.count_elements(
            compositions,
            count_mode=ElemCountMode.composition,
            exclude_elements=exclude_elements,
        )


def test_count_elements_fill_value() -> None:
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = pmv_pd.count_elements(compositions, count_mode=ElemCountMode.composition)
    expected = pd.Series(
        {"Fe": 22, "O": 63, "P": 12}, index=pmv.df_ptable.index, name="count"
    )
    pd.testing.assert_series_equal(series, expected, check_dtype=False)

    # Test previous default value (fill_value = 0)
    compositions = [Composition("Fe2O3")] * 5 + [Composition("Fe4P4O16")] * 3
    series = pmv_pd.count_elements(
        compositions, count_mode=ElemCountMode.composition, fill_value=0
    )
    expected = pd.Series(
        {"Fe": 22, "O": 63, "P": 12}, index=pmv.df_ptable.index, name="count"
    ).fillna(0)
    pd.testing.assert_series_equal(series, expected, check_dtype=False)


def test_count_formulas_basic() -> None:
    """Test basic functionality with formula strings."""
    data = ["Fe2O3", "Fe4O6", "FeO", "Li2O", "LiFeO2"]
    df_out = pmv_pd.count_formulas(data)

    # Test DataFrame structure
    assert set(df_out.columns) == {"arity_name", "chem_sys", "count"}
    assert len(df_out) == 3  # 2 binary systems (Fe-O, Li-O) and 1 ternary (Li-Fe-O)

    # Test arity counts
    arity_counts = df_out.groupby("arity_name")["count"].sum()
    assert arity_counts["binary"] == 4  # Fe2O3, Fe4O6, FeO, Li2O
    assert arity_counts["ternary"] == 1  # LiFeO2


@pytest.mark.parametrize(
    ("data", "error_match"),
    [
        ([], "Empty input: data sequence is empty"),
        (["Fe2O3", "NotAFormula"], "Invalid formula"),
    ],
)
def test_count_formulas_raises(data: list, error_match: str) -> None:
    """Test count_formulas error handling."""
    with pytest.raises(ValueError, match=error_match):
        pmv_pd.count_formulas(data)


def test_count_formulas_composition_objects() -> None:
    """Test handling of Composition objects."""
    data = [
        Composition("Fe2O3"),
        Composition("Fe4O6"),  # same as Fe2O3 when reduced
        Composition("FeO"),
        Composition("Li2O"),
    ]
    df_out = pmv_pd.count_formulas(data, group_by="reduced_formula")

    # Should have 3 unique reduced formulas: Fe2O3 (2 entries), FeO, Li2O
    assert len(df_out) == 3
    assert df_out["count"].sum() == 4

    # Test that Fe2O3 and Fe4O6 are counted together
    fe_o_counts = df_out[df_out["formula"].str.contains("Fe")]
    assert len(fe_o_counts) == 2  # Fe2O3 and FeO
    assert fe_o_counts[fe_o_counts["formula"] == "Fe2O3"]["count"].iloc[0] == 2


@pytest.mark.parametrize(
    ("group_by", "expected_formulas", "expected_counts"),
    [
        (
            "formula",
            ["Fe2O3", "Fe4O6", "FeO"],  # all formulas kept separate
            [1, 1, 1],
        ),
        (
            "reduced_formula",
            ["Fe2O3", "FeO"],  # Fe4O6 -> Fe2O3
            [2, 1],
        ),
        (
            "chem_sys",
            ["Fe-O"],  # all Fe-O formulas grouped together
            [3],
        ),
    ],
)
def test_count_formulas_grouping_modes(
    group_by: FormulaGroupBy, expected_formulas: list[str], expected_counts: list[int]
) -> None:
    """Test different grouping modes."""
    data = ["Fe2O3", "Fe4O6", "FeO"]
    df_out = pmv_pd.count_formulas(data, group_by=group_by)

    if group_by == "chem_sys":
        assert list(df_out["chem_sys"]) == expected_formulas
    else:
        assert list(df_out["formula"]) == expected_formulas
    assert list(df_out["count"]) == expected_counts


def test_count_formulas_mixed_input() -> None:
    """Test handling of mixed input types."""
    data = [
        "Fe2O3",
        Composition("Fe4O6"),
        "Fe-O",  # chemical system string
        Composition("FeO"),
    ]
    df_out = pmv_pd.count_formulas(data, group_by="chem_sys")

    # All should be grouped into Fe-O system
    assert len(df_out) == 1
    assert df_out["chem_sys"].iloc[0] == "Fe-O"
    assert df_out["count"].iloc[0] == 4


PMG_FORMULA_0 = SI_STRUCTS[0].formula
PMG_FORMULA_1 = SI_STRUCTS[1].formula
PMG_EXPECTED_DICT = {
    f"1 {PMG_FORMULA_0}": SI_STRUCTS[0],
    f"2 {PMG_FORMULA_1}": SI_STRUCTS[1],
}

SI_ISTRUCTURE_0 = IStructure.from_sites(SI_STRUCTS[0])

# Molecule and IMolecule test fixtures
H2O_MOL = Molecule(["H", "H", "O"], [[0, 0, 0], [0, 0, 1.5], [0, 0, 0.75]])
CO2_MOL = Molecule(["C", "O", "O"], [[0, 0, 0], [0, 0, 1.2], [0, 0, -1.2]])
H2O_IMOL = IMolecule(["H", "H", "O"], [[0, 0, 0], [0, 0, 1.5], [0, 0, 0.75]])
CO2_IMOL = IMolecule(["C", "O", "O"], [[0, 0, 0], [0, 0, 1.2], [0, 0, -1.2]])

_test_cases_normalize_structures = [
    # (test_case_name, input_to_normalize_structures, expected_dictionary_output)
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
    # mixed list: Pymatgen Structure and ASE Atoms object
    ("mixed_list_pmg_and_ase", [SI_STRUCTS[0], SI_ATOMS[1]], PMG_EXPECTED_DICT),
    (
        "mixed_dict_pmg_and_ase",
        {"pmg_key": SI_STRUCTS[0], "ase_key": SI_ATOMS[1]},
        {"pmg_key": SI_STRUCTS[0], "ase_key": SI_STRUCTS[1]},
    ),
    # Molecule/IMolecule tests (single, list, dict, mixed)
    ("single_molecule", H2O_MOL, {H2O_MOL.composition.formula: H2O_MOL}),
    ("single_imolecule", H2O_IMOL, {H2O_IMOL.composition.formula: H2O_IMOL}),
    (
        "list_molecules",
        [H2O_MOL, CO2_MOL],
        {
            f"1 {H2O_MOL.composition.formula}": H2O_MOL,
            f"2 {CO2_MOL.composition.formula}": CO2_MOL,
        },
    ),
    (
        "list_imolecules",
        [H2O_IMOL, CO2_IMOL],
        {
            f"1 {H2O_IMOL.composition.formula}": H2O_IMOL,
            f"2 {CO2_IMOL.composition.formula}": CO2_IMOL,
        },
    ),
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
        {
            f"1 {SI_STRUCTS[0].formula}": SI_STRUCTS[0],
            f"2 {H2O_MOL.composition.formula}": H2O_MOL,
        },
    ),
    (
        "mixed_dict_struct_mol",
        {"s": SI_STRUCTS[0], "m": H2O_MOL},
        {"s": SI_STRUCTS[0], "m": H2O_MOL},
    ),
    (
        "mixed_istruct_imol",
        [SI_ISTRUCTURE_0, H2O_IMOL],
        {
            f"1 {SI_ISTRUCTURE_0.formula}": SI_ISTRUCTURE_0,
            f"2 {H2O_IMOL.composition.formula}": H2O_IMOL,
        },
    ),
]


@pytest.mark.parametrize(
    ("test_case_name", "input_raw", "expected_output_dict"),
    _test_cases_normalize_structures,
    ids=[case[0] for case in _test_cases_normalize_structures],
)
def test_normalize_structures(
    test_case_name: str,
    input_raw: Any,
    expected_output_dict: dict[Hashable, Structure | IStructure | Molecule | IMolecule],
) -> None:
    """Test normalize_structures with various inputs including Molecules."""
    del test_case_name

    result_dict = pmv_pd.normalize_structures(input_raw)

    assert result_dict == expected_output_dict


@pytest.mark.parametrize(
    ("invalid_input", "error_match"),
    [
        ("not a structure", "Input must be a pymatgen Structure"),
        (12345, "Input must be a pymatgen Structure"),
        ([SI_STRUCTS[0], "invalid"], "Item must be a pymatgen Structure"),
    ],
)
def test_normalize_structures_errors(invalid_input: Any, error_match: str) -> None:
    """Test error messages include Molecule and IMolecule types."""
    with pytest.raises(TypeError, match=error_match):
        pmv_pd.normalize_structures(invalid_input)


@pytest.mark.parametrize(
    ("series_input", "expected_keys"),
    [
        (pd.Series([SI_STRUCTS[0], SI_STRUCTS[1]], index=["s1", "s2"]), {"s1", "s2"}),
        (pd.Series([H2O_MOL, CO2_MOL], index=["h2o", "co2"]), {"h2o", "co2"}),
        (pd.Series([SI_STRUCTS[0], H2O_MOL], index=["s", "m"]), {"s", "m"}),
    ],
)
def test_normalize_structures_pandas_series(
    series_input: pd.Series, expected_keys: set[str]
) -> None:
    """Test normalize_structures with pandas Series (Structures and Molecules)."""
    result = pmv_pd.normalize_structures(series_input)
    assert set(result.keys()) == expected_keys
    for key in expected_keys:
        assert result[key] == series_input[key]


@pytest.mark.parametrize("empty_input", [[], {}])
def test_normalize_structures_empty(empty_input: list | dict) -> None:
    """Test normalize_structures raises error on empty inputs."""
    with pytest.raises(ValueError, match="Cannot plot empty set of structures"):
        pmv_pd.normalize_structures(empty_input)


# Mock classes for testing is_ase_atoms
class MockAseAtoms:
    """Mock ASE Atoms class for testing."""

    __module__ = "ase.atoms"
    __qualname__ = "Atoms"


class MockMsonAtoms:
    """Mock MSONAtoms class for testing."""

    __module__ = "pymatgen.io.ase"
    __qualname__ = "MSONAtoms"


class NotAse:
    """A mock class that is not an ASE Atoms object."""

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
    """Test the is_ase_atoms function with various inputs."""
    assert pmv_pd.is_ase_atoms(obj) == expected


def test_is_phonopy_atoms() -> None:
    """Test is_phonopy_atoms with real PhonopyAtoms object."""
    pytest.importorskip("phonopy")
    from phonopy.structure.atoms import PhonopyAtoms

    phonopy_atoms = PhonopyAtoms(
        symbols=["Si"], positions=[[0, 0, 0]], cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    assert pmv_pd.is_phonopy_atoms(phonopy_atoms)


@pytest.mark.parametrize(
    "obj", [Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]]), "string", 123, None]
)
def test_is_phonopy_atoms_false(obj: object) -> None:
    """Test is_phonopy_atoms returns False for non-PhonopyAtoms objects."""
    assert pmv_pd.is_phonopy_atoms(obj) is False


class DummyClass:
    def __init__(self, name: str) -> None:
        self.name = name
        self.formula = name  # Add a formula attribute to mimic Structure


@pytest.mark.parametrize("cls", [Structure, DummyClass, Lattice])
def test_normalize_to_dict(cls: type[Structure | DummyClass | Lattice]) -> None:
    # Test with a single instance
    single_instance = {
        Structure: Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]]),
        DummyClass: DummyClass("dummy"),
        Lattice: Lattice.cubic(5),
    }[cls]
    result = pmv_pd.normalize_to_dict(single_instance, cls=cls)
    assert isinstance(result, dict)
    assert len(result) == 1
    assert "" in result
    assert isinstance(result[""], cls)

    # Test with a list of instances
    instance_list = [single_instance, single_instance]
    result = pmv_pd.normalize_to_dict(instance_list, cls=cls)
    assert isinstance(result, dict)
    assert len(result) == 2
    assert all(isinstance(s, cls) for s in result.values())
    expected_keys = {
        Structure: {"Si1", "Si1 1"},
        DummyClass: {"dummy", "dummy 1"},
        Lattice: {"Lattice", "Lattice 1"},
    }[cls]
    assert set(result) == expected_keys

    # Test with a dictionary of instances
    instance_dict = {"item1": single_instance, "item2": single_instance}
    result = pmv_pd.normalize_to_dict(instance_dict, cls=cls)
    assert result == instance_dict

    # Test with invalid input
    cls_name = cls.__name__
    err_msg = f"Invalid inputs, expected {cls_name} or dict/list/tuple of {cls_name}"
    with pytest.raises(TypeError, match=err_msg):
        pmv_pd.normalize_to_dict("invalid input", cls=cls)

    # Test with mixed valid and invalid inputs in a list
    with pytest.raises(TypeError, match=err_msg):
        pmv_pd.normalize_to_dict([single_instance, "invalid"], cls=cls)


@pytest.mark.parametrize(
    ("cls1", "cls2"),
    [
        (Structure, DummyClass),
        (DummyClass, Lattice),
        (Structure, Lattice),
    ],
)
def test_normalize_to_dict_mixed_classes(
    cls1: type[Structure | DummyClass], cls2: type[Structure | DummyClass]
) -> None:
    obj_map = {
        Structure: Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]]),
        DummyClass: DummyClass("dummy1"),
        Lattice: Lattice.cubic(5),
    }
    instance1 = obj_map[cls1]
    instance2 = obj_map[cls2]
    cls_name = cls1.__name__

    with pytest.raises(
        TypeError,
        match=f"Invalid inputs, expected {cls_name} or dict/list/tuple of {cls_name}",
    ):
        pmv_pd.normalize_to_dict([instance1, instance2], cls=cls1)


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
        pmv_pd.df_to_arrays("foo", y_true, y_pred)  # type: ignore[arg-type]

    bad_col_name = "not-real-col-name"
    with pytest.raises(KeyError) as exc:
        pmv_pd.df_to_arrays(df_regr, bad_col_name, df_regr.columns[0])

    assert "not-real-col-name" in str(exc.value)


def test_df_to_arrays_strict() -> None:
    args = pmv_pd.df_to_arrays(42, "foo", "bar", strict=False)  # type: ignore[arg-type]
    assert args == ["foo", "bar"]

    with pytest.raises(TypeError, match="df should be pandas DataFrame or None"):
        pmv_pd.df_to_arrays(42, "foo", "bar", strict=True)  # type: ignore[arg-type]


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

    # If density column is specified, check if it exists
    if density_col:
        assert density_col in df_binned
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
        (pd.Series([1, 225, 167]), [1, 225, 167]),
        (["Fm-3m", "P1", "Pnma"], [225, 1, 62]),  # Hermann-Mauguin symbols -> numbers
    ],
)
def test_normalize_spacegroups(data: list | pd.Series, expected: list) -> None:
    """Test normalize_spacegroups with various input types."""
    assert list(pmv_pd.normalize_spacegroups(data)) == expected


def test_normalize_spacegroups_with_structures() -> None:
    """Test normalize_spacegroups with pymatgen Structure objects."""
    result = pmv_pd.normalize_spacegroups(SI_STRUCTS)
    assert len(result) == len(SI_STRUCTS)
    # Si structures should have space group 227 (Fd-3m) or similar cubic
    assert all(1 <= spg <= 230 for spg in result)


def test_normalize_spacegroups_with_ase_atoms() -> None:
    """Test normalize_spacegroups with ASE Atoms objects."""
    result = pmv_pd.normalize_spacegroups(SI_ATOMS)
    assert len(result) == len(SI_ATOMS)
    assert all(1 <= spg <= 230 for spg in result)


def test_normalize_spacegroups_empty_raises() -> None:
    """Test that empty input raises ValueError."""
    with pytest.raises(ValueError, match="Cannot normalize empty spacegroup data"):
        pmv_pd.normalize_spacegroups([])


@pytest.mark.parametrize("invalid_val", [0, -1, 231, 500])
def test_normalize_spacegroups_invalid_number_raises(invalid_val: int) -> None:
    """Test that out-of-range spacegroup numbers raise ValueError."""
    with pytest.raises(ValueError, match=r"Space group numbers must be in \[1, 230\]"):
        pmv_pd.normalize_spacegroups([1, invalid_val, 225])


def test_sankey_flow_data_returns_expected_keys() -> None:
    """Test sankey_flow_data returns all expected keys."""
    df_test = pd.DataFrame({"A": ["x", "x", "y"], "B": ["p", "q", "p"]})
    result = pmv_pd.sankey_flow_data(df_test, ["A", "B"])
    assert set(result) == {
        "source",
        "target",
        "value",
        "labels",
        "source_indices",
        "target_indices",
    }


@pytest.mark.parametrize(
    ("labels_with_counts", "check_char", "should_contain"),
    [(True, ":", True), ("percent", "%", True), (False, ":", False)],
)
def test_sankey_flow_data_labels(
    labels_with_counts: bool | str, check_char: str, should_contain: bool
) -> None:
    """Test sankey_flow_data label formatting options."""
    df_test = pd.DataFrame({"A": ["x", "x", "y"], "B": ["p", "q", "p"]})
    result = pmv_pd.sankey_flow_data(
        df_test, ["A", "B"], labels_with_counts=labels_with_counts
    )
    has_char = any(check_char in label for label in result["labels"])
    assert has_char == should_contain


def test_sankey_flow_data_invalid_cols_raises() -> None:
    """Test that invalid columns raise ValueError."""
    df_test = pd.DataFrame({"A": ["x"], "B": ["p"]})
    with pytest.raises(ValueError, match="should specify exactly two columns"):
        pmv_pd.sankey_flow_data(df_test, ["A"])
