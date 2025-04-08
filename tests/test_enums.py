from __future__ import annotations

import pickle
import sys

import pytest

from pymatviz.enums import Key, LabelEnum, StrEnum


# ruff: noqa: RUF001


class DummyEnum(LabelEnum):
    TEST1 = "test1", "Test Label 1", "Test Description 1"
    TEST2 = "test2", "Test Label 2"


def test_str_enum() -> None:
    # ensure all pymatviz Enums classes are subclasses of StrEnum
    # either the standard library StrEnum if 3.11+ or our own StrEnum
    assert issubclass(StrEnum, str)
    if sys.version_info >= (3, 11):
        from enum import StrEnum as StdLibStrEnum

        assert StrEnum is StdLibStrEnum
    else:
        assert issubclass(StrEnum, str)
        assert StrEnum.__name__ == "StrEnum"


def test_key_enum() -> None:
    # access any attribute to trigger @unique decorator check
    assert Key.energy_per_atom == "energy_per_atom"
    assert Key.volume == "volume"


def test_pickle_enum() -> None:
    key = Key.energy_per_atom
    pickled_key = pickle.dumps(key)
    unpickled_key = pickle.loads(pickled_key)  # noqa: S301

    # ensure key unpickles to str, not Key (don't use isinstance check as
    # isinstance(StrEnum, str) is True)
    assert type(unpickled_key) is str
    assert unpickled_key == "energy_per_atom"
    assert unpickled_key == Key.energy_per_atom
    assert type(key) is Key

    assert Key.energy.__reduce_ex__(1) == (str, ("energy",))


def test_label_enum_new() -> None:
    assert DummyEnum.TEST1 == "test1"
    assert DummyEnum.TEST1.label == "Test Label 1"
    assert DummyEnum.TEST1.description == "Test Description 1"

    # check label and description are not settable
    assert DummyEnum.TEST1.label == "Test Label 1"
    with pytest.raises(AttributeError):
        # ignore mypy error from label attribute being read-only
        DummyEnum.TEST1.label = "New Label"  # type: ignore[misc]

    assert DummyEnum.TEST1.description == "Test Description 1"
    with pytest.raises(AttributeError):
        DummyEnum.TEST1.description = "New Description"  # type: ignore[misc]


def test_label_enum_repr() -> None:
    assert repr(DummyEnum.TEST1) == DummyEnum.TEST1.label == "Test Label 1"


@pytest.mark.parametrize(
    ("key", "expected_symbol"),
    [
        # Thermodynamic
        (Key.energy, "E"),
        (Key.enthalpy, "H"),
        (Key.entropy, "S"),
        (Key.free_energy, "F"),
        (Key.gibbs_free_energy, "G"),
        (Key.helmholtz_free_energy, "A"),
        (Key.internal_energy, "U"),
        (Key.heat_of_formation, "E<sub>form</sub>"),
        (Key.heat_of_reaction, "E<sub>rxn</sub>"),
        (Key.lattice_energy, "E<sub>lattice</sub>"),
        # Structural
        (Key.n_sites, "N<sub>sites</sub>"),
        (Key.volume, "V"),
        (Key.density, "ρ"),
        (Key.pressure, "P"),
        (Key.n_atoms, "N<sub>atoms</sub>"),
        (Key.n_elements, "N<sub>elements</sub>"),
        (Key.coord_num, "N<sub>coord</sub>"),
        (Key.max_pair_dist, "d<sub>max</sub>"),
        # Electronic
        (Key.bandgap, "E<sub>gap</sub>"),
        (Key.fermi_energy, "E<sub>Fermi</sub>"),
        (Key.work_function, "E<sub>work</sub>"),
        (Key.electron_affinity, "E<sub>aff</sub>"),
        (Key.effective_mass, "m<sub>eff</sub>"),
        (Key.carrier_concentration, "n"),
        (Key.mobility, "μ"),
        (Key.polarizability, "α"),
        (Key.polarization, "P"),
        # Mechanical
        (Key.bulk_modulus, "K"),
        (Key.shear_modulus, "G"),
        (Key.young_modulus, "Y"),
        (Key.poisson_ratio, "ν"),
        (Key.forces, "F"),
        (Key.stress, "σ"),
        (Key.virial, "V"),
        # Thermal & Magnetic
        (Key.melting_point, "T<sub>melt</sub>"),
        (Key.boiling_point, "T<sub>boil</sub>"),
        (Key.critical_temp, "T<sub>crit</sub>"),
        (Key.critical_pressure, "P<sub>crit</sub>"),
        (Key.critical_vol, "V<sub>crit</sub>"),
        (Key.magnetic_moment, "μ<sub>B</sub>"),
        (Key.debye_temp, "T<sub>D</sub>"),
        # Phonon
        (Key.last_ph_dos_peak, "ω<sub>max</sub>"),
        (Key.max_ph_freq, "Ω<sub>max</sub>"),
        (Key.min_ph_freq, "Ω<sub>min</sub>"),
        # Metrics
        (Key.accuracy, "Acc"),
        (Key.auc, "AUC"),
        (Key.mae, "MAE"),
        (Key.r2, "R²"),
        (Key.pearson, "r<sub>P</sub>"),
        (Key.spearman, "r<sub>S</sub>"),
        (Key.kendall, "r<sub>K</sub>"),
        (Key.mse, "MSE"),
        (Key.rmse, "RMSE"),
        (Key.f1, "F1"),
        (Key.precision, "P"),
        (Key.recall, "R"),
        (Key.fpr, "FPR"),
        (Key.tpr, "TPR"),
        (Key.fnr, "FNR"),
        (Key.tnr, "TNR"),
        (Key.tp, "TP"),
        (Key.fp, "FP"),
        (Key.tn, "TN"),
        (Key.fn, "FN"),
        (Key.n_structures, "N<sub>structs</sub>"),
        (Key.n_materials, "N<sub>materials</sub>"),
    ],
)
def test_key_symbols(key: Key, expected_symbol: str) -> None:
    """Test Key enum symbols."""
    assert key.symbol == expected_symbol


def test_key_symbol_none() -> None:
    """Test that keys without symbols return None."""
    keys_without_symbols = [
        Key.structure,
        Key.formula,
        Key.composition,
        Key.crystal_system,
        Key.spg_symbol,
        Key.basis_set,
        Key.model_name,
        Key.description,
    ]
    for key in keys_without_symbols:
        assert key.symbol is None, f"Expected {key} to have no symbol"


def test_key_symbol_unit_category_desc() -> None:
    """Test that all keys have a valid symbol, unit, category, and description."""
    for key in Key:
        symbol = key.symbol
        assert isinstance(symbol, str | type(None)), f"Invalid {symbol=} for {key=}"
        if isinstance(symbol, str):
            assert symbol.strip(), f"Empty or whitespace symbol for {key}"

        unit = key.unit
        assert isinstance(unit, str | type(None)), f"Invalid {unit=} for {key=}"

        category = key.category
        assert isinstance(category, str), f"Invalid {category=} for {key=}"

        desc = key.desc
        assert isinstance(desc, str | type(None)), f"Invalid {desc=} for {key=}"


def test_keys_yaml_and_enum_are_in_sync() -> None:
    """Test that all keys in keys.yml are defined in Key enum and vice versa."""
    from pymatviz.enums import _keys

    # Check for differences
    only_in_yaml = set(map(str, _keys)) - set(map(str, Key)) - {"yield", "material_id"}
    only_in_enum = set(map(str, Key)) - set(map(str, _keys)) - {"yield_", "mat_id"}

    assert only_in_enum == set(), f"keys in enum but not in YAML: {only_in_enum}"
    assert only_in_yaml == set(), f"keys in YAML but not in enum: {only_in_yaml}"


def test_key_categories_are_valid() -> None:
    """Test that all keys have valid categories that match the class structure."""
    valid_categories = [
        "structural",
        "electronic",
        "thermodynamic",
        "mechanical",
        "thermal",
        "magnetic",
        "phonon",
        "optical",
        "surface",
        "defect",
        "crystal_symmetry_properties",
        "dft",
        "ml",
        "metrics",
        "computational_details",
        "identifiers_and_metadata",
        "code",
        "synthesis_related",
        "performance_indicators",
        "environmental_indicators",
        "composition",
        "chemical",
        "structure_prototyping",
        "economic",
        "molecular_dynamics",
    ]

    for key in Key:
        category = key.category
        assert category in valid_categories, f"bad {category=} for {key=}"


def test_key_units_are_consistent() -> None:
    """Test that keys with the same physical quantity use consistent units."""
    energy_keys = [
        Key.energy,
        Key.energy_per_atom,
        Key.heat_of_formation,
        Key.activation_energy,
        Key.bandgap,
    ]
    for key in energy_keys:
        assert key.unit in ("eV", "eV/atom"), f"Unexpected {key.unit=}"

    temperature_keys = [
        Key.temperature,
        Key.melting_point,
        Key.critical_temp,
        Key.debye_temp,
        Key.curie_temperature,
    ]
    for key in temperature_keys:
        assert key.unit == "K", f"Expected K unit for {key}, got {key.unit}"


def test_key_label_formatting() -> None:
    """Test that all labels are properly formatted."""
    for key in Key:
        label = key.label
        assert (
            label[0].isupper()
            or label[0].isdigit()
            or label.startswith(("r2SCAN", "q-Point", "k-Point"))
        ), f"{label=} should be capitalized"

        assert not label.endswith((".", ",")), f"{label=} ends with punctuation"

        assert label.strip() == label, f"{label=} has outer whitespace"

        assert "  " not in label, f"{label=} has multiple spaces"


def test_key_value_matches_name() -> None:
    """Test that all Key enum values match their names."""
    for key in Key:
        # yield is a reserved word in Python, so had to be suffixed, hence can't match
        # key name can't match value
        if key in (Key.yield_, Key.mat_id):
            continue
        name, value = key.name, str(key)
        assert name == value, f"{name=} doesn't match {value=}"


def test_key_html_validity() -> None:
    """Test that HTML subscripts in symbols are properly formatted."""
    for key in Key:
        for field in ("symbol", "unit"):
            value = getattr(key, field)
            if value and "<sub>" in value:
                # Check matching closing tags
                n_sub, n_sub_end = value.count("<sub>"), value.count("</sub>")
                assert n_sub == n_sub_end, f"Mismatched sub tags in {value}"
                # Check proper nesting
                assert "</sub>" not in value[: value.index("<sub>")]


@pytest.mark.parametrize(
    ("key", "expected_description"),
    [
        (Key.step, "Step of a job/task/optimizer."),
        (Key.state, "State of a job/task/computation."),
        (Key.frame_id, "Molecular dynamics or geometry optimization frame."),
        # Keys that should have no description
        (Key.energy, None),
        (Key.volume, None),
        (Key.bandgap, None),
    ],
)
def test_key_descriptions(key: Key, expected_description: str | None) -> None:
    """Test that Key enum descriptions match the YAML data."""
    assert key.desc == expected_description, f"Unexpected description for {key}"


@pytest.mark.parametrize(
    ("key", "expected_unit"),
    [
        # Basic units
        (Key.volume, "Å<sup>3</sup>"),
        (Key.energy, "eV"),
        (Key.temperature, "K"),
        (Key.pressure, "Pa"),
        # Complex units
        (Key.carrier_concentration, "cm<sup>-3</sup>"),
        (Key.mobility, "cm<sup>2</sup>/(V⋅s)"),
        (Key.thermal_conductivity, "W/(m⋅K)"),
        (Key.fracture_toughness, "MPa⋅m<sup>1/2</sup>"),
        # Units with subscripts
        (Key.magnetic_moment, "μ<sub>B</sub>"),
        (Key.energy_per_atom, "eV/atom"),
        (Key.heat_capacity, "J/K"),
        # Mixed sub/superscripts
        (Key.specific_heat_capacity, "J/kg⋅K"),
        (Key.thermal_resistivity, "K⋅m/W"),
        # No units
        (Key.formula, None),
        (Key.structure, None),
        (Key.crystal_system, None),
    ],
)
def test_key_units(key: Key, expected_unit: str | None) -> None:
    """Test that Key enum units are correctly formatted with HTML tags."""
    assert key.unit == expected_unit


def test_unit_html_consistency() -> None:
    """Test that all units follow consistent HTML formatting rules."""
    for key in Key:
        unit = key.unit
        if unit is None:
            continue

        # Check proper HTML tag nesting
        if "<sup>" in unit:
            assert unit.count("<sup>") == unit.count("</sup>"), (
                f"Mismatched sup tags in {unit}"
            )
            assert "</sup>" not in unit[: unit.index("<sup>")], (
                f"Improper sup tag nesting in {unit}"
            )

        if "<sub>" in unit:
            assert unit.count("<sub>") == unit.count("</sub>"), (
                f"Mismatched sub tags in {unit}"
            )
            assert "</sub>" not in unit[: unit.index("<sub>")], (
                f"Improper sub tag nesting in {unit}"
            )

        # Check no nested tags
        if "<sup>" in unit and "<sub>" in unit:
            sup_start = unit.index("<sup>")
            sup_end = unit.index("</sup>")
            sub_start = unit.index("<sub>")
            sub_end = unit.index("</sub>")
            assert not (sup_start < sub_start < sup_end), "Nested sup/sub tags"
            assert not (sub_start < sup_start < sub_end), "Nested sub/sup tags"


def test_unit_special_characters() -> None:
    """Test that special characters in units are consistently used."""
    for key in Key:
        unit = key.unit
        if unit is None:
            continue

        # Check for proper middle dot usage
        assert "·" not in unit, f"ASCII middle dot in {unit} of {key}, use ⋅ instead"

        # Check for proper minus sign usage
        if "-" in unit:
            assert "<sup>-" in unit, (
                f"ASCII hyphen in {unit}, use <sup>-...</sup> for exponents"
            )

        # Common units should use standard symbols
        if "angstrom" in unit:
            assert "Å" in unit, f"Use Å symbol in {unit} of {key}"
        if "micro" in unit:
            assert "μ" in unit, f"Use μ symbol in {unit} of {key}"
        if "ohm" in unit:
            assert "Ω" in unit, f"Use Ω symbol in {unit} of {key}"
        if "kelvin" in unit:
            assert "K" in unit, f"Use K symbol in {unit} of {key}"
        if "pascal" in unit:
            assert "Pa" in unit, f"Use Pa symbol in {unit} of {key}"


def test_unit_formatting_consistency() -> None:
    """Test that similar quantities use consistent unit formatting."""
    ev_keys = {k for k in Key if k.unit and "eV" in k.unit}
    valid_units = {
        "eV",
        "eV/atom",
        "eV/K",
        "eV/Å",
        "meV",
        "eV/Å<sup>3</sup>",
        "states/eV",
    }
    for key in ev_keys:
        unit = key.unit
        assert unit in valid_units, f"Unexpected {unit=} for {key=}"
