from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING
from unittest.mock import Mock, call, patch

import pytest
import requests

from pymatviz.enums import Files, Key, LabelEnum


if TYPE_CHECKING:
    from pathlib import Path


# ruff: noqa: RUF001


class DummyEnum(LabelEnum):
    """Test enum with label and description metadata."""

    TEST1 = "test1", "Test Label 1", "Test Description 1"
    TEST2 = "test2", "Test Label 2"


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


def test_label_enum_value_and_repr() -> None:
    """LabelEnum members compare equal to their value and repr as their label."""
    assert DummyEnum.TEST1 == "test1"
    assert repr(DummyEnum.TEST1) == "Test Label 1"
    assert DummyEnum.TEST2.description == ""  # description is optional


@pytest.mark.parametrize(
    ("attr_name", "old_value", "new_value"),
    [
        ("label", "Test Label 1", "New Label"),
        ("description", "Test Description 1", "New Description"),
    ],
)
def test_label_enum_attributes_read_only(
    attr_name: str, old_value: str, new_value: str
) -> None:
    assert getattr(DummyEnum.TEST1, attr_name) == old_value
    with pytest.raises(AttributeError):
        setattr(DummyEnum.TEST1, attr_name, new_value)


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
        # Keys without symbols
        (Key.structure, None),
        (Key.formula, None),
        (Key.composition, None),
        (Key.crystal_system, None),
        (Key.spg_symbol, None),
        (Key.basis_set, None),
        (Key.model_name, None),
        (Key.description, None),
    ],
)
def test_key_symbols(key: Key, expected_symbol: str | None) -> None:
    """Test Key enum symbols."""
    assert key.symbol == expected_symbol


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
        assert key.unit == "K", f"Expected K unit for {key=}, got {key.unit}"

    valid_ev_units = {
        "eV",
        "eV/atom",
        "eV/K",
        "eV/Å",
        "meV",
        "eV/Å<sup>3</sup>",
        "states/eV",
    }
    for key in Key:
        if key.unit and "eV" in key.unit:
            assert key.unit in valid_ev_units, f"Unexpected {key.unit=} for {key=}"


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
    """Sub/superscript tags in symbols and units are balanced and not nested."""
    for key in Key:
        for field in ("symbol", "unit"):
            value = getattr(key, field) or ""
            for tag in ("sub", "sup"):
                if f"<{tag}>" in value:
                    n_open, n_close = value.count(f"<{tag}>"), value.count(f"</{tag}>")
                    assert n_open == n_close, f"Mismatched {tag} tags in {value}"
                    assert f"</{tag}>" not in value[: value.index(f"<{tag}>")]
            if "<sup>" in value and "<sub>" in value:
                sup_start, sup_end = value.index("<sup>"), value.index("</sup>")
                sub_start, sub_end = value.index("<sub>"), value.index("</sub>")
                assert not (sup_start < sub_start < sup_end), f"Nested tags {value}"
                assert not (sub_start < sup_start < sub_end), f"Nested tags {value}"


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


def test_files_enum(tmp_path: Path) -> None:
    """Files members expose paths, URLs, labels and lookups relative to base_dir."""
    # The main Files enum from pymatviz.enums has "" as its _base_dir by default
    assert Files._base_dir == ""

    class TestDataFiles(Files, base_dir=str(tmp_path), auto_download=False):
        # member = "path_relative_to_base_dir", "url", "label", "description"
        mp_energies = (
            "mp/2025-02-01-mp-energies.csv.gz",
            "https://figshare.com/files/789012",
            "MP Energies Test",
            "Test description for MP energies",
        )
        wbm_summary = (
            "wbm/2023-12-13-wbm-summary.csv.gz",
            "https://figshare.com/files/345678",
            "WBM Summary Test",
        )

    mp_file = TestDataFiles.mp_energies
    assert TestDataFiles._base_dir == str(tmp_path)
    assert mp_file.value == mp_file.rel_path == "mp/2025-02-01-mp-energies.csv.gz"
    assert mp_file.file_path == f"{tmp_path}/mp/2025-02-01-mp-energies.csv.gz"
    assert mp_file.url == "https://figshare.com/files/789012"
    assert (mp_file.label, mp_file.description) == (
        "MP Energies Test",
        "Test description for MP energies",
    )
    assert repr(mp_file) == "TestDataFiles.mp_energies"
    assert str(mp_file) == mp_file.name == "mp_energies"
    assert TestDataFiles.wbm_summary.description == ""
    assert TestDataFiles.from_label("WBM Summary Test") is TestDataFiles.wbm_summary

    invalid_label = "this-label-does-not-exist-for-sure"
    with pytest.raises(ValueError, match=f"label='{invalid_label}' not found"):
        Files.from_label(invalid_label)


EXAMPLE_URL = "https://example.com/file.txt"


@pytest.mark.parametrize(
    ("auto_download", "url", "pre_existing", "response_ok", "expect_download"),
    [
        (True, EXAMPLE_URL, False, True, True),
        (True, EXAMPLE_URL, True, True, False),
        (False, EXAMPLE_URL, False, True, False),
        (True, "", False, True, False),
        (True, EXAMPLE_URL, False, False, True),
    ],
    ids=["downloads-missing", "skips-existing", "disabled", "no-url", "http-error"],
)
def test_files_enum_auto_download(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    auto_download: bool,
    url: str,
    pre_existing: bool,
    response_ok: bool,
    expect_download: bool,
) -> None:
    """file_path downloads missing files only when enabled and a URL is set."""

    class TestFiles(Files, base_dir=str(tmp_path), auto_download=auto_download):
        test_file = "data/file.txt", url, "Test File"

    path = f"{tmp_path}/data/file.txt"
    os.makedirs(os.path.dirname(path))
    if pre_existing:
        with open(path, mode="wb") as file:
            file.write(b"old content")

    response = Mock(content=b"new content")
    if not response_ok:
        response.raise_for_status.side_effect = requests.HTTPError("HTTP Error: 404")

    with patch("requests.get", return_value=response) as mock_get:
        if response_ok:
            assert TestFiles.test_file.file_path == path
        else:
            with pytest.raises(requests.HTTPError, match="HTTP Error: 404"):
                _ = TestFiles.test_file.file_path

    assert mock_get.call_args_list == ([call(url)] if expect_download else [])
    stdout = capsys.readouterr().out
    assert ("Downloading" in stdout) is expect_download
    assert (f"Downloading {url} to {path}" in stdout) is expect_download
    expected_content = (
        b"new content"
        if expect_download and response_ok
        else b"old content"
        if pre_existing
        else None
    )
    if expected_content is None:
        assert not os.path.isfile(path)
    else:
        with open(path, mode="rb") as file:
            assert file.read() == expected_content
