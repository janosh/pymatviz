from __future__ import annotations

import os
import pickle
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import requests

from pymatviz.enums import Files, Key, LabelEnum, StrEnum


if TYPE_CHECKING:
    from pathlib import Path


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


def test_files_enum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error handling and base_dir in Files enum."""

    # The main Files enum from pymatviz.enums has "" as its _base_dir by default
    assert Files._base_dir == ""

    # Test custom base_dir with a local subclass
    class SubFiles(Files, base_dir="custom_foo_dir"):
        # EnumMemberName = "path_relative_to_base_dir", "url", "label", "description"
        sub_file_example = (
            "data/example.txt",
            "http://example.com/data.txt",
            "Example File",
            "An example file for testing",
        )

    assert SubFiles._base_dir == "custom_foo_dir"
    example_member = SubFiles.sub_file_example
    assert example_member.value == "data/example.txt"
    assert example_member.rel_path == "data/example.txt"

    # To test the .file_path property without triggering downloads, ensure
    # _auto_download is False
    monkeypatch.setattr(SubFiles, "_auto_download", False)
    assert example_member.file_path == "custom_foo_dir/data/example.txt"

    # Test __repr__ and __str__ methods using SubFiles
    assert repr(example_member) == "SubFiles.sub_file_example"
    assert str(example_member) == "sub_file_example"

    # Test invalid label lookup for the main Files enum
    # This assumes Files has members with labels.
    invalid_label_to_test = "this-label-does-not-exist-for-sure"
    with pytest.raises(ValueError, match=f"label='{invalid_label_to_test}' not found"):
        Files.from_label(invalid_label_to_test)


def test_data_files_enum(tmp_path: Path) -> None:
    """Test Files enum functionality with a local, populated enum."""

    class TestDataFiles(Files, base_dir=str(tmp_path), auto_download=False):
        mp_energies = (
            "mp/2025-02-01-mp-energies.csv.gz",
            "https://figshare.com/files/123456",  # Example URL
            "MP Energies Test",
            "Test description for MP energies",
        )
        wbm_summary = (
            "wbm/2023-12-13-wbm-summary.csv.gz",
            "https://figshare.com/files/789012",  # Example URL
            "WBM Summary Test",
            "Test description for WBM summary",
        )

    assert repr(TestDataFiles.mp_energies) == "TestDataFiles.mp_energies"
    assert str(TestDataFiles.mp_energies) == "mp_energies"

    assert TestDataFiles.mp_energies.rel_path == "mp/2025-02-01-mp-energies.csv.gz"
    assert (
        TestDataFiles.mp_energies.file_path
        == f"{tmp_path}/mp/2025-02-01-mp-energies.csv.gz"
    )
    assert TestDataFiles.mp_energies.name == "mp_energies"
    assert TestDataFiles.mp_energies.url.startswith("https://figshare.com/files/")

    assert TestDataFiles.wbm_summary.rel_path == "wbm/2023-12-13-wbm-summary.csv.gz"
    assert (
        TestDataFiles.wbm_summary.file_path
        == f"{tmp_path}/wbm/2023-12-13-wbm-summary.csv.gz"
    )
    assert TestDataFiles.wbm_summary.url.startswith("https://figshare.com/files/")


# Define a local enum for URL testing to avoid dependency on global Files state
class LocalTestFilesForUrls(Files):
    file1_figshare_url = (
        "path1/file1.txt",
        "https://figshare.com/files/111111",
        "File 1 Figshare",
    )
    file2_empty_url = ("path2/file2.txt", "", "File 2 Empty URL")
    # Removed file2_bad_url as this test expects all parametrized cases to pass
    # or handle gracefully. Testing for assertion failures with bad URLs
    # would typically be a separate test case using pytest.raises.


@pytest.mark.parametrize("data_file", LocalTestFilesForUrls)
def test_data_files_enum_urls(
    data_file: Files, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that each URL in data-files.yml is a valid Figshare download URL."""

    name, url = data_file.name, data_file.url

    if not url:  # If no URL is defined, it passes this specific Figshare check.
        return

    # check that URL is a figshare download
    assert "figshare.com/files/" in url, (
        f"URL for {name} is not a Figshare download URL: {url}"
    )

    # Mock requests.head to avoid actual network calls
    class MockResponse:
        status_code = 200

    def mock_head(*_args: str, **_kwargs: dict[str, str]) -> MockResponse:
        return MockResponse()

    monkeypatch.setattr(requests, "head", mock_head)

    # check that the URL is valid by sending a head request
    response = requests.head(url, allow_redirects=True, timeout=5)
    assert response.status_code in {200, 403}, f"Invalid URL for {name}: {url}"


def test_files_enum_auto_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Test auto-download behavior in Files class."""

    # This is the file path RELATIVE to base_dir
    test_file_rel_path = "test_data/actual_file.txt"
    test_file_url = "https://example.com/actual_file.txt"
    test_file_label = "My Test File"
    test_file_desc = "A file for testing downloads"

    # Define a base TestFiles class.
    # _auto_download will be monkeypatched per test case or set in subclasses.
    class TestFileEnum(Files, base_dir=str(tmp_path), auto_download=False):
        actual_test_file = (
            test_file_rel_path,
            test_file_url,
            test_file_label,
            test_file_desc,
        )

    file_to_test = TestFileEnum.actual_test_file
    expected_abs_path = f"{tmp_path}/{test_file_rel_path}"

    os.makedirs(os.path.dirname(expected_abs_path), exist_ok=True)

    # Mock successful request
    mock_response_content = b"test content for download"

    class MockSuccessfulResponse:
        status_code = 200
        content = mock_response_content

        def raise_for_status(self) -> None:
            pass

    # Mock failed request
    class MockFailedResponse:
        status_code = 404
        content = b"not found"

        def raise_for_status(self) -> None:
            raise requests.exceptions.HTTPError(f"HTTP Error: {self.status_code}")

    # Mock stdin (though not used by current Files enum)
    class MockStdin:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", MockStdin())

    # Test 1: Auto-download enabled, file does NOT exist. Download should happen.
    if os.path.exists(expected_abs_path):
        os.remove(expected_abs_path)
    monkeypatch.setattr(TestFileEnum, "_auto_download", True)

    with patch("requests.get", return_value=MockSuccessfulResponse()) as mock_get_dl:
        retrieved_path = file_to_test.file_path  # Access property
        assert retrieved_path == expected_abs_path
        mock_get_dl.assert_called_once_with(test_file_url)
        stdout, _ = capsys.readouterr()
        assert f"Downloading {test_file_url} to {expected_abs_path}" in stdout
        assert os.path.isfile(expected_abs_path)
        with open(expected_abs_path, "rb") as f_handle:
            assert f_handle.read() == mock_response_content

    # Test 2: Auto-download enabled, file already exists. No download attempt.
    assert os.path.isfile(expected_abs_path)  # Should exist from Test 1
    monkeypatch.setattr(TestFileEnum, "_auto_download", True)  # Keep True

    with patch("requests.get") as mock_get_exists:
        retrieved_path_exists = file_to_test.file_path
        assert retrieved_path_exists == expected_abs_path
        mock_get_exists.assert_not_called()
        stdout_exists, _ = capsys.readouterr()
        assert "Downloading" not in stdout_exists

    # Test 3: Auto-download disabled, file does NOT exist. No download attempt.
    if os.path.exists(expected_abs_path):
        os.remove(expected_abs_path)
    monkeypatch.setattr(TestFileEnum, "_auto_download", False)

    with patch("requests.get") as mock_get_disabled:
        retrieved_path_disabled = file_to_test.file_path
        assert retrieved_path_disabled == expected_abs_path
        mock_get_disabled.assert_not_called()
        assert not os.path.isfile(expected_abs_path)
        stdout_disabled, _ = capsys.readouterr()
        assert "Downloading" not in stdout_disabled

    # Test 4: Auto-download enabled, but URL is empty. No download.
    class TestFileNoUrl(Files, base_dir=str(tmp_path), auto_download=True):
        no_url_file = "test_data/no_url.txt", "", "No URL File"

    file_no_url = TestFileNoUrl.no_url_file
    path_no_url = f"{tmp_path}/{file_no_url.value}"  # Construct path from enum value
    os.makedirs(os.path.dirname(path_no_url), exist_ok=True)
    if os.path.exists(path_no_url):
        os.remove(path_no_url)

    with patch("requests.get") as mock_get_no_url:
        # _auto_download is True from class definition
        retrieved_path_no_url = file_no_url.file_path
        assert retrieved_path_no_url == path_no_url
        mock_get_no_url.assert_not_called()  # No URL, so no call
        assert not os.path.isfile(path_no_url)
        stdout_no_url, _ = capsys.readouterr()
        assert "Downloading" not in stdout_no_url

    # Test 5: Auto-download enabled, download fails (e.g., 404 error).
    class TestFileFailDownload(Files, base_dir=str(tmp_path), auto_download=True):
        fail_dl_file = (
            "test_data/fail.txt",
            "https://example.com/will_fail.txt",
            "Fail DL",
        )

    file_fail_dl = TestFileFailDownload.fail_dl_file
    path_fail_dl = f"{tmp_path}/{file_fail_dl.value}"  # Construct path from enum value
    url_fail_dl = file_fail_dl.url  # Get URL from enum member
    os.makedirs(os.path.dirname(path_fail_dl), exist_ok=True)
    if os.path.exists(path_fail_dl):
        os.remove(path_fail_dl)

    with patch("requests.get", return_value=MockFailedResponse()) as mock_get_fail:
        # _auto_download is True from class definition
        with pytest.raises(requests.exceptions.HTTPError, match="HTTP Error: 404"):
            _ = file_fail_dl.file_path  # This will trigger download attempt & fail
        mock_get_fail.assert_called_once_with(url_fail_dl)
        assert not os.path.isfile(path_fail_dl)
        stdout_fail, _ = capsys.readouterr()
        # The print message includes the URL and the full path
        assert f"Downloading {url_fail_dl} to {path_fail_dl}" in stdout_fail
