from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call, patch
from urllib.error import HTTPError

import pytest

from pymatviz.enums import ElemCountMode, Files, LabelEnum


if TYPE_CHECKING:
    from pathlib import Path


class DummyEnum(LabelEnum):
    """Test enum with label and description metadata."""

    TEST1 = "test1", "Test Label 1", "Test Description 1"
    TEST2 = "test2", "Test Label 2"


def test_pickle_enum() -> None:
    """Enum members unpickle to plain strings, robust to renamed or moved enums."""
    unpickled = pickle.loads(pickle.dumps(ElemCountMode.occurrence))  # noqa: S301
    assert type(unpickled) is str
    assert unpickled == ElemCountMode.occurrence == "occurrence"


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

    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"new content"
    error = None if response_ok else HTTPError(url, 404, "Not Found", None, None)  # ty: ignore[invalid-argument-type]

    with patch(
        "pymatviz.enums.urlopen", return_value=response, side_effect=error
    ) as mock_get:
        if response_ok:
            assert TestFiles.test_file.file_path == path
        else:
            with pytest.raises(HTTPError, match="HTTP Error 404"):
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
