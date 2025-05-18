"""Unit tests for pymatviz.treemap.py_pkg.py"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Final
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.express as px
import pytest

import pymatviz as pmv


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from pymatviz.treemap.py_pkg import ShowCounts


# Dummy package structure for testing
# src/
# ├── my_pkg
# │   ├── __init__.py       (0 lines)
# │   ├── module1.py        (4 lines)
# │   └── submodule
# │       ├── __init__.py   (0 lines)
# │       ├── module2.py    (9 lines)
# │       └── module3.py    (1 line)
# └── another_pkg
#     ├── __init__.py       (0 lines)
#     └── main.py           (3 lines)
@pytest.fixture(scope="module")
def dummy_pkg_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Creates a dummy package structure in a temporary directory."""
    src_dir = tmp_path_factory.mktemp("src")

    # my_pkg structure
    my_pkg_dir = src_dir / "my_pkg"
    my_pkg_dir.mkdir()
    (my_pkg_dir / "__init__.py").touch()
    (my_pkg_dir / "module1.py").write_text(
        """# comment
import os

def func1():
    pass # another comment

x = 1
y = 2"""
    )

    submodule_dir = my_pkg_dir / "submodule"
    submodule_dir.mkdir()
    (submodule_dir / "__init__.py").touch()
    (submodule_dir / "module2.py").write_text(
        """# comment1
# comment2

import sys

class MyClass:
    def __init__(self):
        self.a = 1

    def method(self):
        print("hello")
        return True

z = MyClass()
z.method()
print(sys.path)"""
    )
    (submodule_dir / "module3.py").write_text(
        """# Just a comment
pass"""
    )
    (submodule_dir / "module4_typed.py").write_text(
        """from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd # type: ignore
    from collections import abc # type: ignore

def func_typed():
    pass
"""
    )

    # another_pkg structure
    another_pkg_dir = src_dir / "another_pkg"
    another_pkg_dir.mkdir()
    (another_pkg_dir / "__init__.py").touch()
    (another_pkg_dir / "main.py").write_text(
        """# Main module
print("Running main")

VAR = 100"""
    )

    return src_dir


@pytest.fixture(autouse=True)
def _add_dummy_to_path(dummy_pkg_path: Path) -> Generator[None, None, None]:
    """Temporarily add the dummy package source directory to sys.path."""
    sys.path.insert(0, str(dummy_pkg_path))
    yield
    sys.path.pop(0)


@pytest.mark.parametrize(
    ("file_content", "expected_lines"),
    [
        ("", 0),
        ("\n\n", 0),
        ("# comment\n\n# another comment", 0),
        ("import os\nx = 1", 2),
        ("import os\n# comment\nx = 1\n\ny=2 # inline comment", 3),
    ],
)
def test_count_lines(tmp_path: Path, file_content: str, expected_lines: int) -> None:
    """Test count_lines function with various file contents."""
    test_file = tmp_path / "test.py"
    test_file.write_text(file_content)
    assert pmv.treemap.py_pkg.count_lines(str(test_file)) == expected_lines


def test_count_lines_non_existent(tmp_path: Path) -> None:
    """Test count_lines with a non-existent file."""
    assert pmv.treemap.py_pkg.count_lines(str(tmp_path / "non_existent.py")) == 0


def test_default_module_formatter() -> None:
    """Test the default module formatter."""
    assert (
        pmv.treemap.py_pkg.default_module_formatter("my_module", 1234, 5000)
        == "my_module (1,234 lines, 24.7%)"
    )
    assert (
        pmv.treemap.py_pkg.default_module_formatter("another.mod", 50, 100)
        == "another.mod (50 lines, 50.0%)"
    )
    # Test zero total
    assert pmv.treemap.py_pkg.default_module_formatter("zero", 0, 0) == "zero (0 lines)"


@pytest.mark.parametrize(
    ("package_name", "expected_end"),
    [
        ("my_pkg", "src/my_pkg"),
        ("another_pkg", "src/another_pkg"),
        pytest.param(
            "pytest", "site-packages/pytest", marks=pytest.mark.slow
        ),  # Installed pkg
        ("non_existent_pkg", ""),  # Should not find
    ],
)
def test_find_package_path(
    dummy_pkg_path: Path, package_name: str, expected_end: str
) -> None:
    """Test find_package_path for local dummy and non-existent packages.
    Skip checking installed packages unless running with --runslow.
    """
    # For local dummy packages, check absolute path relative to dummy_pkg_path
    # For installed packages, just check if expected_end is in the path
    is_dummy = package_name in ("my_pkg", "another_pkg")

    # Don't need to change CWD if we check absolute paths or use importlib
    # original_cwd = os.getcwd()
    # os.chdir(dummy_pkg_path.parent)
    # try:
    found_path = pmv.treemap.py_pkg.find_package_path(package_name)

    if expected_end:
        # Check absolute path for dummy packages
        if is_dummy:
            # Construct expected absolute path end
            expected_abs_path = str(dummy_pkg_path / package_name).replace("\\", "/")
            assert found_path.replace("\\", "/").endswith(expected_abs_path)
        # Check substring for installed packages (like site-packages)
        elif "site-packages" in expected_end:
            assert expected_end in found_path.replace("\\", "/")
        else:  # Fallback for unexpected cases, keep original relative logic?
            rel_path = os.path.relpath(found_path, start=dummy_pkg_path.parent)
            assert rel_path.replace("\\", "/") == expected_end
    else:
        assert found_path == ""
    # finally:
    # os.chdir(original_cwd)


def test_collect_package_modules_not_found() -> None:
    """Test collect_package_modules with a package that cannot be found."""
    df_modules = pmv.treemap.py_pkg.collect_package_modules(
        ["non_existent_package_12345"]
    )
    assert df_modules.empty


def test_find_package_path_fallbacks(dummy_pkg_path: Path) -> None:
    """Test fallback mechanisms in find_package_path using mocking."""
    target_pkg = "my_pkg"
    # find_package_path returns the absolute path
    expected_abs_path = str(dummy_pkg_path / target_pkg).replace("\\", "/")

    # 1. Test fallback to importlib.util when importlib.resources fails
    with (
        patch("importlib.resources.files", side_effect=ImportError, create=True),
        patch("importlib.util.find_spec") as mock_find_spec,
    ):
        # Mock find_spec to return a spec pointing to the dummy __init__.py
        mock_spec = MagicMock()
        mock_spec.origin = str(dummy_pkg_path / target_pkg / "__init__.py")
        mock_find_spec.return_value = mock_spec
        found_path = pmv.treemap.py_pkg.find_package_path(target_pkg)
        assert found_path.replace("\\", "/") == expected_abs_path

    # 2. Test fallback when find_spec returns spec for a single module (not __init__.py)
    with (
        patch("importlib.resources.files", side_effect=ImportError, create=True),
        patch("importlib.util.find_spec") as mock_find_spec,
    ):
        mock_spec = MagicMock()
        # Point origin to a module file instead of __init__.py
        mock_spec.origin = str(dummy_pkg_path / target_pkg / "module1.py")
        mock_find_spec.return_value = mock_spec
        found_path = pmv.treemap.py_pkg.find_package_path(target_pkg)
        assert found_path.replace("\\", "/") == expected_abs_path

    # 3. Test fallback to site-packages check (mocking site)
    with (
        patch("importlib.resources.files", side_effect=ImportError, create=True),
        patch("importlib.util.find_spec", return_value=None),
        patch("site.getsitepackages", return_value=[str(dummy_pkg_path.parent)]),
        patch("os.path.isdir") as mock_isdir,
    ):
        expected_site_path = str(dummy_pkg_path.parent / target_pkg)
        # Paths find_package_path checks before site-packages
        cwd_path = f"{os.path.abspath(os.getcwd())}/{target_pkg}"
        src_path = f"{os.path.abspath(os.getcwd())}/src/{target_pkg}"

        # Mock isdir returns False for initial checks, True only for expected site path
        mock_isdir.side_effect = lambda p: p == expected_site_path

        found_path = pmv.treemap.py_pkg.find_package_path(target_pkg)
        assert found_path.replace("\\", "/") == expected_site_path.replace("\\", "/")
        # Check that isdir was called with the expected paths
        mock_isdir.assert_any_call(cwd_path)
        mock_isdir.assert_any_call(src_path)
        mock_isdir.assert_any_call(expected_site_path)
        # Verify it was called exactly 3 times in this flow
        assert mock_isdir.call_count == 3


def test_py_pkg_treemap_cell_text_fn() -> None:
    """Test the cell_text_fn parameter."""
    # Default (True) - uses default_module_formatter
    fig_true = pmv.py_pkg_treemap(["my_pkg", "another_pkg"])
    labels_true = fig_true.data[0].labels
    assert any("my_pkg (" in lbl for lbl in labels_true)
    assert any("another_pkg (" in lbl for lbl in labels_true)
    assert any(" lines" in lbl for lbl in labels_true)
    assert any("%)" in lbl for lbl in labels_true)

    # False - no counts on package labels
    fig_false = pmv.py_pkg_treemap(["my_pkg", "another_pkg"], cell_text_fn=False)
    labels_false = fig_false.data[0].labels
    assert "my_pkg" in labels_false
    assert "another_pkg" in labels_false
    assert not any(
        "(" in lbl for lbl in labels_false if lbl in ("my_pkg", "another_pkg")
    )

    # Custom formatter
    def custom_mod_fmt(pkg: str, count: int, total: int) -> str:
        return f"PKG: {pkg} [{count}/{total}]"

    fig_custom = pmv.py_pkg_treemap(
        ["my_pkg", "another_pkg"], cell_text_fn=custom_mod_fmt
    )
    labels_custom = fig_custom.data[0].labels
    assert any(lbl.startswith("PKG: my_pkg [") for lbl in labels_custom)
    assert any(lbl.startswith("PKG: another_pkg [") for lbl in labels_custom)


def test_py_pkg_treemap_empty() -> None:
    """Test treemap generation when no modules are found (e.g., high min_lines)."""
    with pytest.raises(
        ValueError,
        match="No Python modules found .* after filtering by cell_size_fn",
    ):
        pmv.py_pkg_treemap("my_pkg", cell_size_fn=lambda _cell: 0)


def test_py_pkg_treemap_invalid_show_counts() -> None:
    """Test ValueError is raised for invalid show_counts value."""
    with pytest.raises(ValueError, match="Invalid show_counts='invalid_value'"):
        pmv.py_pkg_treemap("my_pkg", show_counts="invalid_value")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("base_url", "expect_link", "expected_texttemplate", "expected_textinfo"),
    [
        (
            "https://github.com/test/repo/blob/main",
            True,
            "<a href='%{customdata[3]}' target='_blank'>%{customdata[2]}</a><br>"
            "%{value:,}<br>%{percentEntry}",
            "none",
        ),
        (
            None,
            False,
            "%{label}<br>%{value:,}<br>%{percentEntry}",
            "label+value+percent entry",
        ),
    ],
    ids=["with_base_url", "without_base_url"],
)
def test_py_pkg_treemap_base_url(
    base_url: str | None,
    expect_link: bool,
    expected_texttemplate: str | None,
    expected_textinfo: str,
) -> None:
    """Test base_url handling for links and hover info."""
    pkg_name = "my_pkg"
    show_counts: Final = "value+percent"

    fig = pmv.py_pkg_treemap(pkg_name, base_url=base_url, show_counts=show_counts)
    trace = fig.data[0]

    # Check texttemplate and textinfo
    assert trace.texttemplate == expected_texttemplate
    assert trace.textinfo == expected_textinfo

    # Check customdata structure and content based on base_url
    custom_data = trace.customdata
    assert custom_data is not None
    assert isinstance(custom_data, np.ndarray)
    assert custom_data.ndim == 2
    expected_cols = 12
    assert custom_data.shape[1] == expected_cols, (
        f"Expected {expected_cols} columns in customdata, got {custom_data.shape[1]}"
    )

    file_urls = custom_data[:, 3]  # Index 3 is file_url
    if expect_link:
        assert base_url is not None  # Ensure base_url is not None for type checking
        # Check file_url (index 3) contains the base_url
        ids_with_url = trace.ids  # Get IDs to correlate
        file_node_indices = [
            i for i, id_val in enumerate(ids_with_url) if ".py" in id_val
        ]
        assert all(
            base_url in str(file_urls[i])
            for i in file_node_indices
            if file_urls[i] is not None
        )
        # Check repo_path_segment (index 1) looks like relative paths
        assert all(".py" in str(custom_data[i, 1]) for i in file_node_indices)
        # Check leaf_label (index 2) looks like leaf labels
        assert all("/" not in str(custom_data[i, 2]) for i in file_node_indices)
    else:
        # Check file_url is None (represented as 'None' string in object array)
        assert np.all(file_urls == "None")

    # Check hovertemplate generation (always generated)
    hovertemplate = trace.hovertemplate
    assert hovertemplate is not None
    assert len(hovertemplate) > 0
    assert "<b>%{customdata[2]}</b>" in hovertemplate  # leaf_label
    assert "Path: %{customdata[1]}<br>" in hovertemplate  # repo_path_segment
    assert (
        "Lines: %{customdata[11]:,}<br>" in hovertemplate
    )  # Correct: actual line_count from customdata
    assert (
        "Cell Value: %{value:,}<br>" in hovertemplate
    )  # Check for Cell Value which is the sizing metric
    assert "Functions: %{customdata[5]:,}<br>" in hovertemplate  # n_functions
    assert "Methods: %{customdata[9]:,}<br>" in hovertemplate  # n_methods
    assert (
        "Type Imports: %{customdata[8]:,}<br>" in hovertemplate
    )  # n_type_checking_imports


def test_py_pkg_treemap_analysis_failure() -> None:
    """Test hover text when _analyze_py_file returns None due to an error."""
    pkg_name = "my_pkg"

    # Mock _analyze_py_file to simulate failure
    failed_analysis_result = {
        "n_classes": 0,
        "n_functions": 0,
        "n_methods": 0,
        "n_internal_imports": 0,
        "n_external_imports": 0,
        "n_type_checking_imports": 0,
    }
    with patch(
        "pymatviz.treemap.py_pkg._analyze_py_file", return_value=failed_analysis_result
    ) as mock_analyze:
        fig = pmv.py_pkg_treemap(pkg_name)

        # Check that analyze was called
        assert mock_analyze.call_count > 0

        # Check hovertemplate exists (it should, just won't show analysis data)
        hovertemplate = fig.data[0].hovertemplate
        assert hovertemplate is not None

        # Check customdata reflects the failed analysis (NaNs filled with 0)
        custom_data = fig.data[0].customdata
        assert custom_data is not None
        # Indices for analysis results: 4 (classes) to 8 (type imports)
        analysis_data = custom_data[:, 4:9]
        # Assert that these values are 0 (since NaNs are filled with 0)
        assert np.all(analysis_data.astype(float) == 0)


# Add slow marker based on fixture request name
def pytest_configure(config: pytest.Config) -> None:
    """Register slow marker."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Add skip marker to slow tests if --runslow not given."""
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def test_internal_imports(dummy_pkg_path: Path) -> None:
    """Test counting of internal imports using AST analysis."""
    pkg_name = "my_pkg"
    # Add a file with internal imports
    # Ensure no leading spaces on lines that matter for parsing
    internal_import_content = (
        "from .submodule import module2 # internal import from sibling dir\n"
        "from . import module1 # internal import from same dir\n"
        "import os # external import\n"
        "\n"
        "class InternalUser:\n"
        "    pass\n"
    )
    (dummy_pkg_path / pkg_name / "internal_user.py").write_text(internal_import_content)

    fig = pmv.py_pkg_treemap(pkg_name)
    hovertemplate = fig.data[0].hovertemplate
    assert hovertemplate is not None

    # Check the template for expected analysis counts
    # NOTE: We check the template string itself, not the rendered text
    # Actual counts for internal_user.py: 1 class, 0 func, 2 internal, 1 external
    assert "<b>%{customdata[2]}</b><br>" in hovertemplate  # Check leaf label marker
    assert "Internal Imports: %{customdata[6]:,}<br>" in hovertemplate
    assert "External Imports: %{customdata[7]:,}<br>" in hovertemplate
    assert "Classes: %{customdata[4]:,}<br>" in hovertemplate
    assert "Functions: %{customdata[5]:,}<br>" in hovertemplate
    assert "Methods: %{customdata[9]:,}<br>" in hovertemplate

    # Optional: Verify the actual data in customdata for the specific node
    ids = fig.data[0].ids
    custom_data = fig.data[0].customdata
    internal_user_idx = -1
    for idx, id_val in enumerate(ids):
        if id_val.endswith("/internal_user"):
            internal_user_idx = idx
            break
    assert internal_user_idx != -1, "internal_user node not found in ids"

    # Cast customdata values to int before comparison
    assert int(custom_data[internal_user_idx, 7]) == 2  # n_internal_imports
    assert int(custom_data[internal_user_idx, 8]) == 1  # n_external_imports
    assert int(custom_data[internal_user_idx, 4]) == 1  # n_classes
    assert int(custom_data[internal_user_idx, 5]) == 0  # n_functions
    assert (
        int(custom_data[internal_user_idx, 6]) == 0
    )  # n_methods for InternalUser class


def test_top_level_module(dummy_pkg_path: Path) -> None:
    """Test handling of Python files directly under the package root."""
    pkg_name = "my_pkg"
    # Add a file directly in the package root
    (dummy_pkg_path / pkg_name / "top_level_mod.py").write_text("print('top level')")

    fig = pmv.py_pkg_treemap(pkg_name, group_by="module")  # Use module grouping

    # Check if top_level_mod is present in labels or ids
    # Its label should be just 'top_level_mod' as it has no parent module part
    assert "top_level_mod" in fig.data[0].labels
    assert any(id_val.endswith("/top_level_mod") for id_val in fig.data[0].ids)

    # Check hovertemplate contains the placeholder for the leaf label
    hovertemplate = fig.data[0].hovertemplate
    assert "<b>%{customdata[2]}</b>" in hovertemplate  # Checks for leaf label marker

    # Customdata is generated, but file_url (index 3) should be None
    custom_data = fig.data[0].customdata
    assert custom_data is not None
    # Check for string 'None' as None gets coerced in object array
    assert np.all(custom_data[:, 3] == "None")


def test_analyze_unicode_error(
    dummy_pkg_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test graceful handling of UnicodeDecodeError during file analysis."""
    pkg_name = "my_pkg"
    bad_file_path = dummy_pkg_path / pkg_name / "bad_encoding.py"
    # Write bytes that are invalid UTF-8
    bad_file_path.write_bytes(b"invalid byte seq \xff here")

    # Run treemap generation
    fig = pmv.py_pkg_treemap(pkg_name)
    captured = capsys.readouterr()  # Capture print output

    # Check that a warning was printed
    assert f"Warning: Could not analyze {bad_file_path}: 'utf-8' codec" in captured.out

    # Check that the bad file node does NOT exist because its cell_value will be 0
    for id_val in fig.data[0].ids:
        assert "bad_encoding" not in id_val, (
            f"bad_encoding.py with {id_val=} should be omitted due to zero cell_value"
        )


@pytest.mark.parametrize(
    ("metadata_dict", "expected_url_in_customdata"),
    [
        # Case 1: No Home-page, no Project-URL
        ({}, False),
        # Case 2: Home-page exists but not GitHub
        ({"Home-page": "https://example.com"}, False),
        # Case 3: Project-URL exists, but no GitHub
        ({"Project-URL": ["Source Code, https://gitlab.com/user/repo"]}, False),
        # Case 4: Malformed Project-URL
        ({"Project-URL": ["GitStuff https://github.com/user/repo"]}, False),
        # Case 5: Valid Home-page (GitHub)
        ({"Home-page": "https://github.com/user/repo"}, True),
        # Case 6: Valid Project-URL (GitHub)
        (
            {
                "Project-URL": [
                    "Homepage, https://example.com",
                    "Repository, https://github.com/user/repo2",
                ]
            },
            True,
        ),
    ],
    ids=[
        "no_urls",
        "homepage_not_github",
        "project_url_not_github",
        "malformed_project_url",
        "github_homepage",
        "github_project_url",
    ],
)
def test_auto_base_url_metadata(
    metadata_dict: dict[str, list[str]],
    expected_url_in_customdata: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test base_url='auto' with various mocked package metadata."""
    pkg_name = "my_pkg"

    mock_metadata = MagicMock()
    mock_metadata.get.side_effect = lambda key, default=None: metadata_dict.get(
        key, default
    )
    mock_metadata.get_all.side_effect = lambda key, default=None: metadata_dict.get(
        key, default or []
    )

    # Mock importlib.metadata.metadata to return our mock
    monkeypatch.setattr("importlib.metadata.metadata", lambda _: mock_metadata)

    fig = pmv.py_pkg_treemap(pkg_name, base_url="auto")

    custom_data = fig.data[0].customdata

    if expected_url_in_customdata:
        assert custom_data is not None, "Customdata should exist when URL is found"
        expected_cols = 12
        assert custom_data.shape[1] == expected_cols
        file_urls = custom_data[:, 3]  # Fourth column (index 3) is file_url
        # Check if at least one file URL (for a .py file) contains github.com/blob/main
        assert any(
            url and "github.com/user/" in url and "/blob/main/" in url
            for url in file_urls
            if isinstance(url, str)
        ), "Expected a valid GitHub blob URL in customdata"
    else:
        assert (
            custom_data is None
            or custom_data.shape[1] != 12
            or not any(
                url and "github.com" in url
                for url in custom_data[:, 3]
                if isinstance(url, str)
            )
        ), "Customdata should be None or not contain GitHub URLs"


def test_auto_base_url_metadata_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Test base_url='auto' when metadata lookup raises an exception."""
    pkg_name = "my_pkg"

    def mock_metadata_raises(_name: str) -> None:
        raise RuntimeError("Metadata fetch failed")

    monkeypatch.setattr("importlib.metadata.metadata", mock_metadata_raises)

    fig = pmv.py_pkg_treemap(pkg_name, base_url="auto")
    captured = capsys.readouterr()

    # Check warning printed and no URL generated
    assert (
        "Warning: Error processing metadata for my_pkg: Metadata fetch failed"
        in captured.out
    )
    # Customdata is generated, but file_url (index 3) should be None
    custom_data = fig.data[0].customdata
    assert custom_data is not None
    assert np.all(custom_data[:, 3] == "None")


@pytest.mark.parametrize(
    ("show_counts_value", "expected_textinfo", "expected_template_part"),
    [
        ("value", "label+value", "%{label}<br>%{value:,}"),
        ("percent", "label+percent entry", None),  # No specific template needed
        (False, "label", None),  # No specific template needed
    ],
)
def test_show_counts_no_base_url(
    show_counts_value: ShowCounts,
    expected_textinfo: str,
    expected_template_part: str | None,
) -> None:
    """Test different show_counts values when base_url is None."""
    pkg_name = "my_pkg"
    fig = pmv.py_pkg_treemap(pkg_name, base_url=None, show_counts=show_counts_value)

    assert fig.data[0].textinfo == expected_textinfo
    if expected_template_part:
        assert fig.data[0].texttemplate == expected_template_part
    else:
        # For percent and False, texttemplate might be None or default Plotly
        assert fig.data[0].texttemplate != "needs_checking"  # Placeholder assert


@pytest.fixture
def mock_pkg_data() -> pd.DataFrame:
    """Create a mock DataFrame similar to collect_package_modules output."""
    data = {
        "package": ["pkg1", "pkg1", "pkg1", "pkg2"],
        "package_name_raw": ["pkg1", "pkg1", "pkg1", "pkg2"],
        "full_module": ["pkg1.modA", "pkg1.modB.sub", "pkg1.modC", "pkg2.modX"],
        "filename": ["modA", "sub", "modC", "modX"],
        "directory": ["pkg1", "pkg1", "pkg1", "pkg2"],
        "top_module": ["modA", "modB", "modC", "modX"],
        "line_count": [100, 50, 150, 200],
        "file_path": [
            "/path/pkg1/modA.py",
            "/path/pkg1/modB/sub.py",
            "/path/pkg1/modC.py",
            "/path/pkg2/modX.py",
        ],
        "repo_path_segment": [
            "pkg1/modA.py",
            "pkg1/modB/sub.py",
            "pkg1/modC.py",
            "pkg2/modX.py",
        ],
        "leaf_label": ["modA", "sub", "modC", "modX"],
        "module_parts": [["modA"], ["modB", "sub"], ["modC"], ["modX"]],
        "depth": [1, 2, 1, 1],
        "n_classes": [2, 1, 3, 4],
        "n_functions": [5, 2, 8, 10],
        "n_internal_imports": [1, 0, 2, 3],
        "n_external_imports": [4, 1, 5, 6],
        "n_type_checking_imports": [0, 1, 0, 2],
        "n_methods": [3, 1, 4, 5],  # Added n_methods to mock data
    }
    return pd.DataFrame(data)


def test_py_pkg_treemap_tooltip_data(mock_pkg_data: pd.DataFrame) -> None:
    """Test that pmv.py_pkg_treemap generates correct customdata and hovertemplate."""
    # Simulate the DataFrame as it would be *after* collect_package_modules
    # but *before* cell_value calculation if we were testing that part directly.
    # However, py_pkg_treemap calculates cell_value internally.
    # Since this test uses default calculator, cell_value will equal line_count.
    # The mock for collect_package_modules should provide the raw data.
    # py_pkg_treemap will then add 'cell_value' and other derived columns.

    # The DataFrame returned by the mocked collect_package_modules
    mock_collect_df = mock_pkg_data.copy()

    dot_path = "pymatviz.treemap.py_pkg.collect_package_modules"
    with (
        patch(dot_path, return_value=mock_collect_df) as mock_collect,
        patch(
            "pymatviz.treemap.py_pkg.px.treemap", wraps=px.treemap
        ) as mock_px_treemap,
    ):
        expected_hovertemplate = (
            "<b>%{customdata[2]}</b><br>"  # leaf_label
            "Path: %{customdata[1]}<br>"  # repo_path_segment
            "Cell Value: %{value:,}<br>"  # This is cell_value (new sizing metric)
            "Lines: %{customdata[11]:,}<br>"  # Actual line_count (index 11)
            # percent_of_package_cell_value (index 10)
            "%{customdata[10]:.1%} of %{customdata[0]} (by cell value)<br>"
            "Classes: %{customdata[4]:,}<br>"  # n_classes (index 4)
            "Functions: %{customdata[5]:,}<br>"  # n_functions (index 5)
            "Methods: %{customdata[9]:,}<br>"  # n_methods (index 9)
            "Internal Imports: %{customdata[6]:,}<br>"  # n_internal_imports (index 6)
            "External Imports: %{customdata[7]:,}<br>"  # n_external_imports (index 7)
            "Type Imports: %{customdata[8]:,}<br>"  # n_type_checking_imports (index 8)
            "<extra></extra>"  # Remove Plotly trace info box
        )
        expected_custom_data_cols = [
            "package_name_raw",  # 0
            "repo_path_segment",  # 1
            "leaf_label",  # 2
            "file_url",  # 3
            "n_classes",  # 4
            "n_functions",  # 5
            "n_internal_imports",  # 6
            "n_external_imports",  # 7
            "n_type_checking_imports",  # 8
            "n_methods",  # 9
            "percent_of_package_cell_value",  # 10
            "line_count",  # 11
        ]

        fig = pmv.py_pkg_treemap(packages=["pkg1", "pkg2"], base_url=None)

        mock_collect.assert_called_once()
        mock_px_treemap.assert_called_once()

        passed_custom_data_arg = mock_px_treemap.call_args.kwargs["custom_data"]

        assert len(fig.data) == 1
        assert fig.data[0].hovertemplate == expected_hovertemplate

        # For constructing the expected custom_data, we need to simulate what
        # py_pkg_treemap does with the data from collect_package_modules.
        # It adds 'cell_value', 'package_total_cell_value',
        # 'percent_of_package_cell_value', 'file_url', and formats 'package' labels.

        # Start with the mocked collect_df and add expected derived columns
        df_for_custom_data_calc = mock_collect_df.copy()
        # Since default calculator is used, cell_value = line_count
        df_for_custom_data_calc["cell_value"] = df_for_custom_data_calc["line_count"]
        package_cell_value_totals = df_for_custom_data_calc.groupby("package_name_raw")[
            "cell_value"
        ].sum()
        df_for_custom_data_calc["package_total_cell_value"] = df_for_custom_data_calc[
            "package_name_raw"
        ].map(package_cell_value_totals)
        df_for_custom_data_calc["percent_of_package_cell_value"] = (
            df_for_custom_data_calc["cell_value"]
            / df_for_custom_data_calc["package_total_cell_value"]
        ).fillna(0)
        df_for_custom_data_calc["file_url"] = None  # base_url is None

        analysis_cols = [
            "n_classes",
            "n_functions",
            "n_internal_imports",
            "n_external_imports",
            "n_type_checking_imports",
            "n_methods",
        ]
        for col in analysis_cols:
            if col in df_for_custom_data_calc:
                df_for_custom_data_calc[col] = df_for_custom_data_calc[col].fillna(0)
            else:  # Should not happen if mock_pkg_data is complete
                df_for_custom_data_calc[col] = 0

        expected_leaf_customdata_df = df_for_custom_data_calc[expected_custom_data_cols]

        generated_customdata_df = pd.DataFrame(
            passed_custom_data_arg, columns=expected_custom_data_cols
        )

        # Sort for consistent comparison
        generated_customdata_df_sorted = generated_customdata_df.sort_values(
            by="repo_path_segment"
        ).reset_index(drop=True)
        expected_leaf_customdata_df_sorted = expected_leaf_customdata_df.sort_values(
            by="repo_path_segment"
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            generated_customdata_df_sorted,
            expected_leaf_customdata_df_sorted,
            check_dtype=False,
            check_exact=False,
            atol=1e-6,
        )

        # Sanity check a specific value
        pkg1_mod_a_percent_gen = generated_customdata_df_sorted[
            generated_customdata_df_sorted["leaf_label"] == "modA"
        ]["percent_of_package_cell_value"].iloc[0]
        assert pkg1_mod_a_percent_gen == pytest.approx(100 / 300)
        assert generated_customdata_df_sorted["file_url"].isna().all()


def _calc_sum_functions_classes(metrics: pmv.treemap.py_pkg.ModuleStats) -> int:
    """Test calculator: sum of functions and classes."""
    return metrics.n_functions + metrics.n_classes


def _calc_methods_x_10_plus_lines(metrics: pmv.treemap.py_pkg.ModuleStats) -> int:
    """Test calculator: methods * 10 + line_count."""
    return metrics.n_methods * 10 + metrics.line_count


def _sizer_omit_specific(metrics: pmv.treemap.py_pkg.ModuleStats) -> int:
    """Return 0 for 'module1' and 'main', otherwise line_count."""
    if metrics.filename in {"module1", "main"}:
        return 0
    return metrics.line_count


@pytest.mark.parametrize(
    ("calculator", "expected_value_logic", "expected_filenames", "test_id"),
    [
        (
            None,  # Default behavior (line_count)
            lambda metrics: metrics.line_count,
            {"module1", "module2", "module3", "module4_typed", "main", "top_level_mod"},
            "default_line_count",
        ),
        (
            _calc_sum_functions_classes,
            _calc_sum_functions_classes,
            # module3, main, top_level_mod have 0
            {"module1", "module2", "module4_typed"},
            "sum_functions_classes",
        ),
        (
            _calc_methods_x_10_plus_lines,
            _calc_methods_x_10_plus_lines,
            {"module1", "module2", "module3", "module4_typed", "main"},
            "methods_x_10_plus_lines",
        ),
        (
            lambda metrics: metrics.n_type_checking_imports
            + metrics.n_internal_imports,
            lambda metrics: metrics.n_type_checking_imports
            + metrics.n_internal_imports,
            {"module4_typed"},
            "type_plus_internal_imports",
        ),
        (
            _sizer_omit_specific,  # omits module1 and main
            _sizer_omit_specific,
            {"module2", "module3", "module4_typed", "top_level_mod"},
            "omit_module1_main",
        ),
    ],
)
def test_py_pkg_treemap_cell_size_fn(
    calculator: pmv.treemap.py_pkg.CellSizeFn | None,
    expected_value_logic: pmv.treemap.py_pkg.CellSizeFn,
    expected_filenames: set[str],
    test_id: str,  # For clarity in test output
) -> None:
    """Test the cell_size_fn argument in py_pkg_treemap.

    Verifies that the 'cell_value' column in the DataFrame passed to px.treemap
    is correctly computed based on the provided calculator, and that modules
    resulting in a cell_value of 0 are omitted.
    """

    with patch("plotly.express.treemap") as mock_px_treemap:
        pmv.py_pkg_treemap(
            ["my_pkg", "another_pkg"],  # Test with multiple packages
            cell_size_fn=calculator,
            group_by="file",  # Group by file for easier checking of filenames
        )

    mock_px_treemap.assert_called_once()
    # Ensure we are accessing the args tuple correctly from the call object
    call_object = mock_px_treemap.call_args
    assert call_object is not None, "px.treemap was not called or call_args is None"
    positional_args, keyword_args = (
        call_object.args,
        call_object.kwargs,
    )  # Get both args and kwargs
    assert len(positional_args) > 0, (
        "px.treemap was called with no positional arguments"
    )
    df_passed_to_plotly: pd.DataFrame = positional_args[0]

    assert "cell_value" in df_passed_to_plotly
    assert "filename" in df_passed_to_plotly

    actual_filenames = set(df_passed_to_plotly["filename"])
    assert actual_filenames >= expected_filenames, (
        f"Test ID: {test_id} - Mismatch in expected present filenames. "
        f"Expected: {expected_filenames}, Got: {actual_filenames}"
    )

    # Iterate through the DataFrame and check if cell_value matches expected logic
    for _idx, row in df_passed_to_plotly.iterrows():
        # Construct ModuleStats from the row data
        # Ensure all necessary fields for ModuleStats are present in the row
        # collect_package_modules ensures this.
        module_metrics_dict = row.to_dict()
        # Filter dict to only include keys expected by ModuleStats constructor
        # to prevent TypeError if df_passed_to_plotly has extra columns.
        metrics_keys = pmv.treemap.py_pkg.ModuleStats._fields
        filtered_metrics_dict = {
            key: module_metrics_dict[key]
            for key in metrics_keys
            if key in module_metrics_dict
        }
        # Some metrics might be NaN if a file couldn't be analyzed, fill with 0
        for key in [
            "n_classes",
            "n_functions",
            "n_methods",
            "n_internal_imports",
            "n_external_imports",
            "n_type_checking_imports",
        ]:
            if pd.isna(filtered_metrics_dict.get(key)):
                filtered_metrics_dict[key] = 0

        try:
            metrics_instance = pmv.treemap.py_pkg.ModuleStats(**filtered_metrics_dict)
        except TypeError as exc:
            missing_keys = set(metrics_keys) - set(filtered_metrics_dict)
            if missing_keys:
                raise AssertionError(
                    f"{missing_keys=} for ModuleStats in row: {row.to_dict()}"
                ) from exc
            raise  # Re-raise if it's a different TypeError

        expected_val = expected_value_logic(metrics_instance)
        assert expected_val > 0, (
            f"Test ID: {test_id} - Expected value logic should not result in 0 for a "
            f"file that is supposed to be present. File: {row.get('filename')}"
        )
        assert row["cell_value"] == expected_val, (
            f"Test ID: {test_id} - Mismatch for file {row.get('file_path', 'N/A')}. "
            f"Expected: {expected_val}, Got: {row['cell_value']}"
        )

    # Also check that the 'values' argument to px.treemap was 'cell_value'
    assert keyword_args["values"] == "cell_value"
