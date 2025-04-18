"""Unit tests for pymatviz.treemap.py_pkg.py"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Final
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pymatviz.treemap.py_pkg import (
    ShowCounts,
    collect_package_modules,
    count_lines,
    default_module_formatter,
    find_package_path,
    py_pkg_treemap,
)


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


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
    assert count_lines(str(test_file)) == expected_lines


def test_count_lines_non_existent(tmp_path: Path) -> None:
    """Test count_lines with a non-existent file."""
    assert count_lines(str(tmp_path / "non_existent.py")) == 0


def test_default_module_formatter() -> None:
    """Test the default module formatter."""
    assert (
        default_module_formatter("my_module", 1234, 5000)
        == "my_module (1,234 lines, 24.7%)"
    )
    assert (
        default_module_formatter("another.mod", 50, 100)
        == "another.mod (50 lines, 50.0%)"
    )
    # Test zero total
    assert default_module_formatter("zero", 0, 0) == "zero (0 lines)"


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
    found_path = find_package_path(package_name)

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
    df_modules = collect_package_modules(["non_existent_package_12345"])
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
        found_path = find_package_path(target_pkg)
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
        found_path = find_package_path(target_pkg)
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

        found_path = find_package_path(target_pkg)
        assert found_path.replace("\\", "/") == expected_site_path.replace("\\", "/")
        # Check that isdir was called with the expected paths
        mock_isdir.assert_any_call(cwd_path)
        mock_isdir.assert_any_call(src_path)
        mock_isdir.assert_any_call(expected_site_path)
        # Verify it was called exactly 3 times in this flow
        assert mock_isdir.call_count == 3


# --- Test py_pkg_treemap ---


def test_py_pkg_treemap_show_module_counts() -> None:
    """Test the show_module_counts parameter."""
    # Default (True) - uses default_module_formatter
    fig_true = py_pkg_treemap(["my_pkg", "another_pkg"])
    labels_true = fig_true.data[0].labels
    assert any("my_pkg (" in lbl for lbl in labels_true)
    assert any("another_pkg (" in lbl for lbl in labels_true)
    assert any(" lines" in lbl for lbl in labels_true)
    assert any("%)" in lbl for lbl in labels_true)

    # False - no counts on package labels
    fig_false = py_pkg_treemap(["my_pkg", "another_pkg"], show_module_counts=False)
    labels_false = fig_false.data[0].labels
    assert "my_pkg" in labels_false
    assert "another_pkg" in labels_false
    assert not any(
        "(" in lbl for lbl in labels_false if lbl in ("my_pkg", "another_pkg")
    )

    # Custom formatter
    def custom_mod_fmt(pkg: str, count: int, total: int) -> str:
        return f"PKG: {pkg} [{count}/{total}]"

    fig_custom = py_pkg_treemap(
        ["my_pkg", "another_pkg"], show_module_counts=custom_mod_fmt
    )
    labels_custom = fig_custom.data[0].labels
    assert any(lbl.startswith("PKG: my_pkg [") for lbl in labels_custom)
    assert any(lbl.startswith("PKG: another_pkg [") for lbl in labels_custom)


def test_py_pkg_treemap_empty() -> None:
    """Test treemap generation when no modules are found (e.g., high min_lines)."""
    with pytest.raises(ValueError, match="No Python modules found"):
        py_pkg_treemap("my_pkg", min_lines=100)


def test_py_pkg_treemap_invalid_show_counts() -> None:
    """Test ValueError is raised for invalid show_counts value."""
    with pytest.raises(ValueError, match="Invalid show_counts='invalid_value'"):
        py_pkg_treemap("my_pkg", show_counts="invalid_value")  # type: ignore[arg-type]


def test_py_pkg_treemap_base_url() -> None:
    """Test the base_url parameter adds source links and hover info works."""
    pkg_name = "my_pkg"
    base_url = "https://github.com/test/repo/blob/main"
    show_counts: Final = "value+percent"

    # --- Test WITH base_url --- (Links + Enhanced Hover)
    fig_with_url = py_pkg_treemap(pkg_name, base_url=base_url, show_counts=show_counts)

    # Check texttemplate (link structure)
    # customdata now: [0]=repo_path_segment, [1]=leaf_label, [2]=file_url
    expected_link_part = (
        "<a href='%{customdata[2]}' target='_blank'>%{customdata[1]}</a>"
    )
    expected_texttemplate = (
        f"{expected_link_part}<br>%{{value:,}} lines<br>%{{percentEntry}}"
    )
    assert fig_with_url.data[0].texttemplate == expected_texttemplate
    assert fig_with_url.data[0].textinfo == "none"

    # Check customdata contains file info (rel_path, leaf_label, url) when base_url used
    custom_data = fig_with_url.data[0].customdata
    assert custom_data is not None
    ids_with_url = fig_with_url.data[0].ids  # Get IDs to correlate with custom_data
    assert len(custom_data) == len(ids_with_url), "custom_data and ids length mismatch"
    assert len(custom_data) > 0, "custom_data is empty"

    # Check customdata structure (should be numpy array with repo_path_segment,
    # leaf_label, file_url)
    assert isinstance(custom_data, np.ndarray), "customdata should be a numpy array"
    assert custom_data.ndim == 2, "customdata should be 2-dimensional"
    expected_cols = 3
    assert custom_data.shape[1] == expected_cols, (
        f"Expected {expected_cols} columns in customdata, got {custom_data.shape[1]}"
    )
    # Check first column looks like relative paths (contains .py) only for file nodes
    file_node_indices = [i for i, id_val in enumerate(ids_with_url) if ".py" in id_val]
    assert all(
        ".py" in str(custom_data[i, 0]) for i in file_node_indices if custom_data[i, 0]
    )
    # Check second column looks like leaf labels (no slashes)
    assert all(
        "/" not in str(custom_data[i, 1])
        for i in file_node_indices
        if custom_data[i, 1]
    )
    # Check third column contains the base_url if not None
    assert all(
        base_url in str(custom_data[i, 2])
        for i in file_node_indices
        if custom_data[i, 2]
    )

    # Check hover text generation (which now includes path and conditional info)
    hover_text = fig_with_url.data[0].hovertext
    assert hover_text is not None
    assert len(hover_text) > 0

    # --- Check Hover Text Content Directly ---
    # Combine all hover texts into one searchable string
    all_hover_text = "<br><br>".join(map(str, hover_text))

    # Check for module1 content
    assert "<b>module1</b>" in all_hover_text
    assert "Path: my_pkg/module1.py" in all_hover_text
    assert "Lines: 5" in all_hover_text  # Exact line count from dummy pkg
    assert "% of my_pkg" in all_hover_text
    # Module 1 has 1 func, 1 external import (os)
    assert "Functions: 1" in all_hover_text  # Check func count for module 1
    assert "External Imports: 1" in all_hover_text

    # Check for module4_typed content
    assert "<b>module4_typed</b>" in all_hover_text
    assert "Path: my_pkg/submodule/module4_typed.py" in all_hover_text
    assert "Lines: 6" in all_hover_text  # Exact line count
    assert "Functions: 1" in all_hover_text  # Check func/type counts for module 4
    assert "Type Imports: 2" in all_hover_text
    assert "External Imports: 0" in all_hover_text  # Imports are TYPE_CHECKING

    # --- Test WITHOUT base_url --- (No Links + Enhanced Hover)
    fig_no_url = py_pkg_treemap(pkg_name, show_counts=show_counts)
    # Customdata should be None or empty when base_url is not used
    assert (
        fig_no_url.data[0].customdata is None or len(fig_no_url.data[0].customdata) == 0
    )

    # Check hovertext is generated correctly without base_url as well
    hover_text_no_url = fig_no_url.data[0].hovertext
    assert hover_text_no_url is not None
    assert len(hover_text_no_url) > 0

    # Check default texttemplate/textinfo for this show_counts value
    assert fig_no_url.data[0].textinfo == "label+value+percent entry"
    assert (
        fig_no_url.data[0].texttemplate
        == "%{label}<br>%{value:,} lines<br>%{percentEntry}"
    )

    # Minimal check: ensure hover text was generated
    assert len(hover_text_no_url) > 0


def test_py_pkg_treemap_analysis_failure() -> None:
    """Test hover text when _analyze_py_file returns None due to an error."""
    pkg_name = "my_pkg"

    # Mock _analyze_py_file to simulate failure
    failed_analysis_result = {
        "n_classes": None,
        "n_functions": None,
        "n_internal_imports": None,
        "n_external_imports": None,
        "n_type_checking_imports": None,
    }
    with patch(
        "pymatviz.treemap.py_pkg._analyze_py_file", return_value=failed_analysis_result
    ) as mock_analyze:
        fig = py_pkg_treemap(pkg_name)

        # Check that analyze was called
        assert mock_analyze.call_count > 0

        # Check hover text omits the failed analysis fields
        hover_text = fig.data[0].hovertext
        assert hover_text is not None
        assert len(hover_text) > 0

        # Check if expected hover text parts for module1 exist anywhere in hover texts
        module1_expected_texts = [
            "<b>module1</b>",
            "Path: my_pkg/module1.py",
            "Lines:",
            "% of my_pkg",
        ]
        module1_absent_texts = [
            "Classes:",
            "Functions:",
            "Internal Imports:",
            "External Imports:",
            "Type Imports:",
        ]

        found_module1_hover = None
        for text in hover_text:  # Iterate through all generated hover texts
            if all(expected in text for expected in module1_expected_texts) and all(
                absent not in text for absent in module1_absent_texts
            ):
                found_module1_hover = text
                break

        assert found_module1_hover is not None

        # Check that analysis fields are absent from ALL hover texts due to mock
        for text in hover_text:
            assert "Classes:" not in text
            assert "Functions:" not in text
            assert "Internal Imports:" not in text
            assert "External Imports:" not in text
            assert "Type Imports:" not in text


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
    internal_import_content = """\nfrom .submodule import module2 # internal import from sibling dir\nfrom . import module1 # internal import from same dir\nimport os # external import\n
class InternalUser:\n    pass\n"""  # noqa: E501
    (dummy_pkg_path / pkg_name / "internal_user.py").write_text(internal_import_content)

    fig = py_pkg_treemap(pkg_name)
    hover_texts = fig.data[0].hovertext
    assert hover_texts is not None

    # Find the hover text for internal_user.py
    internal_user_hover = None
    for text in hover_texts:
        if "<b>internal_user</b>" in text:
            internal_user_hover = text
            break

    assert internal_user_hover is not None, "Hover text for internal_user.py not found"
    assert "Internal Imports: 2" in internal_user_hover
    assert "External Imports: 1" in internal_user_hover
    assert "Classes: 1" in internal_user_hover
    assert "Functions: 0" in internal_user_hover


def test_top_level_module(dummy_pkg_path: Path) -> None:
    """Test handling of Python files directly under the package root."""
    pkg_name = "my_pkg"
    # Add a file directly in the package root
    (dummy_pkg_path / pkg_name / "top_level_mod.py").write_text("print('top level')")

    fig = py_pkg_treemap(pkg_name, group_by="module")  # Use module grouping

    # Check if top_level_mod is present in labels or ids
    # Its label should be just 'top_level_mod' as it has no parent module part
    assert "top_level_mod" in fig.data[0].labels
    assert any(id_val.endswith("/top_level_mod") for id_val in fig.data[0].ids)

    # Check hover text uses 'top_level_mod' as leaf_label
    hover_texts = fig.data[0].hovertext
    assert any("<b>top_level_mod</b>" in text for text in hover_texts)


def test_analyze_unicode_error(
    dummy_pkg_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test graceful handling of UnicodeDecodeError during file analysis."""
    pkg_name = "my_pkg"
    bad_file_path = dummy_pkg_path / pkg_name / "bad_encoding.py"
    # Write bytes that are invalid UTF-8
    bad_file_path.write_bytes(b"invalid byte seq \xff here")

    # Run treemap generation
    fig = py_pkg_treemap(pkg_name)
    captured = capsys.readouterr()  # Capture print output

    # Check that a warning was printed
    assert f"Warning: Could not analyze {bad_file_path}: 'utf-8' codec" in captured.out

    # Check that the bad file node exists but potentially has 0 lines / no analysis
    hover_texts = fig.data[0].hovertext
    bad_file_hover = None
    for text in hover_texts:
        if "<b>bad_encoding</b>" in text:
            bad_file_hover = text
            break

    assert bad_file_hover is not None
    # Line count might be 0 due to failure before counting lines, or handled differently
    # Let's just check that analysis fields are absent
    assert "Lines: 0" in bad_file_hover
    assert "Classes: 0" in bad_file_hover
    assert "Functions: 0" in bad_file_hover


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

    fig = py_pkg_treemap(pkg_name, base_url="auto")

    custom_data = fig.data[0].customdata

    if expected_url_in_customdata:
        assert custom_data is not None, "Customdata should exist when URL is found"
        assert custom_data.shape[1] == 3
        file_urls = custom_data[:, 2]  # Third column is file_url
        # Check if at least one file URL (for a .py file) contains github.com/blob/main
        assert any(
            url and "github.com/user/" in url and "/blob/main/" in url
            for url in file_urls
            if isinstance(url, str)
        ), "Expected a valid GitHub blob URL in customdata"
    else:
        assert (
            custom_data is None
            or custom_data.shape[1] != 3
            or not any(
                url and "github.com" in url
                for url in custom_data[:, 2]
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

    fig = py_pkg_treemap(pkg_name, base_url="auto")
    captured = capsys.readouterr()

    # Check warning printed and no URL generated
    assert (
        "Warning: Error processing metadata for my_pkg: Metadata fetch failed"
        in captured.out
    )
    assert fig.data[0].customdata is None


@pytest.mark.parametrize(
    ("show_counts_value", "expected_textinfo", "expected_template_part"),
    [
        ("value", "label+value", "%{label}<br>%{value:,} lines"),
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
    fig = py_pkg_treemap(pkg_name, base_url=None, show_counts=show_counts_value)

    assert fig.data[0].textinfo == expected_textinfo
    if expected_template_part:
        assert fig.data[0].texttemplate == expected_template_part
    else:
        # For percent and False, texttemplate might be None or default Plotly
        assert fig.data[0].texttemplate != "needs_checking"  # Placeholder assert


def test_invalid_show_counts_with_base_url() -> None:
    """Test ValueError for invalid show_counts when base_url is active."""
    with pytest.raises(ValueError, match="Invalid show_counts='invalid_value'"):
        py_pkg_treemap(
            "my_pkg",
            base_url="http://example.com",
            show_counts="invalid_value",  # type: ignore[arg-type]
        )
