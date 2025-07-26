"""Unit tests for pymatviz.treemap.py_pkg.py."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Final
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest

import pymatviz as pmv


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from pymatviz.treemap.py_pkg import CellSizeFn, ModuleFormatter, ShowCounts


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
    """Test find_package_path for local dummy and non-existent packages."""
    is_dummy = package_name in ("my_pkg", "another_pkg")
    found_path = pmv.treemap.py_pkg.find_package_path(package_name)

    if expected_end:
        if is_dummy:
            expected_abs_path = str(dummy_pkg_path / package_name).replace("\\", "/")
            assert found_path.replace("\\", "/").endswith(expected_abs_path)
        elif "site-packages" in expected_end:
            assert expected_end in found_path.replace("\\", "/")
        else:
            rel_path = os.path.relpath(found_path, start=dummy_pkg_path.parent)
            assert rel_path.replace("\\", "/") == expected_end
    else:
        assert found_path == ""


def test_find_package_path_fallbacks(dummy_pkg_path: Path) -> None:
    """Test fallback mechanisms in find_package_path using mocking."""
    target_pkg = "my_pkg"
    expected_abs_path = str(dummy_pkg_path / target_pkg).replace("\\", "/")

    # Test fallback to importlib.util when importlib.resources fails
    with (
        patch("importlib.resources.files", side_effect=ImportError, create=True),
        patch("importlib.util.find_spec") as mock_find_spec,
    ):
        mock_spec = MagicMock()
        mock_spec.origin = str(dummy_pkg_path / target_pkg / "__init__.py")
        mock_find_spec.return_value = mock_spec
        found_path = pmv.treemap.py_pkg.find_package_path(target_pkg)
        assert found_path.replace("\\", "/") == expected_abs_path

    # Test fallback when find_spec returns spec for a single module
    with (
        patch("importlib.resources.files", side_effect=ImportError, create=True),
        patch("importlib.util.find_spec") as mock_find_spec,
    ):
        mock_spec = MagicMock()
        mock_spec.origin = str(dummy_pkg_path / target_pkg / "module1.py")
        mock_find_spec.return_value = mock_spec
        found_path = pmv.treemap.py_pkg.find_package_path(target_pkg)
        assert found_path.replace("\\", "/") == expected_abs_path

    # Test fallback to site-packages check
    with (
        patch("importlib.resources.files", side_effect=ImportError, create=True),
        patch("importlib.util.find_spec", return_value=None),
        patch("site.getsitepackages", return_value=[str(dummy_pkg_path.parent)]),
        patch("os.path.isdir") as mock_isdir,
    ):
        expected_site_path = str(dummy_pkg_path.parent / target_pkg)

        mock_isdir.side_effect = lambda p: p == expected_site_path
        found_path = pmv.treemap.py_pkg.find_package_path(target_pkg)
        assert found_path.replace("\\", "/") == expected_site_path.replace("\\", "/")
        assert mock_isdir.call_count == 3


@pytest.mark.parametrize(
    ("cell_text_fn", "expected_in_labels", "expected_not_in_labels"),
    [
        (True, [" lines", "%)", "my_pkg (", "another_pkg ("], []),  # Default formatter
        (False, ["my_pkg", "another_pkg"], ["(", " lines"]),  # No formatting
        (
            lambda pkg,
            count,
            total: f"PKG: {pkg} [{count}/{total}]",  # Custom formatter
            ["PKG: my_pkg [", "PKG: another_pkg ["],
            [],
        ),
    ],
)
def test_py_pkg_treemap_cell_text_fn(
    cell_text_fn: ModuleFormatter | bool,
    expected_in_labels: list[str],
    expected_not_in_labels: list[str],
) -> None:
    """Test the cell_text_fn parameter with different options."""
    fig = pmv.py_pkg_treemap(["my_pkg", "another_pkg"], cell_text_fn=cell_text_fn)
    labels = fig.data[0].labels

    for expected in expected_in_labels:
        assert any(expected in str(lbl) for lbl in labels), (
            f"Expected '{expected}' in labels"
        )

    for not_expected in expected_not_in_labels:
        if cell_text_fn is False:
            # For False case, check specific package labels don't have formatting
            pkg_labels = [lbl for lbl in labels if lbl in ("my_pkg", "another_pkg")]
            assert not any(not_expected in str(lbl) for lbl in pkg_labels)
        else:
            assert not any(not_expected in str(lbl) for lbl in labels), (
                f"Unexpected '{not_expected}' in labels"
            )


@pytest.mark.parametrize(
    ("invalid_input", "expected_error"),
    [
        (lambda _cell: 0, "No Python modules found .* after filtering by cell_size_fn"),
        ("invalid_value", "Invalid show_counts='invalid_value'"),
    ],
)
def test_py_pkg_treemap_errors(
    invalid_input: CellSizeFn | ShowCounts, expected_error: str
) -> None:
    """Test error conditions."""
    if callable(invalid_input):
        with pytest.raises(ValueError, match=expected_error):
            pmv.py_pkg_treemap("my_pkg", cell_size_fn=invalid_input)
    else:
        with pytest.raises(ValueError, match=expected_error):
            pmv.py_pkg_treemap("my_pkg", show_counts=invalid_input)


@pytest.mark.parametrize(
    ("test_base_url", "expect_link"),
    [
        ("https://github.com/test/repo/blob/main", True),
        (None, False),
    ],
)
def test_py_pkg_treemap_base_url(test_base_url: str | None, expect_link: bool) -> None:
    """Test base_url handling for links and hover info."""
    pkg_name = "my_pkg"
    show_counts: Final = "value+percent"

    fig = pmv.py_pkg_treemap(pkg_name, base_url=test_base_url, show_counts=show_counts)
    trace = fig.data[0]

    # Check texttemplate and textinfo
    if expect_link:
        expected_texttemplate = (
            "<a href='%{customdata[3]}' target='_blank'>%{customdata[2]}</a><br>"
            "%{value:,}<br>%{percentEntry}"
        )
        assert trace.texttemplate == expected_texttemplate
        assert trace.textinfo == "none"
    else:
        assert trace.texttemplate == "%{label}<br>%{value:,}<br>%{percentEntry}"
        assert trace.textinfo == "label+value+percent entry"

    # Check customdata structure
    custom_data = trace.customdata
    assert custom_data is not None
    assert custom_data.shape[1] == 12

    file_urls = custom_data[:, 3]  # Index 3 is file_url
    if expect_link:
        file_node_indices = [i for i, id_val in enumerate(trace.ids) if ".py" in id_val]
        assert all(
            str(test_base_url) in str(file_urls[i])
            for i in file_node_indices
            if file_urls[i] is not None
        )
    else:
        assert np.all(file_urls == "None")

    # Check hovertemplate generation
    hovertemplate = trace.hovertemplate
    assert hovertemplate is not None
    assert "<b>%{customdata[2]}</b>" in hovertemplate
    assert "Path: %{customdata[1]}<br>" in hovertemplate
    assert "Lines: %{customdata[11]:,}<br>" in hovertemplate


def test_py_pkg_treemap_analysis_scenarios() -> None:
    """Test various analysis scenarios including failures."""
    pkg_name = "my_pkg"

    # Test normal analysis
    fig_normal = pmv.py_pkg_treemap(pkg_name)
    assert isinstance(fig_normal, go.Figure)

    # Test analysis failure
    failed_result = {
        "n_classes": 0,
        "n_functions": 0,
        "n_methods": 0,
        "n_internal_imports": 0,
        "n_external_imports": 0,
        "n_type_checking_imports": 0,
    }
    with patch("pymatviz.treemap.py_pkg._analyze_py_file", return_value=failed_result):
        fig_failed = pmv.py_pkg_treemap(pkg_name)
        assert isinstance(fig_failed, go.Figure)
        custom_data = fig_failed.data[0].customdata
        analysis_data = custom_data[:, 4:9]
        assert np.all(analysis_data.astype(float) == 0)


def test_collect_package_modules_not_found() -> None:
    """Test collect_package_modules with a package that cannot be found."""
    df_modules = pmv.treemap.py_pkg.collect_package_modules(
        ["non_existent_package_12345"]
    )
    assert df_modules.empty


def test_internal_imports(dummy_pkg_path: Path) -> None:
    """Test counting of internal imports using AST analysis."""
    pkg_name = "my_pkg"
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
    assert "Internal Imports: %{customdata[6]:,}<br>" in hovertemplate
    assert "External Imports: %{customdata[7]:,}<br>" in hovertemplate
    assert "Classes: %{customdata[4]:,}<br>" in hovertemplate

    # Verify actual data for the specific node
    ids = fig.data[0].ids
    custom_data = fig.data[0].customdata
    internal_user_idx = next(
        (idx for idx, id_val in enumerate(ids) if id_val.endswith("/internal_user")), -1
    )
    assert internal_user_idx != -1

    # Verify the analysis data
    assert int(custom_data[internal_user_idx, 7]) == 2  # n_internal_imports
    assert int(custom_data[internal_user_idx, 8]) == 1  # n_external_imports
    assert int(custom_data[internal_user_idx, 4]) == 1  # n_classes


def test_top_level_module(dummy_pkg_path: Path) -> None:
    """Test handling of Python files directly under the package root."""
    pkg_name = "my_pkg"
    (dummy_pkg_path / pkg_name / "top_level_mod.py").write_text("print('top level')")

    fig = pmv.py_pkg_treemap(pkg_name, group_by="module")
    assert "top_level_mod" in fig.data[0].labels
    assert any(id_val.endswith("/top_level_mod") for id_val in fig.data[0].ids)


def test_analyze_unicode_error(
    dummy_pkg_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test graceful handling of UnicodeDecodeError during file analysis."""
    pkg_name = "my_pkg"
    bad_file_path = dummy_pkg_path / pkg_name / "bad_encoding.py"
    bad_file_path.write_bytes(b"invalid byte seq \xff here")

    fig = pmv.py_pkg_treemap(pkg_name)
    captured = capsys.readouterr()

    assert f"Warning: Could not analyze {bad_file_path}: 'utf-8' codec" in captured.out
    # Check that the bad file node does NOT exist because its cell_value will be 0
    for id_val in fig.data[0].ids:
        assert "bad_encoding" not in id_val


@pytest.mark.parametrize(
    ("metadata_dict", "expected_url_in_customdata"),
    [
        ({}, False),  # No URLs
        ({"Home-page": "https://example.com"}, False),  # Non-GitHub
        (
            {"Project-URL": ["Source Code, https://gitlab.com/user/repo"]},
            False,
        ),  # Non-GitHub
        ({"Home-page": "https://github.com/user/repo"}, True),  # GitHub homepage
        (
            {"Project-URL": ["Repository, https://github.com/user/repo2"]},
            True,
        ),  # GitHub project URL
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

    monkeypatch.setattr("importlib.metadata.metadata", lambda _: mock_metadata)

    fig = pmv.py_pkg_treemap(pkg_name, base_url="auto")
    custom_data = fig.data[0].customdata

    if expected_url_in_customdata:
        assert custom_data is not None
        assert custom_data.shape[1] == 12
        file_urls = custom_data[:, 3]
        assert any(
            url and "github.com/user/" in url and "/blob/main/" in url
            for url in file_urls
            if isinstance(url, str)
        )
    elif custom_data is not None:
        assert not any(
            url and "github.com" in url
            for url in custom_data[:, 3]
            if isinstance(url, str)
        )


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

    assert (
        "Warning: Error processing metadata for my_pkg: Metadata fetch failed"
        in captured.out
    )
    custom_data = fig.data[0].customdata
    assert custom_data is not None
    assert np.all(custom_data[:, 3] == "None")


@pytest.mark.parametrize(
    ("show_counts_value", "expected_textinfo"),
    [
        ("value", "label+value"),
        ("percent", "label+percent entry"),
        (False, "label"),
    ],
)
def test_show_counts_no_base_url(
    show_counts_value: ShowCounts,
    expected_textinfo: str,
) -> None:
    """Test different show_counts values when base_url is None."""
    fig = pmv.py_pkg_treemap("my_pkg", base_url=None, show_counts=show_counts_value)
    assert fig.data[0].textinfo == expected_textinfo


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
        "n_methods": [3, 1, 4, 5],
    }
    return pd.DataFrame(data)


def test_py_pkg_treemap_tooltip_data(mock_pkg_data: pd.DataFrame) -> None:
    """Test that pmv.py_pkg_treemap generates correct customdata and hovertemplate."""
    mock_collect_df = mock_pkg_data.copy()

    with (
        patch(
            "pymatviz.treemap.py_pkg.collect_package_modules",
            return_value=mock_collect_df,
        ),
        patch(
            "pymatviz.treemap.py_pkg.px.treemap", wraps=px.treemap
        ) as mock_px_treemap,
    ):
        expected_hovertemplate = (
            "<b>%{customdata[2]}</b><br>"
            "Path: %{customdata[1]}<br>"
            "Cell Value: %{value:,}<br>"
            "Lines: %{customdata[11]:,}<br>"
            "%{customdata[10]:.1%} of %{customdata[0]} (by cell value)<br>"
            "Classes: %{customdata[4]:,}<br>"
            "Functions: %{customdata[5]:,}<br>"
            "Methods: %{customdata[9]:,}<br>"
            "Internal Imports: %{customdata[6]:,}<br>"
            "External Imports: %{customdata[7]:,}<br>"
            "Type Imports: %{customdata[8]:,}<br>"
            "<extra></extra>"
        )

        fig = pmv.py_pkg_treemap(packages=["pkg1", "pkg2"], base_url=None)

        assert fig.data[0].hovertemplate == expected_hovertemplate
        passed_custom_data_arg = mock_px_treemap.call_args.kwargs["custom_data"]

        # Test specific percentage calculation
        generated_customdata_df = pd.DataFrame(
            passed_custom_data_arg,
            columns=[
                "package_name_raw",
                "repo_path_segment",
                "leaf_label",
                "file_url",
                "n_classes",
                "n_functions",
                "n_internal_imports",
                "n_external_imports",
                "n_type_checking_imports",
                "n_methods",
                "percent_of_package_cell_value",
                "line_count",
            ],
        )

        pkg1_mod_a_percent = generated_customdata_df[
            generated_customdata_df["leaf_label"] == "modA"
        ]["percent_of_package_cell_value"].iloc[0]
        assert pkg1_mod_a_percent == pytest.approx(100 / 300)


# Calculator functions for testing
def _calc_sum_functions_classes(metrics: pmv.treemap.py_pkg.ModuleStats) -> int:
    return metrics.n_functions + metrics.n_classes


def _calc_methods_x_10_plus_lines(metrics: pmv.treemap.py_pkg.ModuleStats) -> int:
    return metrics.n_methods * 10 + metrics.line_count


def _sizer_omit_specific(metrics: pmv.treemap.py_pkg.ModuleStats) -> int:
    if metrics.filename in {"module1", "main"}:
        return 0
    return metrics.line_count


@pytest.mark.parametrize(
    ("calculator", "expected_filenames", "test_id"),
    [
        (None, {"module1", "module2", "module3", "module4_typed", "main"}, "default"),
        (
            _calc_sum_functions_classes,
            {"module1", "module2", "module4_typed"},
            "func+class",
        ),
        (
            _calc_methods_x_10_plus_lines,
            {"module1", "module2", "module3", "module4_typed", "main"},
            "methods*10+lines",
        ),
        (
            lambda m: m.n_type_checking_imports + m.n_internal_imports,
            {"module4_typed"},
            "imports",
        ),
        (
            _sizer_omit_specific,
            {"module2", "module3", "module4_typed"},
            "omit_specific",
        ),
    ],
)
def test_py_pkg_treemap_cell_size_fn(
    calculator: pmv.treemap.py_pkg.CellSizeFn | None,
    expected_filenames: set[str],
    test_id: str,  # noqa: ARG001
) -> None:
    """Test the cell_size_fn argument in py_pkg_treemap."""
    with patch("plotly.express.treemap") as mock_px_treemap:
        pmv.py_pkg_treemap(
            ["my_pkg", "another_pkg"],
            cell_size_fn=calculator,
            group_by="file",
        )

    call_object = mock_px_treemap.call_args
    df_passed_to_plotly: pd.DataFrame = call_object.args[0]

    assert "cell_value" in df_passed_to_plotly
    actual_filenames = set(df_passed_to_plotly["filename"])
    assert actual_filenames >= expected_filenames

    # Verify cell_value calculations match expectations
    for _idx, row in df_passed_to_plotly.iterrows():
        module_metrics_dict = row.to_dict()
        metrics_keys = pmv.treemap.py_pkg.ModuleStats._fields
        filtered_metrics_dict = {
            key: module_metrics_dict[key]
            for key in metrics_keys
            if key in module_metrics_dict
        }

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

        metrics_instance = pmv.treemap.py_pkg.ModuleStats(
            package=filtered_metrics_dict.get("package", ""),
            full_module=filtered_metrics_dict.get("full_module", ""),
            filename=filtered_metrics_dict.get("filename", ""),
            directory=filtered_metrics_dict.get("directory", ""),
            top_module=filtered_metrics_dict.get("top_module", ""),
            line_count=filtered_metrics_dict.get("line_count", 0),
            file_path=filtered_metrics_dict.get("file_path", ""),
            repo_path_segment=filtered_metrics_dict.get("repo_path_segment", ""),
            leaf_label=filtered_metrics_dict.get("leaf_label", ""),
            module_parts=filtered_metrics_dict.get("module_parts", []),
            depth=filtered_metrics_dict.get("depth", 0),
            n_classes=filtered_metrics_dict.get("n_classes", 0),
            n_functions=filtered_metrics_dict.get("n_functions", 0),
            n_methods=filtered_metrics_dict.get("n_methods", 0),
            n_internal_imports=filtered_metrics_dict.get("n_internal_imports", 0),
            n_external_imports=filtered_metrics_dict.get("n_external_imports", 0),
            n_type_checking_imports=filtered_metrics_dict.get(
                "n_type_checking_imports", 0
            ),
        )

        if calculator is None:
            expected_val = metrics_instance.line_count
        else:
            expected_val = calculator(metrics_instance)  # type: ignore[assignment]

        assert expected_val > 0, (
            f"Expected value > 0 for present file: {row.get('filename')}"
        )
        assert int(row["cell_value"]) == expected_val
