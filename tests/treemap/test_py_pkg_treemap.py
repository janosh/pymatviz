"""Unit tests for pymatviz.treemap.py_pkg.py."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Final
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

    from pymatviz.treemap.py_pkg import ModuleFormatter, ShowCounts


@pytest.fixture(scope="module")
def dummy_pkg_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Creates a dummy package structure in a temporary directory."""
    src_dir = tmp_path_factory.mktemp("src")

    # my_pkg structure
    files = {
        "my_pkg/__init__.py": "",
        "my_pkg/module1.py": """# comment
import os

def func1():
    pass # another comment

x = 1
y = 2""",
        "my_pkg/submodule/__init__.py": "",
        "my_pkg/submodule/module2.py": """# comment1
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
print(sys.path)""",
        "my_pkg/submodule/module3.py": """# Just a comment
pass""",
        "my_pkg/submodule/module4_typed.py": """from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from collections import abc

def func_typed():
    pass
""",
    }

    # another_pkg structure
    files |= {
        "another_pkg/__init__.py": "",
        "another_pkg/main.py": """# Main module
print("Running main")

VAR = 100""",
    }
    for rel_path, content in files.items():
        file_path = src_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    return src_dir


@pytest.fixture(autouse=True)
def _add_dummy_to_path(dummy_pkg_path: Path) -> Generator[None, None, None]:
    """Temporarily add the dummy package source directory to sys.path."""
    sys.path.insert(0, str(dummy_pkg_path))
    yield
    sys.path.pop(0)


@pytest.mark.parametrize(
    ("file_content", "prefixes", "expected_lines"),
    [
        ("", ("#",), 0),
        ("\n\n", ("#",), 0),
        ("# comment\n\n# another comment", ("#",), 0),
        ("import os\nx = 1", ("#",), 2),
        ("import os\n# comment\nx = 1\n\ny=2 # inline comment", ("#",), 3),
        ("// rust comment\nlet x = 1;\nlet y = 2;\n", ("//",), 2),
        ("# python comment\nx = 1\ny = 2\n", ("#",), 2),
        ("-- lua comment\nprint('hi')\n", ("--",), 1),
        ("// mixed\n# both\nreal_code = 1\n", ("//", "#"), 1),
        ("code_only\n", ("//",), 1),
        ("// only comments\n// and more\n", ("//",), 0),
        ("# retained without prefixes\n", (), 1),
        (b"valid = 1\n\xff", ("#",), 0),
    ],
)
def test_count_lines(
    tmp_path: Path,
    file_content: str | bytes,
    prefixes: tuple[str, ...],
    expected_lines: int,
) -> None:
    """Count source lines across comment syntaxes, blank files, and invalid UTF-8."""
    test_file = tmp_path / "test.src"
    test_file.write_bytes(
        file_content.encode() if isinstance(file_content, str) else file_content
    )
    assert pmv.treemap.py_pkg.count_lines(str(test_file), prefixes) == expected_lines


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
    assert pmv.treemap.py_pkg.default_module_formatter("zero", 0, 0) == "zero (0 lines)"


@pytest.mark.parametrize(
    ("package_name", "expected_end"),
    [
        ("my_pkg", "src/my_pkg"),
        ("another_pkg", "src/another_pkg"),
        pytest.param("pytest", "site-packages/pytest", marks=pytest.mark.slow),
        ("non_existent_pkg", ""),  # Should not find
    ],
)
def test_find_package_path(
    dummy_pkg_path: Path, package_name: str, expected_end: str
) -> None:
    """Test find_package_path for local dummy and non-existent packages."""
    found_path = pmv.treemap.py_pkg.find_package_path(package_name).replace("\\", "/")
    if "site-packages" in expected_end:
        assert expected_end in found_path
    elif expected_end:
        expected_abs_path = str(dummy_pkg_path / package_name).replace("\\", "/")
        assert found_path.endswith(expected_abs_path)
    else:
        assert found_path == ""


def test_find_package_path_fallbacks(dummy_pkg_path: Path) -> None:
    """Test fallback mechanisms in find_package_path using mocking."""
    target_pkg = "my_pkg"
    expected_abs_path = str(dummy_pkg_path / target_pkg).replace("\\", "/")

    # When importlib.resources fails, fall back to scanning sys.path
    # (dummy_pkg_path was prepended to sys.path by the fixture)
    with patch("importlib.resources.files", side_effect=ImportError, create=True):
        found_path = pmv.treemap.py_pkg.find_package_path(target_pkg)
        assert found_path.replace("\\", "/") == expected_abs_path

    with (
        patch("importlib.resources.files", side_effect=ImportError, create=True),
        patch("site.getsitepackages", return_value=[str(dummy_pkg_path.parent)]),
        patch("os.path.isdir") as mock_isdir,
    ):
        expected_site_path = str(dummy_pkg_path.parent / target_pkg)
        mock_isdir.side_effect = lambda p: p == expected_site_path
        found_path = pmv.treemap.py_pkg.find_package_path(target_pkg)
        assert found_path.replace("\\", "/") == expected_site_path.replace("\\", "/")


@pytest.mark.parametrize(
    ("cell_text_fn", "expected_in_labels", "expected_not_in_labels"),
    [
        (True, [" lines", "%)", "my_pkg (", "another_pkg ("], []),  # Default formatter
        (False, ["my_pkg", "another_pkg"], ["(", " lines"]),  # No formatting
        (  # Custom formatter
            lambda pkg, count, total: f"PKG: {pkg} [{count}/{total}]",
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
    labels, parents = np.array(fig.data[0].labels), np.array(fig.data[0].parents)

    for expected in expected_in_labels:
        assert any(expected in str(lbl) for lbl in labels)

    pkg_labels = labels[parents == ""]
    assert len(pkg_labels) == 2
    for not_expected in expected_not_in_labels:
        assert not any(not_expected in str(lbl) for lbl in pkg_labels)


@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        (
            dict(cell_size_fn=lambda _cell: 0),
            "No source files found .* after filtering by cell_size_fn",
        ),
        (dict(show_counts="invalid_value"), "Invalid show_counts='invalid_value'"),
        (dict(group_by="package"), "Invalid group_by='package', must be one of"),
        (
            dict(color_by="bogus_col"),
            r"Invalid color_by='bogus_col', must be a dict.*'line_count'",
        ),
        (dict(color_by=""), "Invalid color_by='', must be a dict"),
        (dict(color_by="coverage"), "Invalid color_by='coverage'"),
        (dict(color_by="file_path"), "Invalid color_by='file_path'"),  # non-numeric
        (dict(color_by={}), "color_by dict must map file paths to values"),
    ],
)
def test_py_pkg_treemap_errors(kwargs: dict[str, Any], expected_error: str) -> None:
    """Empty filtered data and invalid show_counts/group_by/color_by raise."""
    with pytest.raises(ValueError, match=expected_error):
        pmv.py_pkg_treemap("my_pkg", **kwargs)


@pytest.mark.parametrize(
    ("test_base_url", "expect_link"),
    [
        ("https://github.com/janosh/pymatviz/blob/main", True),
        (None, False),
    ],
)
def test_py_pkg_treemap_base_url(test_base_url: str | None, expect_link: bool) -> None:
    """Test base_url handling for links and hover info."""
    pkg_name = "my_pkg"
    show_counts: Final = "value+percent"

    fig = pmv.py_pkg_treemap(pkg_name, base_url=test_base_url, show_counts=show_counts)
    trace = fig.data[0]

    expected_texttemplate = (
        "<a href='%{customdata[3]}' target='_blank'>%{customdata[2]}</a><br>"
        "%{value:,}<br>%{percentEntry}"
        if expect_link
        else "%{label}<br>%{value:,}<br>%{percentEntry}"
    )
    assert trace.texttemplate == expected_texttemplate
    assert trace.textinfo == "none"

    custom_data = trace.customdata
    assert custom_data is not None
    assert custom_data.shape[1] == 12

    file_urls = custom_data[:, 3]  # Index 3 is file_url
    if expect_link:
        assert all(
            str(test_base_url) in str(file_urls[idx])
            for idx, id_val in enumerate(trace.ids)
            if ".py" in id_val and file_urls[idx] is not None
        )
    else:
        assert not any(
            isinstance(url, str) and url.startswith("http") for url in file_urls
        )

    hovertemplate = trace.hovertemplate
    assert hovertemplate is not None
    assert "<b>%{customdata[2]}</b>" in hovertemplate
    assert "Path: %{customdata[1]}<br>" in hovertemplate
    assert "Lines: %{customdata[11]:,}<br>" in hovertemplate


def test_collect_package_modules_not_found() -> None:
    """Unknown packages raise instead of being silently skipped."""
    with pytest.raises(
        ValueError, match="Could not find package path for 'non_existent_package_12345'"
    ):
        pmv.treemap.py_pkg.collect_package_modules(
            ["my_pkg", "non_existent_package_12345"]
        )


def test_internal_imports(dummy_pkg_path: Path) -> None:
    """Test counting of internal imports using AST analysis."""
    pkg_name = "my_pkg"
    (dummy_pkg_path / pkg_name / "internal_user.py").write_text(
        "from .submodule import module2 # internal import from sibling dir\n"
        "from . import module1 # internal import from same dir\n"
        "import my_pkg.submodule.module2 # absolute internal import\n"
        "from my_pkg import module1 # absolute internal import\n"
        "import os # external import\n\n"
        "class InternalUser:\n    pass\n"
    )

    fig = pmv.py_pkg_treemap(pkg_name)
    hovertemplate = fig.data[0].hovertemplate
    assert hovertemplate is not None

    for hover_field in (
        "Internal Imports: %{customdata[7]:,}<br>",
        "External Imports: %{customdata[8]:,}<br>",
        "Classes: %{customdata[4]:,}<br>",
    ):
        assert hover_field in hovertemplate

    ids = fig.data[0].ids
    custom_data = fig.data[0].customdata
    internal_user_idx = next(
        (idx for idx, id_val in enumerate(ids) if id_val.endswith("/internal_user")), -1
    )
    assert internal_user_idx != -1

    assert [int(custom_data[internal_user_idx, idx]) for idx in (7, 8, 4)] == [4, 1, 1]


def test_top_level_module(dummy_pkg_path: Path) -> None:
    """Test handling of Python files directly under the package root."""
    (dummy_pkg_path / "my_pkg" / "top_level_mod.py").write_text("print('top level')")

    fig = pmv.py_pkg_treemap("my_pkg", group_by="module")
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
        # Non-GitHub
        ({"Project-URL": ["Source Code, https://gitlab.com/ase/ase"]}, False),
        # GitHub homepage
        ({"Home-page": "https://github.com/janosh/pymatviz"}, True),
        # GitHub project URL
        ({"Project-URL": ["Repository, https://github.com/janosh/matterviz"]}, True),
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
    # Ensure membership and indexing mirror dict semantics
    mock_metadata.__contains__.side_effect = metadata_dict.__contains__
    mock_metadata.__getitem__.side_effect = metadata_dict.__getitem__

    monkeypatch.setattr("importlib.metadata.metadata", lambda _: mock_metadata)

    fig = pmv.py_pkg_treemap(pkg_name, base_url="auto")
    custom_data = fig.data[0].customdata

    if expected_url_in_customdata:
        assert custom_data is not None
        assert custom_data.shape[1] == 12
        file_urls = custom_data[:, 3]
        assert any(
            url and "github.com/" in url and "/blob/main/" in url
            for url in file_urls
            if isinstance(url, str)
        ), f"{file_urls=} did not contain a GitHub URL"
    elif custom_data is not None:
        assert not any(
            url and "github.com" in url
            for url in custom_data[:, 3]
            if isinstance(url, str)
        )


def test_auto_base_url_metadata_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected metadata lookup errors propagate for base_url='auto'."""

    def mock_metadata_raises(_name: str) -> None:
        raise RuntimeError("Metadata fetch failed")

    monkeypatch.setattr("importlib.metadata.metadata", mock_metadata_raises)

    with pytest.raises(RuntimeError, match="Metadata fetch failed"):
        pmv.py_pkg_treemap("my_pkg", base_url="auto")


@pytest.mark.parametrize(
    ("show_counts_value", "expected_textinfo", "has_template"),
    [
        ("value", "none", True),  # Uses custom template for comma formatting
        ("percent", "label+percent entry", False),  # Uses built-in textinfo
        (False, "label", False),  # Uses built-in textinfo
    ],
)
def test_show_counts_no_base_url(
    show_counts_value: ShowCounts, expected_textinfo: str, has_template: bool
) -> None:
    """Test different show_counts values when base_url is None."""
    fig = pmv.py_pkg_treemap("my_pkg", base_url=None, show_counts=show_counts_value)
    assert fig.data[0].textinfo == expected_textinfo
    if has_template:
        assert fig.data[0].texttemplate is not None
    else:
        assert fig.data[0].texttemplate is None


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
            "Methods: %{customdata[6]:,}<br>"
            "Internal Imports: %{customdata[7]:,}<br>"
            "External Imports: %{customdata[8]:,}<br>"
            "Type Imports: %{customdata[9]:,}<br>"
            "<extra></extra>"
        )

        fig = pmv.py_pkg_treemap(packages=["pkg1", "pkg2"], base_url=None)

        assert fig.data[0].hovertemplate == expected_hovertemplate
        custom_data = mock_px_treemap.call_args.kwargs["custom_data"]
        mod_a = custom_data[custom_data["leaf_label"] == "modA"].iloc[0]
        assert mod_a["percent_of_package_cell_value"] == pytest.approx(100 / 300)


def test_py_pkg_treemap_wraps_long_underscore_labels(
    mock_pkg_data: pd.DataFrame,
) -> None:
    """Long underscore labels are wrapped near the middle underscore."""
    mock_collect_df = mock_pkg_data.iloc[:1].copy()
    long_label = "very_long_module_name"
    mock_collect_df["leaf_label"] = long_label
    mock_collect_df["filename"] = long_label
    mock_collect_df["top_module"] = long_label
    mock_collect_df["full_module"] = f"pkg1.{long_label}"
    mock_collect_df["repo_path_segment"] = f"pkg1/{long_label}.py"
    mock_collect_df["module_parts"] = [[long_label]]

    with patch(
        "pymatviz.treemap.py_pkg.collect_package_modules",
        return_value=mock_collect_df,
    ):
        fig = pmv.py_pkg_treemap(packages="pkg1", base_url=None)

    assert "very_long_<br>module_name" in fig.data[0].labels


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
    ("calculator", "expected_filenames"),
    [
        (None, {"module1", "module2", "module3", "module4_typed", "main"}),
        (_calc_sum_functions_classes, {"module1", "module2", "module4_typed"}),
        (
            _calc_methods_x_10_plus_lines,
            {"module1", "module2", "module3", "module4_typed", "main"},
        ),
        (lambda m: m.n_type_checking_imports + m.n_internal_imports, {"module4_typed"}),
        (_sizer_omit_specific, {"module2", "module3", "module4_typed"}),
    ],
    ids=["default", "func+class", "methods*10+lines", "imports", "omit_specific"],
)
def test_py_pkg_treemap_cell_size_fn(
    calculator: pmv.treemap.py_pkg.CellSizeFn | None,
    expected_filenames: set[str],
) -> None:
    """Test the cell_size_fn argument in py_pkg_treemap."""
    with patch("plotly.express.treemap") as mock_px_treemap:
        pmv.py_pkg_treemap(
            ["my_pkg", "another_pkg"], cell_size_fn=calculator, group_by="file"
        )

    df_passed_to_plotly: pd.DataFrame = mock_px_treemap.call_args.kwargs["data_frame"]

    assert "cell_value" in df_passed_to_plotly
    assert set(df_passed_to_plotly["filename"]) >= expected_filenames

    numeric_fields = {
        "line_count",
        "depth",
        "n_classes",
        "n_functions",
        "n_methods",
        "n_internal_imports",
        "n_external_imports",
        "n_type_checking_imports",
    }

    for _, row in df_passed_to_plotly.iterrows():
        metrics_instance = pmv.treemap.py_pkg.ModuleStats._make(
            0 if field in numeric_fields and pd.isna(row[field]) else row[field]
            for field in pmv.treemap.py_pkg.ModuleStats._fields
        )
        expected_val = (
            metrics_instance.line_count
            if calculator is None
            else calculator(metrics_instance)
        )

        assert expected_val > 0
        assert int(row["cell_value"]) == expected_val


def test_normalize_package_input() -> None:
    """Test _normalize_package_input function with various inputs."""
    assert pmv.treemap.py_pkg._normalize_package_input("numpy") == "numpy"

    assert pmv.treemap.py_pkg._normalize_package_input(np) == "numpy"

    import numpy.random

    assert pmv.treemap.py_pkg._normalize_package_input(numpy.random) == "numpy"

    class MockObj:
        pass

    mock_obj = MockObj()
    assert pmv.treemap.py_pkg._normalize_package_input(mock_obj) == str(mock_obj)


def _write_package_files(pkg_dir: Path, files: dict[str, int]) -> None:
    """Create a package tree with files containing the requested line counts."""
    pkg_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, n_lines in files.items():
        file_path = pkg_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        for parent in [pkg_dir, *file_path.parent.parents]:
            if parent == pkg_dir or pkg_dir in parent.parents:
                (parent / "__init__.py").touch()
        _write_py_lines(file_path, n_lines)


@pytest.mark.parametrize("min_lines", [0, 20])
@pytest.mark.parametrize("equal_area", [False, True])
def test_mapped_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, min_lines: int, equal_area: bool
) -> None:
    """Mapped colors respect exact paths, labels, scale, and area-weighted parents."""
    pkg_dir = tmp_path / "src" / "color_pkg"
    _write_package_files(
        pkg_dir, {"longer/mod.py": 4, "longer/other.py": 16, "shorter/mod.py": 3}
    )
    monkeypatch.setattr(
        pmv.treemap.py_pkg, "find_package_path", lambda _name: str(pkg_dir)
    )
    fig = pmv.py_pkg_treemap(
        "color_pkg",
        color_by={
            str(pkg_dir / "longer/mod.py"): 100,
            "src/color_pkg/longer/mod.py": 20,  # absolute path takes precedence
            "src/color_pkg/longer/other.py": 50,
            "unrelated/mod.py": 90,  # no filename guessing for shorter/mod.py
        },
        cell_text_fn=False,
        cell_size_fn=(lambda _cell: 1) if equal_area else None,
        min_lines_for_split=min_lines,
        color="line_count",  # color_by controls the cells
        labels={"color_value": "Coverage (%)"},
        color_range=(0, 100),
    )
    trace = fig.data[0]
    colors = dict(zip(trace.ids, trace.marker.colors, strict=True))
    assert colors["color_pkg/shorter/mod"] == 0
    if not min_lines:
        assert colors["color_pkg/longer/mod"] == 100
        assert colors["color_pkg/longer/other"] == 50
    np.testing.assert_allclose(
        [colors["color_pkg/longer"], colors["color_pkg"]],
        [75, 50] if equal_area else [60, 1200 / 23],
        rtol=2e-15,
        atol=0,  # one weighted division per tree level
    )
    axis = fig.layout.coloraxis
    assert (axis.cmin, axis.cmid, axis.cmax) == (0, 50, 100)
    assert axis.colorbar.title.text == "Coverage (%)"
    assert "Coverage (%): %{color:" in trace.hovertemplate


@pytest.mark.parametrize(
    ("color_by", "title"),
    [({"other.py": 3}, "Color Value"), ("line_count", "Line Count")],
)
def test_colorbar_title(color_by: str | dict[str, float], title: str) -> None:
    """Column and mapping colors have descriptive default colorbar titles."""
    assert (
        pmv.py_pkg_treemap(
            "my_pkg", color_by=color_by
        ).layout.coloraxis.colorbar.title.text
        == title
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_color", "expected_width"),
    [
        ({}, None, None),
        ({"cell_border": {"color": "black", "width": 2}}, "black", 2),
        ({"cell_border": {}}, None, None),
        ({"color_by": "line_count"}, "white", 0.5),
        (
            {"color_by": "line_count", "cell_border": {"color": "red", "width": 3}},
            "red",
            3,
        ),
        ({"color_by": "line_count", "cell_border": {}}, None, None),
    ],
)
def test_py_pkg_treemap_cell_border(
    kwargs: dict[str, Any],
    expected_color: str | None,
    expected_width: float | None,
) -> None:
    """Test cell_border parameter functionality."""
    fig = pmv.py_pkg_treemap("my_pkg", **kwargs)
    marker_line = fig.data[0].marker.line
    assert marker_line.color == expected_color
    if expected_width is not None:
        assert marker_line.width == expected_width


# === collect_source_modules_from_dir ===


@pytest.fixture(scope="module")
def rust_like_source_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a Rust-like source directory structure."""
    src = tmp_path_factory.mktemp("src")
    (src / "lib.rs").write_text("pub mod analysis;\npub mod io;\n")
    for rel_path, content in {
        "analysis/mod.rs": "pub mod convex_hull;\n",
        "analysis/convex_hull.rs": (
            "// Convex hull implementation\n"
            "pub fn compute() -> Vec<f64> {\n    vec![1.0, 2.0]\n}\n"
        ),
        "io/mod.rs": "pub mod lmdb;\n",
        "io/lmdb/dataset.rs": "pub struct Dataset;\n",
    }.items():
        file_path = src / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    return src


def test_collect_source_modules_basic(rust_like_source_dir: Path) -> None:
    """collect_source_modules_from_dir finds all .rs files and builds hierarchy."""
    df_result = pmv.treemap.py_pkg.collect_source_modules_from_dir(
        str(rust_like_source_dir), ["my_crate"], file_extensions=(".rs",)
    )
    assert set(df_result.columns) >= {
        "package",
        "full_module",
        "filename",
        "line_count",
        "module_parts",
        "depth",
        "repo_path_segment",
    }
    assert (df_result["package"] == "my_crate").all()
    assert (df_result[["n_classes", "n_functions"]] == 0).all().all()
    assert set(df_result["filename"]) >= {"lib", "mod", "convex_hull", "dataset"}


@pytest.mark.parametrize(
    ("extra_files", "filename", "expected_module_parts"),
    [
        # Rust new-style module root (analysis.rs next to analysis/) is nested
        ({"analysis.rs": "// new-style mod root\n"}, "analysis", ["analysis"] * 2),
        ({"readme.md": "# Readme\n"}, "readme", None),  # wrong extension
        ({"target/debug.rs": "fn main() {}\n"}, "debug", None),  # ignored dir
    ],
    ids=["mod-root-collision", "extra-md", "ignored-target"],
)
def test_collect_source_modules_filters(
    tmp_path: Path,
    extra_files: dict[str, str],
    filename: str,
    expected_module_parts: list[str] | None,
) -> None:
    """Sibling dirs nest module roots; other extensions and ignored dirs are skipped."""
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis/mod.rs").write_text("pub mod convex_hull;\n")
    for rel_path, content in extra_files.items():
        (tmp_path / rel_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel_path).write_text(content)

    df_result = pmv.treemap.py_pkg.collect_source_modules_from_dir(
        str(tmp_path), ["proj"], file_extensions=(".rs",)
    )
    matches = df_result[df_result["filename"] == filename]
    if expected_module_parts is None:
        assert matches.empty
    else:
        assert matches.iloc[0]["module_parts"] == expected_module_parts


def test_collect_source_modules_nonexistent_dir() -> None:
    """FileNotFoundError raised for non-existent source_dir."""
    with pytest.raises(FileNotFoundError, match="Source directory not found"):
        pmv.treemap.py_pkg.collect_source_modules_from_dir(
            "/nonexistent/path", ["proj"]
        )


def test_collect_source_modules_empty_dir(tmp_path: Path) -> None:
    """Empty directory returns empty DataFrame."""
    empty = tmp_path / "empty_src"
    empty.mkdir()
    df_result = pmv.treemap.py_pkg.collect_source_modules_from_dir(
        str(empty), ["proj"], file_extensions=(".rs",)
    )
    assert df_result.empty


def test_collect_source_modules_mixed_extensions_comment_prefix(tmp_path: Path) -> None:
    """Mixed file extensions use per-file comment prefixes, not a union."""
    src = tmp_path / "mixed"
    src.mkdir()
    (src / "helper.py").write_text("# py comment\n// code in python\nx = 1\n")
    (src / "lib.rs").write_text("// rs comment\n# code in rust\nlet x = 1;\n")
    df_result = pmv.treemap.py_pkg.collect_source_modules_from_dir(
        str(src), ["proj"], file_extensions=(".py", ".rs")
    )
    assert df_result.set_index("filename")["line_count"].to_dict() == {
        "helper": 2,
        "lib": 2,
    }


# === py_pkg_treemap with source_dir integration ===


@pytest.fixture(scope="module")
def rust_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a minimal Rust project with source files."""
    base = tmp_path_factory.mktemp("rust_proj")
    src = base / "src"
    src.mkdir()
    for rel_path, content in {
        "lib.rs": "pub mod utils;\npub fn main() {}\n",
        "utils/mod.rs": "pub fn helper() {}\n",
        "utils/math.rs": (
            "// math utilities\npub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n"
        ),
    }.items():
        file_path = src / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    return src


@pytest.mark.parametrize(
    ("kwargs", "checks"),
    [
        ({"base_url": None}, {"labels": {"lib", "math"}, "no_analysis": True}),
        (
            {"color_by": {"src/lib.rs": 100}, "base_url": None},
            {"hover": "Color Value: %{color:,.2f}", "lib_color": 100},
        ),
        ({"base_url": "auto"}, {"no_urls": True}),
        ({"base_url": "https://github.com/test/repo/blob/main"}, {"has_urls": True}),
    ],
)
def test_treemap_source_dir_modes(
    rust_project: Path, kwargs: dict[str, Any], checks: dict[str, Any]
) -> None:
    """py_pkg_treemap source_dir mode handles labels, coverage, and URLs."""
    fig = pmv.py_pkg_treemap(
        "my_crate",
        source_dir=str(rust_project),
        file_extensions=(".rs",),
        **kwargs,
    )
    trace = fig.data[0]
    if labels := checks.get("labels"):
        assert labels <= set(trace.labels)
    if checks.get("no_analysis"):
        assert all(
            field not in trace.hovertemplate
            for field in ("Classes:", "Functions:", "Methods:")
        )
    if "lib_color" in checks:
        assert (
            trace.marker.colors[list(trace.labels).index("lib")] == checks["lib_color"]
        )
    if hover := checks.get("hover"):
        assert hover in trace.hovertemplate
    file_urls = [str(url) for url in trace.customdata[:, 3]]
    if checks.get("no_urls"):
        assert not any(url.startswith("http") for url in file_urls)
    if checks.get("has_urls"):
        assert any(kwargs["base_url"] in url for url in file_urls)


def test_treemap_source_dir_no_matching_files(tmp_path: Path) -> None:
    """source_dir with no matching file_extensions raises ValueError."""
    empty = tmp_path / "no_rs"
    empty.mkdir()
    (empty / "readme.md").write_text("# hello\n")
    with pytest.raises(ValueError, match="No source files found"):
        pmv.py_pkg_treemap("proj", source_dir=str(empty), file_extensions=(".rs",))


def _write_py_lines(path: Path, n_lines: int) -> None:
    """Write a Python file with n_lines of executable code."""
    path.write_text("\n".join(f"x_{idx} = {idx}" for idx in range(n_lines)))


@pytest.fixture(scope="module")
def deep_pkg_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a package with 3 levels of nesting for depth-pruning tests."""
    src_dir = tmp_path_factory.mktemp("deep_src")
    files = {
        "alpha/big/core.py": 500,
        "alpha/big/helpers.py": 300,
        "alpha/medium/utils.py": 400,
        "alpha/tiny/small.py": 50,
        "gamma/only.py": 200,
        **{f"beta/s{idx}.py": 100 for idx in range(6)},
    }
    _write_package_files(src_dir / "deep_pkg", files)
    return src_dir


@pytest.fixture(autouse=False)
def _deep_pkg_on_path(deep_pkg_path: Path) -> Generator[None, None, None]:
    """Temporarily add the deep package source directory to sys.path."""
    path_str = str(deep_pkg_path)
    sys.path.insert(0, path_str)
    yield
    sys.path.remove(path_str)


@pytest.mark.usefixtures("_deep_pkg_on_path")
class TestMaxModuleDepth:
    """Tests for the max_module_depth parameter."""

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_max_depth_limits_nesting(self, depth: int) -> None:
        """Deeper path components should not appear when depth is capped."""
        trace = pmv.py_pkg_treemap(
            "deep_pkg", max_module_depth=depth, base_url=None
        ).data[0]
        max_id_depth = max(
            id_val.count("/") for id_val in trace.ids if isinstance(id_val, str)
        )
        assert max_id_depth <= depth
        if depth == 1:
            leaf_labels = {
                label
                for label, parent in zip(trace.labels, trace.parents, strict=False)
                if parent
            }
            assert leaf_labels >= {"alpha", "beta", "gamma"}
            assert leaf_labels.isdisjoint({"big", "core", "helpers"})


@pytest.mark.usefixtures("_deep_pkg_on_path")
class TestAdaptivePruning:
    """Tests for min_lines_for_split and max_children_for_split."""

    @staticmethod
    def _trace(**kwargs: Any) -> go.Treemap:
        """Return a deep_pkg treemap trace for pruning tests."""
        return pmv.py_pkg_treemap("deep_pkg", base_url=None, **kwargs).data[0]

    @pytest.mark.parametrize(
        ("min_lines", "max_children"),
        [(300, None), (50, 3)],
    )
    def test_collapse_beta_children(
        self, min_lines: int, max_children: int | None
    ) -> None:
        """Beta's 6 children get collapsed."""
        trace = self._trace(
            max_module_depth=2,
            min_lines_for_split=min_lines,
            max_children_for_split=max_children,
        )
        assert "beta" in trace.labels
        assert set(trace.labels).isdisjoint({f"s{idx}" for idx in range(6)})

    @pytest.mark.parametrize(
        ("min_lines", "expected_present", "expected_absent"),
        [
            (200, {"big", "medium", "other"}, {"tiny"}),
            (600, {"big", "other"}, {"medium", "tiny"}),
        ],
    )
    def test_other_bucket_for_small_children(
        self, min_lines: int, expected_present: set[str], expected_absent: set[str]
    ) -> None:
        """Children below min_lines merge into an 'other' bucket."""
        labels = set(
            self._trace(
                max_module_depth=3,
                min_lines_for_split=min_lines,
                max_children_for_split=5,
            ).labels
        )
        assert expected_present <= labels
        assert labels.isdisjoint(expected_absent)

    def test_max_children_only_at_deepest_level(self) -> None:
        """max_children_for_split does not collapse intermediate levels."""
        labels = set(
            self._trace(
                max_module_depth=3,
                min_lines_for_split=50,
                max_children_for_split=2,
            ).labels
        )
        assert {"alpha", "big", "medium", "core", "helpers"} <= labels

    def test_aggregated_values_are_summed(self) -> None:
        """Collapsed groups have summed line counts."""
        trace = self._trace(max_module_depth=2, min_lines_for_split=500)
        assert list(trace.values)[list(trace.labels).index("beta")] == pytest.approx(
            600, abs=5
        )

    def test_min_lines_uses_line_count_not_cell_value(self) -> None:
        """min_lines_for_split must compare against line_count, not cell_value."""
        labels = set(
            self._trace(
                max_module_depth=3,
                min_lines_for_split=200,
                max_children_for_split=5,
                cell_size_fn=lambda _module: 1,
            ).labels
        )
        assert {"big", "medium"} <= labels

    @pytest.mark.parametrize(
        "base_url",
        [None, "https://github.com/test/repo/blob/main"],
        ids=["no_links", "with_links"],
    )
    def test_depth_limiting_omits_customdata_from_templates(
        self, base_url: str | None
    ) -> None:
        """Depth limiting skips customdata refs in text and hover templates."""
        trace = pmv.py_pkg_treemap(
            "deep_pkg", max_module_depth=2, color_by="line_count", base_url=base_url
        ).data[0]
        assert "customdata" not in (trace.texttemplate or "")
        assert "%{label}" in (trace.texttemplate or "")
        assert "customdata" not in (trace.hovertemplate or "")
        assert "%{label}" in (trace.hovertemplate or "")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_module_depth": -1},
            {"min_lines_for_split": -5},
            {"max_children_for_split": -2},
        ],
    )
    def test_negative_params_raise(self, kwargs: dict[str, int]) -> None:
        """Negative depth/pruning parameters raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            pmv.py_pkg_treemap("deep_pkg", **kwargs)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("scan_dir", [False, True])
def test_python_file_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scan_dir: bool
) -> None:
    """Each scanner reads a source once for lines, functions, methods, and imports."""
    code = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    import pandas as pd\n    import numpy as np\n"
        "import sys\n"
        "class MyClass:\n    def method1(self): pass\n"
        "    async def method2(self): pass\n"
        "def standalone(): pass\n"
    )
    pkg = tmp_path / "metrics_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    (py_file := pkg / "mod.py").write_text(code)
    monkeypatch.syspath_prepend(str(tmp_path))
    with patch("builtins.open", wraps=open) as file_open:
        if scan_dir:
            modules = pmv.treemap.py_pkg.collect_source_modules_from_dir(
                str(pkg), [pkg.name]
            )
        else:
            modules = pmv.treemap.py_pkg.collect_package_modules([pkg.name])
    assert sum(call.args[0] == str(py_file) for call in file_open.call_args_list) == 1
    expected = {
        "line_count": 9,
        "n_classes": 1,
        "n_functions": 1,
        "n_methods": 2,
        "n_internal_imports": 0,
        "n_external_imports": 2,  # typing + sys, NOT the TYPE_CHECKING ones
        "n_type_checking_imports": 2,
    }
    assert (
        modules.set_index("filename").loc["mod", list(expected)].to_dict() == expected
    )
