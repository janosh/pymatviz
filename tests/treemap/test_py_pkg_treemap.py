"""Unit tests for pymatviz.treemap.py_pkg.py."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import pymatviz as pmv
from pymatviz.treemap import py_pkg
from pymatviz.treemap.py_pkg import SOURCE_MAP_PALETTE, _viridis_hex


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from pymatviz.treemap.py_pkg import ModuleFormatter
    from pymatviz.widgets.treemap import TreemapWidget


def tree(widget: TreemapWidget) -> dict[str, dict[str, Any]]:
    """Map node IDs to nodes, adding each node's leaf sum as ``total``."""
    nodes: dict[str, dict[str, Any]] = {}

    def walk(node: dict[str, Any]) -> float:
        """Record the node and return its leaf sum."""
        children = node.get("children")
        total = sum(map(walk, children)) if children else node["value"]
        nodes[node["id"]] = node | {"total": total}
        return total

    for root in widget.data:
        walk(root)
    return nodes


def labels(widget: TreemapWidget) -> set[str]:
    """All node labels of a treemap."""
    return {node["label"] for node in tree(widget).values()}


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
    assert py_pkg.count_lines(str(test_file), prefixes) == expected_lines


def test_count_lines_non_existent(tmp_path: Path) -> None:
    """Test count_lines with a non-existent file."""
    assert py_pkg.count_lines(str(tmp_path / "non_existent.py")) == 0


def test_default_module_formatter() -> None:
    """Test the default module formatter."""
    assert (
        py_pkg.default_module_formatter("my_module", 1234, 5000)
        == "my_module (1,234 lines, 24.7%)"
    )
    assert (
        py_pkg.default_module_formatter("another.mod", 50, 100)
        == "another.mod (50 lines, 50.0%)"
    )
    assert py_pkg.default_module_formatter("zero", 0, 0) == "zero (0 lines)"


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
    found_path = py_pkg.find_package_path(package_name).replace("\\", "/")
    if "site-packages" in expected_end:
        assert expected_end in found_path
    elif expected_end:
        expected_abs_path = str(dummy_pkg_path / package_name).replace("\\", "/")
        assert found_path.endswith(expected_abs_path)
    else:
        assert found_path == ""


@pytest.mark.parametrize("import_error", [ModuleNotFoundError, ImportError])
def test_find_package_path_sys_path_fallback(
    dummy_pkg_path: Path, import_error: type[ImportError]
) -> None:
    """Scan sys.path (prepended with dummy_pkg_path) when importlib can't import,
    also for a package whose __init__ raises ImportError.
    """
    with patch("importlib.resources.files", side_effect=import_error):
        found_path = py_pkg.find_package_path("my_pkg")
    assert found_path.replace("\\", "/") == str(dummy_pkg_path / "my_pkg").replace(
        "\\", "/"
    )


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
    widget = pmv.py_pkg_treemap(["my_pkg", "another_pkg"], cell_text_fn=cell_text_fn)
    pkg_labels = [root["label"] for root in widget.data]
    assert [root["id"] for root in widget.data] == ["my_pkg", "another_pkg"]

    for expected in expected_in_labels:
        assert any(expected in lbl for lbl in pkg_labels)

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
    """Empty filtered data and invalid group_by/color_by raise."""
    with pytest.raises(ValueError, match=expected_error):
        pmv.py_pkg_treemap("my_pkg", **kwargs)


def test_collect_package_modules_not_found() -> None:
    """Unknown packages raise instead of being silently skipped."""
    with pytest.raises(
        ValueError, match="Could not find package path for 'non_existent_package_12345'"
    ):
        py_pkg.collect_package_modules(["my_pkg", "non_existent_package_12345"])


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

    df_modules = py_pkg.collect_package_modules([pkg_name])
    row = df_modules.set_index("filename").loc["internal_user"]
    stats = ("n_internal_imports", "n_external_imports", "n_classes")
    assert [int(row[stat]) for stat in stats] == [4, 1, 1]
    assert "my_pkg/internal_user" in tree(pmv.py_pkg_treemap(pkg_name))


def test_top_level_module(dummy_pkg_path: Path) -> None:
    """Test handling of Python files directly under the package root."""
    (dummy_pkg_path / "my_pkg" / "top_level_mod.py").write_text("print('top level')")

    nodes = tree(pmv.py_pkg_treemap("my_pkg", group_by="module"))
    assert nodes["my_pkg/top_level_mod"]["label"] == "top_level_mod"


def test_analyze_unicode_error(
    dummy_pkg_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Test graceful handling of UnicodeDecodeError during file analysis."""
    pkg_name = "my_pkg"
    bad_file_path = dummy_pkg_path / pkg_name / "bad_encoding.py"
    bad_file_path.write_bytes(b"invalid byte seq \xff here")

    widget = pmv.py_pkg_treemap(pkg_name)
    captured = capsys.readouterr()

    assert f"Warning: Could not analyze {bad_file_path}: 'utf-8' codec" in captured.out
    # Check that the bad file node does NOT exist because its cell_value will be 0
    assert not any("bad_encoding" in node_id for node_id in tree(widget))


def test_py_pkg_treemap_label_text() -> None:
    """Cells show label and compact line count by default (e.g. 54.3k), a single
    package's modules are the top level, other label_text values pass on.
    """
    widget = pmv.py_pkg_treemap("my_pkg")
    assert (widget.label_text, widget.value_format) == ("label+value", ".3~s")
    assert (widget.label_align, widget.padding_inner) == ("top-left", 2)
    assert "--treemap-label-color: white" in widget.style
    assert all(root["id"].startswith("my_pkg/") for root in widget.data)
    widget = pmv.py_pkg_treemap("my_pkg", label_text="label+value", max_depth=2)
    assert (widget.label_text, widget.max_depth) == ("label+value", 2)


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


def test_py_pkg_treemap_nested_nodes(mock_pkg_data: pd.DataFrame) -> None:
    """Module paths nest into slash-joined IDs with leaf line counts as values."""
    with patch(
        "pymatviz.treemap.py_pkg.collect_package_modules", return_value=mock_pkg_data
    ):
        widget = pmv.py_pkg_treemap(packages=["pkg1", "pkg2"], cell_text_fn=False)
    nodes = tree(widget)
    assert {node_id: node["total"] for node_id, node in nodes.items()} == {
        "pkg1": 300,
        "pkg1/modA": 100,
        "pkg1/modB": 50,
        "pkg1/modB/sub": 50,
        "pkg1/modC": 150,
        "pkg2": 200,
        "pkg2/modX": 200,
    }
    assert "value" not in nodes["pkg1/modB"]  # branches sum their leaves
    # without color_by: palette colors per top-level node by size, overlays below
    overlay = "rgba(255, 255, 255, 0.13)"
    assert {node_id: node["color"] for node_id, node in nodes.items()} == {
        "pkg1": SOURCE_MAP_PALETTE[0],
        "pkg2": SOURCE_MAP_PALETTE[1],
        **{node_id: overlay for node_id in nodes if "/" in node_id},
    }


def test_module_tree_rejects_leaf_and_branch() -> None:
    """A path ending at a node that other paths extend is reported, not dropped."""
    paths = np.array([["pkg", "mod", None], ["pkg", "mod", "sub"]], dtype=object)
    with pytest.raises(ValueError, match="'pkg/mod' is a leaf and a branch"):
        py_pkg._module_tree(paths, np.ones(2), None, {"pkg": "pkg"})


# Calculator functions for testing
def _calc_sum_functions_classes(metrics: py_pkg.ModuleStats) -> int:
    return metrics.n_functions + metrics.n_classes


def _calc_methods_x_10_plus_lines(metrics: py_pkg.ModuleStats) -> int:
    return metrics.n_methods * 10 + metrics.line_count


def _sizer_omit_specific(metrics: py_pkg.ModuleStats) -> int:
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
    calculator: py_pkg.CellSizeFn | None,
    expected_filenames: set[str],
) -> None:
    """Test the cell_size_fn argument in py_pkg_treemap."""
    packages = ["my_pkg", "another_pkg"]
    widget = pmv.py_pkg_treemap(
        packages, cell_size_fn=calculator, group_by="file", cell_text_fn=False
    )
    leaves = {
        node_id: node["value"]
        for node_id, node in tree(widget).items()
        if "children" not in node
    }
    assert {node_id.split("/")[-1] for node_id in leaves} >= expected_filenames

    expected: dict[str, float] = {}
    stats_fields = py_pkg.ModuleStats._fields
    for _, row in py_pkg.collect_package_modules(packages).iterrows():
        metrics = py_pkg.ModuleStats(**{key: row[key] for key in stats_fields})
        cell_value = metrics.line_count if calculator is None else calculator(metrics)
        if cell_value > 0:
            leaf_id = f"{metrics.package}/{metrics.filename}"
            expected[leaf_id] = expected.get(leaf_id, 0) + cell_value
    assert leaves == expected


def test_normalize_package_input() -> None:
    """Test _normalize_package_input function with various inputs."""
    assert py_pkg._normalize_package_input("numpy") == "numpy"

    assert py_pkg._normalize_package_input(np) == "numpy"

    import numpy.random

    assert py_pkg._normalize_package_input(numpy.random) == "numpy"


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
    monkeypatch.setattr(py_pkg, "find_package_path", lambda _name: str(pkg_dir))
    widget = pmv.py_pkg_treemap(
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
        color_range=(0, 100),
    )
    colors = {node_id: node["color"] for node_id, node in tree(widget).items()}
    expected: dict[str, float] = {"color_pkg/shorter/mod": 0, "color_pkg/shorter": 0}
    if not min_lines:
        expected |= {"color_pkg/longer/mod": 100, "color_pkg/longer/other": 50}
    expected["color_pkg/longer"] = 75 if equal_area else 60
    assert {key: colors[key] for key in expected} == {
        key: _viridis_hex(val / 100) for key, val in expected.items()
    }


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (-1, "#440154"),
        (0, "#440154"),
        (0.0625, "#461768"),
        (1, "#fde725"),
        (2, "#fde725"),
    ],
)
def test_viridis_hex(fraction: float, expected: str) -> None:
    """Fractions interpolate between viridis stops and clamp to [0, 1]."""
    assert _viridis_hex(fraction) == expected


def test_color_by_column_uses_data_range() -> None:
    """Without color_range, the extreme values map to the ends of the scale."""
    nodes = tree(pmv.py_pkg_treemap("my_pkg", color_by="line_count"))
    colors = {node["color"] for node in nodes.values()}
    assert {"#440154", "#fde725"} <= colors


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
    df_result = py_pkg.collect_source_modules_from_dir(
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

    df_result = py_pkg.collect_source_modules_from_dir(
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
        py_pkg.collect_source_modules_from_dir("/nonexistent/path", ["proj"])


def test_collect_source_modules_empty_dir(tmp_path: Path) -> None:
    """Empty directory returns empty DataFrame."""
    empty = tmp_path / "empty_src"
    empty.mkdir()
    df_result = py_pkg.collect_source_modules_from_dir(
        str(empty), ["proj"], file_extensions=(".rs",)
    )
    assert df_result.empty


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("from demo import helpers\nimport demo.sub\nimport os\n", (3, 2, 1)),
        ("x = " + "+".join(["1"] * 200_000) + "\n", (1, 0, 0)),  # RecursionError
        ("x = " + "-" * 100_000 + "1\n", (1, 0, 0)),  # parser stack overflow
        # scanning demo/subpkg: imports resolve against the enclosing demo package
        ("import demo.subpkg.helpers\nimport os\n", (2, 1, 1)),
    ],
    ids=["import-root", "recursion", "parser-overflow", "subpackage"],
)
def test_collect_source_modules_py_imports(
    tmp_path: Path, code: str, expected: tuple[int, int, int]
) -> None:
    """Imports classify by package dir, not display label; unparsable files count."""
    pkg = tmp_path / "demo"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    scan_dir = pkg
    if "subpkg" in code:
        scan_dir = pkg / "subpkg"
        scan_dir.mkdir()
        (scan_dir / "__init__.py").touch()
    (scan_dir / "mod.py").write_text(code)
    df_result = py_pkg.collect_source_modules_from_dir(str(scan_dir), ["Demo project"])
    cols = ["line_count", "n_internal_imports", "n_external_imports"]
    assert tuple(df_result.set_index("filename").loc["mod", cols]) == expected


def test_collect_source_modules_mixed_extensions_comment_prefix(tmp_path: Path) -> None:
    """Mixed file extensions use per-file comment prefixes, not a union."""
    src = tmp_path / "mixed"
    src.mkdir()
    (src / "helper.py").write_text("# py comment\n// code in python\nx = 1\n")
    (src / "lib.rs").write_text("// rs comment\n# code in rust\nlet x = 1;\n")
    df_result = py_pkg.collect_source_modules_from_dir(
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


@pytest.mark.parametrize("color_by", [None, {"src/lib.rs": 100}])
def test_treemap_source_dir(
    rust_project: Path, color_by: dict[str, float] | None
) -> None:
    """py_pkg_treemap source_dir mode scans non-Python files and colors them."""
    widget = pmv.py_pkg_treemap(
        "my_crate",
        source_dir=str(rust_project),
        file_extensions=(".rs",),
        color_by=color_by,
    )
    nodes = tree(widget)
    assert {"lib", "math"} <= {node["label"] for node in nodes.values()}
    lib_color = next(
        node.get("color") for node in nodes.values() if node["label"] == "lib"
    )
    assert (lib_color == "#fde725") if color_by else (lib_color in SOURCE_MAP_PALETTE)


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
        nodes = tree(pmv.py_pkg_treemap("deep_pkg", max_module_depth=depth))
        assert max(node_id.count("/") for node_id in nodes) <= depth
        if depth == 1:
            child_labels = {
                node["label"] for node_id, node in nodes.items() if "/" in node_id
            }
            assert child_labels >= {"alpha", "beta", "gamma"}
            assert child_labels.isdisjoint({"big", "core", "helpers"})


@pytest.mark.usefixtures("_deep_pkg_on_path")
class TestAdaptivePruning:
    """Tests for min_lines_for_split and max_children_for_split."""

    @staticmethod
    def _labels(**kwargs: Any) -> set[str]:
        """Node labels of a deep_pkg treemap for pruning tests."""
        return labels(pmv.py_pkg_treemap("deep_pkg", cell_text_fn=False, **kwargs))

    @pytest.mark.parametrize(
        ("min_lines", "max_children"),
        [(300, None), (50, 3)],
    )
    def test_collapse_beta_children(
        self, min_lines: int, max_children: int | None
    ) -> None:
        """Beta's 6 children get collapsed."""
        node_labels = self._labels(
            max_module_depth=2,
            min_lines_for_split=min_lines,
            max_children_for_split=max_children,
        )
        assert "beta" in node_labels
        assert node_labels.isdisjoint({f"s{idx}" for idx in range(6)})

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
        node_labels = self._labels(
            max_module_depth=3, min_lines_for_split=min_lines, max_children_for_split=5
        )
        assert expected_present <= node_labels
        assert node_labels.isdisjoint(expected_absent)

    def test_max_children_only_at_deepest_level(self) -> None:
        """max_children_for_split does not collapse intermediate levels."""
        node_labels = self._labels(
            max_module_depth=3, min_lines_for_split=50, max_children_for_split=2
        )
        assert {"alpha", "big", "medium", "core", "helpers"} <= node_labels

    def test_aggregated_values_are_summed(self) -> None:
        """Collapsed groups have summed line counts."""
        widget = pmv.py_pkg_treemap(
            "deep_pkg", max_module_depth=2, min_lines_for_split=500
        )
        assert tree(widget)["deep_pkg/beta"]["total"] == pytest.approx(600, abs=5)

    def test_min_lines_uses_line_count_not_cell_value(self) -> None:
        """min_lines_for_split must compare against line_count, not cell_value."""
        node_labels = self._labels(
            max_module_depth=3,
            min_lines_for_split=200,
            max_children_for_split=5,
            cell_size_fn=lambda _module: 1,
        )
        assert {"big", "medium"} <= node_labels

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
            modules = py_pkg.collect_source_modules_from_dir(str(pkg), [pkg.name])
        else:
            modules = py_pkg.collect_package_modules([pkg.name])
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


def test_other_bucket_distinct_from_other_module(tmp_path: Path) -> None:
    """A real module named "other" doesn't collide with the merged-modules bucket."""
    pkg_dir = tmp_path / "pkg"
    _write_package_files(
        pkg_dir,
        {"x/other/a.py": 600, "x/other/b.py": 600, "x/tiny.py": 5, "x/tiny2.py": 5},
    )
    nodes = tree(
        pmv.py_pkg_treemap(
            "pkg", source_dir=str(pkg_dir), cell_text_fn=False, min_lines_for_split=100
        )
    )
    assert nodes["pkg/x/other"]["total"] == 1200
    assert nodes["pkg/x/__other__"]["label"] == "other"
    assert nodes["pkg/x/__other__"]["value"] == 10


def test_color_by_uniform_values() -> None:
    """A zero-width color range maps every cell to the scale's middle."""
    nodes = tree(pmv.py_pkg_treemap("my_pkg", color_by="depth", color_range=(5, 5)))
    assert {node["color"] for node in nodes.values()} == {_viridis_hex(0.5)}
    assert not any("_color" in node for node in nodes.values())
