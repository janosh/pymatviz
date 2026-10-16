"""Hierarchical treemap visualizations of source code packages.

For visualizing module structure and code distribution within Python packages
or any language project via directory scanning.
"""

from __future__ import annotations

import ast
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, get_args

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pymatviz.typing import ShowCounts


if TYPE_CHECKING:
    from types import ModuleType

ModuleFormatter = Callable[[str, int, int], str]
type GroupBy = Literal["file", "directory", "module"]

_COMMENT_PREFIXES: dict[str, str] = {
    ".py": "#",
    ".pyi": "#",
    ".r": "#",
    ".jl": "#",
    ".rb": "#",
    ".sh": "#",
    ".rs": "//",
    ".ts": "//",
    ".tsx": "//",
    ".js": "//",
    ".jsx": "//",
    ".go": "//",
    ".c": "//",
    ".cpp": "//",
    ".h": "//",
    ".hpp": "//",
}


class ModuleStats(NamedTuple):
    """NamedTuple for holding module metrics, providing attribute access."""

    package: str
    full_module: str
    filename: str
    directory: str
    top_module: str
    line_count: int
    file_path: str
    repo_path_segment: str
    leaf_label: str
    module_parts: list[str]
    depth: int
    n_classes: int
    n_functions: int
    n_methods: int
    n_internal_imports: int
    n_external_imports: int
    n_type_checking_imports: int


type CellSizeFn = Callable[[ModuleStats], float | int]

# Python AST counts per module (zero for non-Python sources)
_ANALYSIS_COLS: tuple[str, ...] = (
    "n_classes",
    "n_functions",
    "n_methods",
    "n_internal_imports",
    "n_external_imports",
    "n_type_checking_imports",
)
# module-data columns usable as color_by (besides "coverage" and dict mappings)
_NUMERIC_COLOR_COLUMNS = ("line_count", "cell_value", "depth", *_ANALYSIS_COLS)


def default_module_formatter(module: str, count: int, total: int) -> str:
    """Default formatter showing module name, line count, and percentage on one line.

    Example output: my_package (1,234 Lines, 15.6%)
    """
    # Avoid ZeroDivisionError if total is 0
    percent_str = f", {count / total:.1%}" if total > 0 else ""
    return f"{module} ({count:,} lines{percent_str})"


def _normalize_package_input(package: str | Any) -> str:
    """Convert package input (string name or module object) to package name string."""
    if isinstance(package, str):
        return package
    if hasattr(package, "__name__"):
        return package.__name__.split(".")[0]
    return str(package)


def count_lines(file_path: str, comment_prefixes: tuple[str, ...] = ("#",)) -> int:
    """Count non-empty, non-comment lines in a source file.

    Args:
        file_path: Path to the source file.
        comment_prefixes: Comment prefix strings to skip (e.g. ``("#",)`` for
            Python, ``("//",)`` for Rust/C/JS). Defaults to Python-style.
    """
    try:
        with open(file_path, encoding="utf-8") as file_handle:
            return sum(
                bool(
                    (stripped := line.strip())
                    and not stripped.startswith(comment_prefixes)
                )
                for line in file_handle
            )
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return 0


def _github_base_url(package: str) -> str | None:
    """GitHub blob URL (assuming main branch) from installed package metadata.

    Checks the Home-page field first, then Project-URL entries. Returns None if
    the package isn't installed or lists no GitHub URL.
    """
    try:
        metadata = importlib.metadata.metadata(package)
    except importlib.metadata.PackageNotFoundError:
        return None

    github_url = None
    homepage = metadata.get("Home-page")  # ty: ignore[unresolved-attribute] https://github.com/astral-sh/ty/issues/1134
    if homepage and "github.com" in homepage:
        github_url = homepage
    else:
        for url_entry in metadata.get_all("Project-URL") or []:
            url_parts = url_entry.split(",", 1)
            if len(url_parts) == 2 and "github.com" in (url := url_parts[1].strip()):
                github_url = url
                break
    return f"{github_url.rstrip('/')}/blob/main" if github_url else None


def find_package_path(package_name: str) -> str:
    """Find the filesystem path for an installed Python package."""
    try:
        from importlib import resources

        package_root = resources.files(package_name)
        if package_root and os.path.isdir(str(package_root)):
            return str(package_root)
    except (ImportError, ModuleNotFoundError, TypeError):
        pass

    # Try sys.path for packages added to path (e.g. in tests)
    for path_entry in sys.path:
        if path_entry and os.path.isdir(package_path := f"{path_entry}/{package_name}"):
            return package_path

    # Try common locations if importlib approach fails
    current_dir = os.path.abspath(os.getcwd())
    possible_paths = [
        f"{current_dir}/{package_name}",
        f"{current_dir}/src/{package_name}",
    ]

    try:  # Also try site-packages
        import site

        site_packages = site.getsitepackages()[0]
        possible_paths.append(f"{site_packages}/{package_name}")
    except (ImportError, IndexError):
        pass

    for path in possible_paths:
        if os.path.isdir(path):
            return path

    return ""  # If we can't find package, return empty string


def _analyze_py_file(file_path: str, package_name: str) -> dict[str, int]:
    """Parse Python file using AST to count classes, functions, and imports."""
    counts = dict.fromkeys(_ANALYSIS_COLS, 0)
    try:
        with open(file_path, encoding="utf-8") as file_handle:
            source = file_handle.read()
        tree = ast.parse(source, filename=file_path)

        # Pre-collect nodes needing special treatment since ast.walk is a flat
        # iterator that would otherwise double-count them: imports inside
        # `if TYPE_CHECKING:` blocks and function defs in class bodies (methods)
        type_checking_imports: set[int] = set()
        method_nodes: set[int] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            ):
                for sub_node in node.body:
                    if isinstance(sub_node, (ast.Import, ast.ImportFrom)):
                        type_checking_imports.add(id(sub_node))
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_nodes.add(id(item))

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                counts["n_classes"] += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if id(node) in method_nodes:
                    counts["n_methods"] += 1
                else:
                    counts["n_functions"] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if id(node) in type_checking_imports:
                    counts["n_type_checking_imports"] += 1
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] == package_name:
                            counts["n_internal_imports"] += 1
                        else:
                            counts["n_external_imports"] += 1
                elif node.level > 0:  # relative import
                    counts["n_internal_imports"] += 1
                elif node.module:
                    if node.module.split(".")[0] == package_name:
                        counts["n_internal_imports"] += 1
                    else:
                        counts["n_external_imports"] += 1

    except (FileNotFoundError, SyntaxError, PermissionError, UnicodeDecodeError) as exc:
        print(f"Warning: Could not analyze {file_path}: {exc}")  # noqa: T201
    return counts


def _parse_lcov_lines(lines: Iterable[str]) -> dict[str, float]:
    """Parse lcov-format lines into a path-to-percentage map.

    Args:
        lines: Iterable of raw lcov lines (may include trailing newlines).

    Returns:
        dict mapping file paths to coverage percentages (0-100).
    """
    coverage_map: dict[str, float] = {}
    current_file: str | None = None
    lines_hit = 0
    lines_found = 0

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("SF:"):
            current_file = line[3:]
            lines_hit = 0
            lines_found = 0
        elif line.startswith("LH:"):
            lines_hit = int(line[3:])
        elif line.startswith("LF:"):
            lines_found = int(line[3:])
        elif line == "end_of_record":
            if current_file and lines_found > 0:
                coverage_map[current_file] = (lines_hit / lines_found) * 100
            current_file = None

    return coverage_map


def parse_lcov_file(lcov_path: str) -> dict[str, float]:
    """Parse an lcov/info coverage file into a path-to-percentage map.

    Handles output from cargo-llvm-cov, lcov, gcov, geninfo, and similar tools
    that produce the lcov trace-file format.

    Args:
        lcov_path: Path to an lcov-format coverage file (typically .info or .lcov).

    Returns:
        dict mapping file paths to coverage percentages (0-100).
    """
    with open(lcov_path, encoding="utf-8") as fh:
        return _parse_lcov_lines(fh)


def _coverage_json_to_map(
    coverage_data: dict[str, Any], base_dir: str | None = None
) -> dict[str, float]:
    """Extract per-file percent_covered from Python coverage JSON.

    Relative paths are resolved against base_dir when given.
    """
    coverage_map: dict[str, float] = {}
    for file_path, file_data in coverage_data.get("files", {}).items():
        summary = file_data.get("summary", {})
        if "percent_covered" in summary:
            path = file_path
            if base_dir is not None:
                path = os.path.normpath(f"{base_dir}/{file_path}")
            coverage_map[path] = summary["percent_covered"]
    return coverage_map


def collect_coverage_data(coverage_data_file: str | None = None) -> dict[str, float]:
    """Collect coverage data from various formats.

    Supports Python coverage JSON (from ``coverage json``), lcov/info files
    (from cargo-llvm-cov, lcov, gcov, etc.), and URLs to either format.
    Format is auto-detected from file extension (.info/.lcov -> lcov, else JSON).

    When coverage_data_file is not provided, attempts to run the ``coverage``
    CLI to generate Python coverage data on the fly.

    Args:
        coverage_data_file: Path to coverage data file or URL. Supported formats:

            - Python coverage JSON (from ``coverage json``)
            - lcov/info files (.info or .lcov extension)

            If provided as a URL, fetches from the remote location (HTTPS only).
            If None, attempts auto-generation via the ``coverage`` CLI.

    Returns:
        dict[str, float]: Mapping from file paths to coverage percentages (0-100).
    """
    if coverage_data_file:
        is_url = coverage_data_file.startswith("https://")
        is_lcov = coverage_data_file.endswith((".info", ".lcov"))

        # lcov format: parse directly (or download and parse in memory for URLs)
        if is_lcov:
            if is_url:
                with urllib.request.urlopen(  # noqa: S310
                    coverage_data_file, timeout=15
                ) as response:
                    return _parse_lcov_lines(
                        response.read().decode("utf-8").splitlines()
                    )
            return parse_lcov_file(coverage_data_file)

        # JSON format (Python coverage json)
        try:
            if is_url:
                # Security: Only HTTPS URLs allowed to prevent file:// and other schemes
                with urllib.request.urlopen(coverage_data_file, timeout=15) as response:  # noqa: S310
                    coverage_data = json.load(response)
            else:
                with open(coverage_data_file, encoding="utf-8") as file_handle:
                    coverage_data = json.load(file_handle)
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            if is_url:
                exc.add_note(
                    f"Failed to fetch coverage data from URL: {coverage_data_file}"
                )
            raise

        # Convert relative paths to absolute paths for local files
        base_dir = (
            None if is_url else os.path.dirname(os.path.abspath(coverage_data_file))
        )
        return _coverage_json_to_map(coverage_data, base_dir)

    # No coverage file given: generate Python coverage JSON on the fly
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_cov_file:
        tmp_json_path = tmp_cov_file.name
    try:
        result = subprocess.run(  # noqa: S603
            ["coverage", "json", "-o", tmp_json_path],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"`coverage json` failed with exit code {result.returncode}, pass "
                f"coverage_data_file explicitly. stderr: {result.stderr.strip()}"
            )
        with open(tmp_json_path, encoding="utf-8") as file_handle:
            return _coverage_json_to_map(json.load(file_handle))
    finally:
        if os.path.isfile(tmp_json_path):
            os.unlink(tmp_json_path)


def collect_package_modules(
    package_names: Sequence[str | ModuleType],
    ignored_dirs: tuple[str, ...] = (".venv", ".git", "node_modules"),
) -> pd.DataFrame:
    """Collect information about all modules in the given packages.

    Args:
        package_names (Sequence[str | ModuleType]): Names of packages to analyze or
            module objects.
        ignored_dirs (tuple[str, ...]): Directories to ignore when scanning packages.

    Returns:
        pd.DataFrame: with package structure info
    """
    all_modules = []

    for pkg_input in package_names:
        package_name = _normalize_package_input(pkg_input)
        pkg_path = find_package_path(package_name)
        if not pkg_path:
            raise ValueError(f"Could not find package path for {package_name!r}")

        # Repo path prefix: pkg_name dir at repo root or src-layout (.../src/pkg_name)
        path_parts = pkg_path.split(os.sep)
        is_src_layout = (
            len(path_parts) >= 2
            and path_parts[-2] == "src"
            and path_parts[-1] == package_name
        )
        repo_path_prefix = f"src/{package_name}" if is_src_layout else package_name

        python_files = []
        for root, dirs, files in os.walk(pkg_path):
            dirs[:] = [folder for folder in dirs if folder not in ignored_dirs]
            python_files += [
                os.path.normpath(f"{root}/{filename}")
                for filename in files
                if filename.endswith(".py") and not filename.startswith("_")
            ]

        for file_path in python_files:
            rel_path = os.path.relpath(file_path, os.path.dirname(pkg_path))
            module_name = (
                rel_path.replace(".py", "").replace(os.sep, ".").replace("\\", ".")
            )
            # module path components, e.g. pymatviz.utils.io -> ['utils', 'io']
            relative_module_name = module_name.removeprefix(f"{package_name}.")
            module_parts = (
                relative_module_name.split(".") if relative_module_name else []
            )
            directory_parts = rel_path.split(os.sep)
            filename = os.path.basename(rel_path).replace(".py", "")
            pkg_rel_path = os.path.relpath(file_path, pkg_path).replace(os.sep, "/")

            all_modules.append(
                {
                    "package": package_name,
                    "full_module": module_name,
                    "filename": filename,
                    "directory": directory_parts[0]
                    if len(directory_parts) > 1
                    else "root",
                    "top_module": module_parts[0] if module_parts else "root",
                    "line_count": count_lines(file_path),
                    "file_path": file_path,
                    "repo_path_segment": f"{repo_path_prefix}/{pkg_rel_path}",
                    "leaf_label": module_parts[-1]  # link text
                    if module_parts
                    else (filename or package_name),
                    "module_parts": module_parts,
                    "depth": len(module_parts),
                    **_analyze_py_file(file_path, package_name),
                }
            )

    return pd.DataFrame(all_modules)


def collect_source_modules_from_dir(
    source_dir: str,
    project_names: list[str],
    file_extensions: tuple[str, ...] = (".py",),
    ignored_dirs: tuple[str, ...] = (
        ".git",
        "target",
        "node_modules",
        ".venv",
        "__pycache__",
        "build",
        "dist",
    ),
) -> pd.DataFrame:
    """Collect information about source files in a directory tree.

    Language-agnostic alternative to :func:`collect_package_modules`. Walks
    the given source directory and builds a module hierarchy from the file
    system structure. Works for any language (Rust, TypeScript, Go, ...).

    Args:
        source_dir: Root directory to scan for source files.
        project_names: Display name(s) for the project. Files are attributed
            to the first name.
        file_extensions: File extensions to include, e.g. ``(".rs",)`` for
            Rust or ``(".ts", ".tsx")`` for TypeScript.
        ignored_dirs: Directory names to skip during traversal.

    Returns:
        pd.DataFrame with the same schema as :func:`collect_package_modules`.
    """
    source_dir = os.path.abspath(source_dir)
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    project_name = project_names[0]
    source_dir_name = os.path.basename(source_dir)

    ignored = set(ignored_dirs)
    all_modules: list[dict[str, Any]] = []
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = sorted(d for d in dirs if d not in ignored)

        for filename in sorted(files):
            ext = os.path.splitext(filename)[1]
            if ext not in file_extensions:
                continue

            file_path = os.path.normpath(f"{root}/{filename}")
            comment_prefix = (_COMMENT_PREFIXES.get(ext, "#"),)
            line_count = count_lines(file_path, comment_prefix)

            rel_path = os.path.relpath(file_path, source_dir)
            rel_parts = rel_path.replace(os.sep, "/").split("/")
            stem = os.path.splitext(rel_parts[-1])[0]

            # Every file gets a unique leaf in the hierarchy.
            # When a file shares its stem with a sibling directory (Rust
            # new-style module roots like burnside.rs + burnside/), nest it
            # inside the directory namespace to avoid non-leaf collisions.
            sibling_dir = f"{root}/{stem}"
            if os.path.isdir(sibling_dir):
                module_parts = [*rel_parts[:-1], stem, stem]
            else:
                module_parts = [*rel_parts[:-1], stem]

            full_module = f"{project_name}.{'.'.join(module_parts)}"
            top_module = module_parts[0]
            directory = rel_parts[0] if len(rel_parts) > 1 else "root"
            leaf_label = module_parts[-1]
            repo_path_segment = f"{source_dir_name}/{rel_path.replace(os.sep, '/')}"

            all_modules.append(
                {
                    "package": project_name,
                    "full_module": full_module,
                    "filename": stem,
                    "directory": directory,
                    "top_module": top_module,
                    "line_count": line_count,
                    "file_path": file_path,
                    "repo_path_segment": repo_path_segment,
                    "leaf_label": leaf_label,
                    "module_parts": module_parts,
                    "depth": len(module_parts),
                    **dict.fromkeys(_ANALYSIS_COLS, 0),
                }
            )

    return pd.DataFrame(all_modules)


def _match_coverage_path(
    file_path: str,
    coverage_map: dict[str, float],
    df_modules: pd.DataFrame,
    package_names: list[str],
) -> float:
    """Match a file path to coverage data using simple strategies.

    Args:
        file_path: Path to the file to match
        coverage_map: Mapping from file paths to coverage percentages
        df_modules: DataFrame with module data including repo_path_segment
        package_names: List of package names to match against

    Returns:
        float: Coverage percentage for the file, or NaN if no match found
    """
    if file_path in coverage_map:  # Try exact match first
        return coverage_map[file_path]

    row_data = df_modules[df_modules["file_path"] == file_path]
    if row_data.empty:
        return float("nan")

    row = row_data.iloc[0]
    repo_path_segment = row["repo_path_segment"]

    # Try matching by repo_path_segment
    for cov_path, cov_value in coverage_map.items():
        cov_path_unix = cov_path.replace(os.sep, "/")
        for pkg_name in package_names:
            pkg_idx = cov_path_unix.find(pkg_name)
            if pkg_idx != -1:
                cov_rel_path = cov_path_unix[pkg_idx:]
                # Direct match or with src/ prefix
                if cov_rel_path in (repo_path_segment, f"src/{repo_path_segment}"):
                    return cov_value

    # Fallback: match by filename
    filename = os.path.basename(file_path)
    matches = [
        (file_path, cov_val)
        for file_path, cov_val in coverage_map.items()
        if os.path.basename(file_path) == filename
    ]

    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        # Find best match by path similarity
        def path_similarity(match_path: str) -> int:
            """Calculate path similarity."""
            try:
                return len(os.path.commonpath([file_path, match_path]))
            except ValueError:
                return len(os.path.basename(match_path))

        best_match = max(matches, key=lambda x: path_similarity(x[0]))
        return best_match[1]

    return float("nan")


def _apply_coverage_weighted_averages(fig: go.Figure, df_treemap: pd.DataFrame) -> None:
    """Apply weighted average coverage calculations to parent nodes in treemap."""
    trace = fig.data[0]
    labels, parents, values = trace.labels, trace.parents, trace.values  # noqa: PD011
    label_index = {label: idx for idx, label in enumerate(labels)}

    # Create a mapping from treemap labels to DataFrame rows
    label_to_row = {}
    for _, row in df_treemap.iterrows():
        # Try to match by various criteria
        for label in labels:
            if (
                # Match by exact line count and module name
                (
                    abs(values[label_index[label]] - row["line_count"]) < 1
                    and row["leaf_label"] in label
                )
                or
                # Match by package name for root nodes
                (row["package_name_raw"] in label and "lines" in label)
                or
                # Match by full module path
                (row["full_module"] in label)
            ):
                label_to_row[label] = row
                break

    # Map leaf nodes to their coverage values
    node_coverage = {label: row["color_value"] for label, row in label_to_row.items()}

    # Calculate weighted averages for parent nodes
    colors = list(trace.marker.colors)
    parent_children: dict[str, list[tuple[str, float]]] = {}
    for idx, parent in enumerate(parents):
        if parent:
            parent_children.setdefault(parent, []).append(labels[idx])

    # Process parent nodes to calculate weighted averages
    for idx, label in enumerate(labels):
        if label not in node_coverage and label in parent_children:
            children_data = []
            for child in parent_children[label]:
                if child in node_coverage:
                    child_idx = label_index[child]
                    child_weight = values[child_idx]
                    child_coverage = node_coverage[child]
                    children_data.append((child_coverage, child_weight))

            if children_data:
                total_weight = sum(weight for _, weight in children_data)
                if total_weight > 0:
                    weighted_avg = (
                        sum(cov * weight for cov, weight in children_data)
                        / total_weight
                    )
                    node_coverage[label] = weighted_avg
                    colors[idx] = weighted_avg

    fig.data[0].marker.colors = tuple(colors)


def py_pkg_treemap(
    packages: str | ModuleType | Sequence[str | ModuleType],
    *,
    source_dir: str | None = None,
    file_extensions: tuple[str, ...] = (".py",),
    base_url: str | None = "auto",
    show_counts: ShowCounts = "value+percent",
    cell_text_fn: ModuleFormatter | bool = default_module_formatter,
    group_by: GroupBy = "module",
    max_module_depth: int | None = None,
    min_lines_for_split: int = 0,
    max_children_for_split: int | None = None,
    cell_size_fn: CellSizeFn | None = None,
    color_by: str | dict[str, float] | None = None,
    coverage_data_file: str | None = None,
    color_range: tuple[float, float] | None = None,
    cell_border: dict[str, Any] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Generate a treemap plot showing module structure of source code packages.

    Works with any language when ``source_dir`` is provided. Without it,
    defaults to Python package introspection via importlib.

    The first level shows the distribution by package, and the second level
    shows modules or files, with cell sizes typically indicating code size.

    Args:
        packages (str | ModuleType | Sequence[str | ModuleType]): Single package
            name/module or list of package names/modules to analyze. When
            ``source_dir`` is provided, this is used only as a display label.
        source_dir (str | None): Path to a source directory to scan instead of
            using Python importlib. Enables language-agnostic usage — e.g. point
            at a Rust ``src/`` or TypeScript ``src/`` folder. When provided,
            files are discovered by walking the directory tree rather than
            introspecting installed Python packages. Defaults to None (Python
            package mode).
        file_extensions (tuple[str, ...]): File extensions to include when using
            ``source_dir``. Defaults to ``(".py",)``. Use ``(".rs",)`` for Rust,
            ``(".ts", ".tsx")`` for TypeScript, etc. Ignored when ``source_dir``
            is None.
        base_url (str | None): Base URL for the source code repository (e.g., GitHub).
            If provided, the label of each module/file cell is turned into a
            clickable link pointing to the corresponding source file. Example:
            'https://github.com/user/repo/blob/main'
            If set to "auto" (default), attempts to automatically find the GitHub URL
            from package metadata (requires package to be installed). Assumes 'main'
            branch. Only works reliably for single-package plots.
            If set to None, no links will be generated.
        show_counts (ShowCounts): How to display counts in treemap cells:
            - "value": Show calculated cell size value
            - "percent": Show percentage of parent
            - "value+percent": Show both (default)
            - False: Don't show counts
        cell_text_fn (ModuleFormatter | bool): How to display top-level names + counts:
            - Function that takes name, count, total count and returns string
            - True: Use default_module_formatter
            - False: Don't add counts to top-level names
        group_by (GroupBy): How to group the package modules:
            - "file": Group by filename
            - "directory": Group by top-level directory
            - "module": Group by top-level module (default)
        max_module_depth (int | None): Maximum depth of the module hierarchy to display
            when group_by="module". Files deeper than this level are aggregated into
            their ancestor at the cutoff depth. For example, max_module_depth=1 shows
            only top-level modules (io, analysis, core, ...), while max_module_depth=2
            also shows sub-packages (io/vasp, analysis/chemenv, ...). If None (default),
            the full module hierarchy is shown.
        min_lines_for_split (int): Minimum total lines of code for a child cell to
            be considered "significant" at a given depth. Children below this
            threshold are merged into an "other" bucket. If no children qualify
            as significant, the parent collapses into a single aggregated cell.
            Only applies when group_by="module". Defaults to 0 (no minimum).
        max_children_for_split (int | None): Maximum number of significant child
            cells a parent may have at the deepest level. If a parent would produce
            more significant children than this, the split is suppressed and the
            parent stays aggregated. Intermediate levels are not affected, so large
            modules always show their sub-packages. Only applies when
            group_by="module". If None (default), no limit.
        cell_size_fn (CellSizeFn | None): A callable that takes a `ModuleStats` object
            (a NamedTuple with fields like line_count, n_classes, n_functions,
            n_methods) and returns a number (int or float) to be used for the
            cell's size in the treemap. If this function returns 0, the
            corresponding module/file will be omitted from the treemap.
            If None (default), cell size is based on `line_count`.
        color_by (str | dict[str, float] | None): Controls heatmap coloring mode:
            - "coverage": Color by test coverage (see `coverage_data_file`)
            - str: Name of a numeric column in the module data to use for coloring
                (e.g. "line_count", "n_classes", "n_functions")
            - dict: Non-empty mapping from absolute file paths to numeric values
            - None: No heatmap coloring (default)
        coverage_data_file (str | None): Path to a coverage JSON file or URL (e.g., from
            `coverage json`). Used when `color_by` is "coverage". If None and
            `color_by="coverage"`, coverage will be collected on the fly. URLs must
            start with "https://".
        color_range (tuple[float, float] | None): Manual range for the color scale
            (min, max). Useful for consistent color mapping across plots or to
            ensure specific values (e.g., 0% coverage) map to specific colors.
            If None, uses the actual min/max values from the data.
        cell_border (dict[str, Any] | None): Cell border styling. If None, defaults to
            white borders with width 0.5 in coverage mode, no borders otherwise.
            Example: dict(color="black", width=2) for thick black borders.
        **kwargs (Any): Additional keyword arguments passed to plotly.express.treemap

    Returns:
        go.Figure: The Plotly figure.

    Tips and Customization:
    - rounded corners: fig.update_traces(marker=dict(cornerradius=5))
    - colors:
        - discrete: color_discrete_sequence=px.colors.qualitative.Set2
        - custom: color_discrete_map={'A': 'red', 'B': 'blue'}
    - max depth: fig.update_traces(maxdepth=2)
    - patterns/textures: fig.update_traces(marker=dict(pattern=dict(shape=["|"])))
    - hover info: fig.update_traces(hovertemplate='<b>%{label}</b><br>Lines: %{value}')
    - custom text display: fig.update_traces(textinfo="label+value+percent")

    Example:
        >>> import pymatviz as pmv
        >>> # Analyze a single package
        >>> fig1 = pmv.py_pkg_treemap("pymatviz")
        >>> # Compare multiple packages
        >>> fig2 = pmv.py_pkg_treemap(["pymatviz", "pymatgen"])
        >>> # Only show files with at least 50 lines of code by using cell_size_fn
        >>> fig3 = pmv.py_pkg_treemap(
        ...     "pymatviz",
        ...     cell_size_fn=lambda cell: (
        ...         cell.line_count if cell.line_count >= 50 else 0
        ...     ),
        ... )
        >>> # Group by top-level directory instead of module
        >>> fig4 = pmv.py_pkg_treemap("pymatviz", group_by="directory")
        >>> # Add source links for a GitHub repository
        >>> fig5 = pmv.py_pkg_treemap(
        ...     "pymatviz", base_url="https://github.com/janosh/pymatviz/blob/main"
        ... )
        >>> # Use heatmap mode with coverage data collected on the fly
        >>> fig6 = pmv.py_pkg_treemap("pymatviz", color_by="coverage")
        >>> # Use heatmap mode with pre-computed coverage data
        >>> fig7 = pmv.py_pkg_treemap(
        ...     "pymatviz", color_by="coverage", coverage_data_file="coverage.json"
        ... )
        >>> # Use heatmap mode with custom values
        >>> custom_values = {"/path/to/module1.py": 85.5, "/path/to/module2.py": 92.1}
        >>> fig8 = pmv.py_pkg_treemap("pymatviz", color_by=custom_values)
        >>> # Use with imported module objects
        >>> import numpy as np
        >>> fig9 = pmv.py_pkg_treemap(np)  # Pass the module object directly
        >>> # Or mix strings and modules
        >>> fig10 = pmv.py_pkg_treemap([np, "pymatviz"])
        >>> # Rust crate coverage treemap (language-agnostic mode)
        >>> fig11 = pmv.py_pkg_treemap(
        ...     "my_rust_crate",
        ...     source_dir="src",
        ...     file_extensions=(".rs",),
        ...     color_by="coverage",
        ...     coverage_data_file="lcov.info",
        ...     base_url="https://github.com/user/repo/blob/main",
        ... )
    """
    if max_module_depth is not None and max_module_depth < 0:
        raise ValueError(f"{max_module_depth=} must be non-negative")
    if min_lines_for_split < 0:
        raise ValueError(f"{min_lines_for_split=} must be non-negative")
    if max_children_for_split is not None and max_children_for_split < 0:
        raise ValueError(f"{max_children_for_split=} must be non-negative")
    if group_by not in get_args(GroupBy.__value__):
        raise ValueError(
            f"Invalid {group_by=}, must be one of {get_args(GroupBy.__value__)}"
        )
    valid_show_counts = get_args(ShowCounts)
    if show_counts not in valid_show_counts:
        raise ValueError(f"Invalid {show_counts=}, must be one of {valid_show_counts}")
    if isinstance(color_by, dict) and not color_by:
        raise ValueError("color_by dict must map file paths to values, got empty dict")

    # Normalize input to handle both single and multiple strings or ModuleType objects
    pkg_list = packages if isinstance(packages, (list, tuple)) else [packages]
    package_names = [_normalize_package_input(pkg) for pkg in pkg_list]

    # importlib metadata is unavailable for non-Python projects
    if source_dir and base_url == "auto":
        base_url = None

    # Per-package source URL base ('auto' looks up GitHub URLs in package metadata)
    if base_url == "auto":
        package_base_urls = {pkg: _github_base_url(pkg) for pkg in package_names}
    else:  # None disables links
        package_base_urls = dict.fromkeys(package_names, base_url)
    has_links = any(package_base_urls.values())

    # Process color_by parameter for heatmap mode
    color_values: dict[str, float] = {}
    if color_by == "coverage":
        color_values = collect_coverage_data(coverage_data_file)
    elif isinstance(color_by, dict):
        color_values = color_by.copy()

    # Collect module information
    if source_dir:
        df_modules = collect_source_modules_from_dir(
            source_dir, package_names, file_extensions
        )
    else:
        df_modules = collect_package_modules(package_names)

    if df_modules.empty:
        location = source_dir or package_names
        raise ValueError(f"No source files found in: {location}")

    if cell_size_fn is None:
        df_modules["cell_value"] = df_modules["line_count"]
    else:
        metrics_keys = ModuleStats._fields
        df_modules["cell_value"] = df_modules.apply(
            lambda row: cell_size_fn(
                ModuleStats(**{key: row[key] for key in metrics_keys if key in row})
            ),
            axis=1,
        )

    # Filter out cells where cell_value is 0
    df_modules = df_modules[df_modules["cell_value"] > 0]

    if df_modules.empty:
        location = source_dir or package_names
        raise ValueError(
            f"No source files found in {location} after filtering by "
            "cell_size_fn(module) == 0."
        )

    # Add color values to the DataFrame if heatmap mode is enabled
    df_treemap = df_modules.copy()
    has_color_mode = color_by is not None
    if color_by == "coverage":
        df_treemap["color_value"] = df_treemap["file_path"].map(
            lambda path: _match_coverage_path(
                path, color_values, df_modules, package_names
            )
        )
    elif isinstance(color_by, dict):  # direct mapping, missing files get 0
        df_treemap["color_value"] = (
            df_treemap["file_path"].map(color_values).fillna(0.0)
        )
    elif isinstance(color_by, str) and color_by in _NUMERIC_COLOR_COLUMNS:
        df_treemap["color_value"] = df_treemap[color_by]  # Use existing column
    elif color_by is not None:
        raise ValueError(
            f"Invalid {color_by=}, must be 'coverage', a dict of file path to "
            f"value, or one of the numeric module columns {_NUMERIC_COLOR_COLUMNS}"
        )

    # Determine the maximum depth of the module hierarchy
    data_max_depth = df_treemap["depth"].max()
    if max_module_depth is not None and group_by == "module":
        effective_depth = min(data_max_depth, max_module_depth)
    else:
        effective_depth = data_max_depth
    depth_was_clamped = effective_depth < data_max_depth
    level_columns = [f"level_{idx}" for idx in range(effective_depth)]

    # Create columns for each level of the hierarchy from 'module_parts'
    for level_idx, col_name in enumerate(level_columns):
        df_treemap[col_name] = df_treemap["module_parts"].map(
            lambda parts, current_idx=level_idx: (
                parts[current_idx] if current_idx < len(parts) else None
            )
        )

    # Adaptive depth pruning. Iterates from the deepest level upward.
    # At each level, "significant" children (>= min_lines_for_split) are
    # counted. If there are 1..max_children significant children, the parent
    # is expanded (small children merge into "other"). If none are significant,
    # the parent collapses to a single cell. max_children_for_split only
    # applies at the deepest level — intermediate levels use min_lines alone,
    # so large modules like io/analysis always show their sub-packages.
    rows_were_merged = False
    if group_by == "module" and (min_lines_for_split > 0 or max_children_for_split):
        for prune_depth in range(effective_depth - 1, 0, -1):
            is_deepest_level = prune_depth == effective_depth - 1
            child_col = f"level_{prune_depth}"
            parent_cols = [f"level_{idx}" for idx in range(prune_depth)]

            rows_to_drop: list[pd.Index] = []
            rows_to_add: list[dict] = []

            for _key, group in df_treemap.groupby(parent_cols, dropna=False):
                unique_children = group[child_col].dropna().unique()
                if len(unique_children) <= 1:
                    continue

                child_line_counts = group.groupby(child_col)["line_count"].sum()
                # Deepest level uses the full threshold; intermediate levels
                # use a lower cap so large modules keep their sub-packages
                effective_min = (
                    min_lines_for_split
                    if is_deepest_level
                    else min(min_lines_for_split, 500)
                )
                significant = child_line_counts[child_line_counts >= effective_min]
                n_sig = len(significant)
                # Only cap children count at the deepest level; intermediate
                # levels expand freely so large modules keep their sub-packages
                if is_deepest_level and max_children_for_split is not None:
                    max_kids = max_children_for_split
                else:
                    max_kids = float("inf")

                if 1 <= n_sig <= max_kids:
                    # EXPAND: keep significant children, merge the rest
                    sig_set = set(significant.index)
                    small_mask = group[child_col].map(
                        lambda val, sig=sig_set: pd.isna(val) or val not in sig
                    )
                    merge_rows = group[small_mask]
                    if merge_rows.empty:
                        continue
                    clear_start, is_expand = prune_depth + 1, True
                else:
                    # COLLAPSE: no children are significant, merge entire group
                    merge_rows = group
                    clear_start, is_expand = prune_depth, False

                rep = merge_rows.iloc[0].copy()
                rep["cell_value"] = merge_rows["cell_value"].sum()
                rep["line_count"] = merge_rows["line_count"].sum()
                if (
                    has_color_mode
                    and "color_value" in merge_rows.columns
                    and rep["line_count"] > 0
                ):
                    rep["color_value"] = (
                        merge_rows["color_value"] * merge_rows["line_count"]
                    ).sum() / rep["line_count"]
                if is_expand:
                    rep[child_col] = "other"
                else:
                    rep["depth"] = prune_depth
                for deeper in range(clear_start, effective_depth):
                    rep[f"level_{deeper}"] = None
                rows_to_drop.append(merge_rows.index)
                rows_to_add.append(rep.to_dict())

            if rows_to_drop:
                rows_were_merged = True
                df_treemap = df_treemap.drop(
                    index=rows_to_drop[0].append(rows_to_drop[1:])
                )
                df_treemap = pd.concat(
                    [df_treemap, pd.DataFrame(rows_to_add)], ignore_index=True
                )

            # Drop level columns that became entirely None after pruning
            level_columns = [
                col for col in level_columns if df_treemap[col].notna().any()
            ]

    # Define the path definition based on group_by
    if group_by == "file":
        path_definition = ["package", "filename"]
    elif group_by == "directory":
        path_definition = ["package", "directory", "filename"]
    else:  # group_by == "module"
        path_definition = ["package", *level_columns]

    # Keep track of raw package names for percentage calculation
    df_treemap["package_name_raw"] = df_treemap["package"]
    package_totals = df_treemap.groupby("package_name_raw")["cell_value"].sum()

    # Calculate percentage of package total for hover text based on the 'cell_value'
    df_treemap["package_total_cell_value"] = df_treemap["package_name_raw"].map(
        package_totals
    )
    df_treemap["percent_of_package_cell_value"] = (
        df_treemap["cell_value"] / df_treemap["package_total_cell_value"]
    ).fillna(0)

    # Format package labels after storing raw name and calculating totals
    if cell_text_fn is True:
        cell_text_fn = default_module_formatter
    if callable(cell_text_fn):
        total_lines = package_totals.sum()

        # When in coverage mode, show coverage percentage instead of line percentage
        if color_by == "coverage":
            # package-level coverage is the line-weighted average of its files
            package_coverage = {}
            for pkg_name in package_names:
                pkg_rows = df_treemap[df_treemap["package_name_raw"] == pkg_name]
                total_pkg_lines = pkg_rows["line_count"].sum()
                weighted_sum = (pkg_rows["color_value"] * pkg_rows["line_count"]).sum()
                package_coverage[pkg_name] = (
                    weighted_sum / total_pkg_lines if total_pkg_lines > 0 else 0.0
                )

            df_treemap["package"] = df_treemap["package_name_raw"].map(
                lambda pkg: (
                    f"{pkg} ({package_totals.get(pkg, 0):,} lines, "
                    f"{package_coverage.get(pkg, 0.0):.1f}% cov)"
                )
            )
        else:
            df_treemap["package"] = df_treemap["package_name_raw"].map(
                lambda pkg: cell_text_fn(pkg, package_totals.get(pkg, 0), total_lines)
            )

    # Full file URLs for linking ("" without a base URL: plotly>=7 casts customdata
    # to str before aggregating parents, NaN then breaks max())
    df_treemap["file_url"] = [
        f"{base}/{segment}" if (base := package_base_urls.get(pkg)) else ""
        for pkg, segment in zip(
            df_treemap["package_name_raw"],
            df_treemap["repo_path_segment"],
            strict=True,
        )
    ]

    # Define custom data columns for hovertemplate and linking
    # Order matters for hovertemplate %{customdata[index]} references
    analysis_cols = list(_ANALYSIS_COLS)
    custom_data_cols: list[str] = [
        "package_name_raw",  # 0
        "repo_path_segment",  # 1
        "leaf_label",  # 2
        "file_url",  # 3
        *analysis_cols,  # 4-9: analysis columns
        "percent_of_package_cell_value",  # 10
        "line_count",  # 11
    ]

    # Add color_value column only if in color mode
    if has_color_mode:
        custom_data_cols.append("color_value")  # 12

    # Fill NaNs in analysis columns with 0 for customdata
    df_treemap[analysis_cols] = (
        df_treemap[analysis_cols].fillna(0).infer_objects(copy=False)
    )
    if has_color_mode:
        df_treemap["color_value"] = (
            df_treemap["color_value"].fillna(0).infer_objects(copy=False)
        )

    color_defaults: dict[str, Any] = (
        dict(color="color_value", color_continuous_scale="Viridis")
        if has_color_mode
        else dict(color_discrete_sequence=px.colors.qualitative.Set3)
    )
    treemap_params = {
        "data_frame": df_treemap,
        "path": path_definition,
        "values": "cell_value",
        "custom_data": df_treemap[custom_data_cols],
        "branchvalues": "total",  # proper aggregation for parent nodes
        **color_defaults,
        **kwargs,
    }

    # None marks rows that end above the deepest level. plotly>=7 casts path columns
    # to str and max()es them, so object columns mixing str and None (-> NaN) raise;
    # pandas' string dtype keeps the gaps as <NA>.
    df_treemap[path_definition] = df_treemap[path_definition].astype("string")

    # Create the treemap
    fig = px.treemap(**treemap_params)

    # Fix parent node colors for coverage heatmaps (weighted averages).
    # Must run before label wrapping so substring matching still works.
    if color_by == "coverage":
        _apply_coverage_weighted_averages(fig, df_treemap)

    # Break long labels at underscores so plotly can wrap text instead of
    # shrinking the font to fit names like "advanced_transformations" on one line.
    trace = fig.data[0]
    max_label_len = 18
    if trace.labels is not None:
        wrapped = []
        for label in trace.labels:
            label_str = str(label)
            if (
                len(label_str) > max_label_len
                and "_" in label_str
                and " " not in label_str
            ):
                mid = len(label_str) // 2
                best_pos = min(
                    (pos for pos in range(len(label_str)) if label_str[pos] == "_"),
                    key=lambda pos: abs(pos - mid),
                )
                label_str = (
                    f"{label_str[: best_pos + 1]}<br>{label_str[best_pos + 1 :]}"
                )
            wrapped.append(label_str)
        fig.data[0].labels = wrapped

    # Configure colorbar for heatmap mode
    if has_color_mode:
        if color_by == "coverage":
            colorbar_title = "Coverage (%)"
        elif isinstance(color_by, str):
            colorbar_title = color_by.replace("_", " ").title()
        else:
            colorbar_title = "Color Value"
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=colorbar_title, title_side="right", thickness=12
            )
        )
        if color_range is None and color_by == "coverage":  # default to data range
            color_range = (
                df_treemap["color_value"].min(),
                df_treemap["color_value"].max(),
            )
        if color_range is not None:
            min_val, max_val = color_range
            fig.layout.coloraxis.update(
                cmin=min_val, cmax=max_val, cmid=(min_val + max_val) / 2
            )

    # Configure hover display using hovertemplate
    has_depth_limit = depth_was_clamped or rows_were_merged
    if has_depth_limit:
        # Aggregated/collapsed nodes inherit per-file customdata from the first
        # merged row, and Plotly synthetic parents get "(?) " placeholders.
        # Use only Plotly built-in fields that are correct for all nodes.
        hovertemplate = (
            "<b>%{label}</b><br>Value: %{value:,}<br>%{percentEntry}<br><extra></extra>"
        )
    else:
        # Indices match the order in custom_data_cols
        base_hovertemplate = (
            "<b>%{customdata[2]}</b><br>"  # leaf_label
            "Path: %{customdata[1]}<br>"  # repo_path_segment
            "Cell Value: %{value:,}<br>"  # cell_value
            "Lines: %{customdata[11]:,}<br>"  # line_count
        )

        if has_color_mode:
            color_idx = len(custom_data_cols) - 1
            if color_by == "coverage":
                base_hovertemplate += f"Coverage: %{{customdata[{color_idx}]:.1f}}%<br>"
            else:
                base_hovertemplate += (
                    f"Color Value: %{{customdata[{color_idx}]:,.2f}}<br>"
                )

        # Only show Python AST analysis fields when they have data
        has_analysis_data = df_treemap[analysis_cols].sum().sum() > 0
        analysis_hover = (
            "Classes: %{customdata[4]:,}<br>"  # n_classes
            "Functions: %{customdata[5]:,}<br>"  # n_functions
            "Methods: %{customdata[6]:,}<br>"  # n_methods
            "Internal Imports: %{customdata[7]:,}<br>"  # n_internal_imports
            "External Imports: %{customdata[8]:,}<br>"  # n_external_imports
            "Type Imports: %{customdata[9]:,}<br>"  # n_type_checking_imports
            if has_analysis_data
            else ""
        )
        hovertemplate = (
            base_hovertemplate
            + "%{customdata[10]:.1%} of %{customdata[0]} (by cell value)<br>"
            + analysis_hover
            + "<extra></extra>"
        )
    # Apply hovertemplate - applies to all traces, which is fine as we have one
    fig.update_traces(hovertemplate=hovertemplate)

    count_part = {
        "percent": "%{percentEntry}",
        "value": "%{value:,}",
        "value+percent": "%{value:,}<br>%{percentEntry}",
        False: "",
    }[show_counts]
    # Aggregated parent nodes lack customdata for links and coverage labels.
    coverage_text = ""
    if not has_depth_limit and color_by == "coverage":
        coverage_text = f"<br>%{{customdata[{len(custom_data_cols) - 1}]:,.1f}}% cov"
    use_links = has_links and not has_depth_limit
    texttemplate = ""
    if use_links or coverage_text or show_counts in ("value", "value+percent"):
        label_part = (
            "<a href='%{customdata[3]}' target='_blank'>%{customdata[2]}</a>"
            if use_links
            else "%{label}"
        )
        texttemplate = (
            label_part + (f"<br>{count_part}" if count_part else "") + coverage_text
        )
        textinfo = "none"
    else:
        textinfo = "label+percent entry" if show_counts == "percent" else "label"
    fig.data[0].textinfo = textinfo
    if texttemplate:
        fig.data[0].texttemplate = texttemplate

    # Set cell borders
    if has_color_mode and cell_border is None:
        cell_border = dict(color="white", width=0.5)
    if cell_border is not None:
        fig.update_traces(marker=dict(line=cell_border))

    fig.layout.margin = dict(l=0, r=0, b=0, t=40, pad=0)
    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig
