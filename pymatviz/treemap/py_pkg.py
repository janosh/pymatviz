"""Hierarchical treemap visualizations of Python packages.

For visualizing module structure and code distribution within Python packages.
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


if TYPE_CHECKING:
    from types import ModuleType

    from pymatviz.typing import ShowCounts

ModuleFormatter = Callable[[str, int, int], str]
GroupBy: TypeAlias = Literal["file", "directory", "module"]


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


CellSizeFn: TypeAlias = Callable[[ModuleStats], float | int]


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


def count_lines(file_path: str) -> int:
    """Count non-empty, non-comment lines in a Python file."""
    try:
        with open(file_path, encoding="utf-8") as file_handle:
            try:  # Read lines safely, handling potential decoding errors
                lines = file_handle.readlines()
            except UnicodeDecodeError:
                return 0  # Treat undecodable files as having 0 countable lines

            return sum(
                bool(line.strip() and not line.strip().startswith("#"))
                for line in lines
            )
    except (FileNotFoundError, PermissionError):
        return 0


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
        if path_entry and os.path.isdir(path_entry):
            package_path = os.path.join(path_entry, package_name)
            if os.path.isdir(package_path):
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
    counts = {
        "n_classes": 0,
        "n_functions": 0,
        "n_methods": 0,
        "n_internal_imports": 0,
        "n_external_imports": 0,
        "n_type_checking_imports": 0,
    }
    try:
        with open(file_path, encoding="utf-8") as file_handle:
            source = file_handle.read()
        tree = ast.parse(source, filename=file_path)

        in_type_checking_block = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            ):
                original_in_block = in_type_checking_block
                in_type_checking_block = True
                for sub_node in node.body:
                    if isinstance(sub_node, (ast.Import, ast.ImportFrom)):
                        counts["n_type_checking_imports"] += 1
                        continue
                in_type_checking_block = original_in_block
                continue

            if in_type_checking_block:
                pass

            if isinstance(node, ast.ClassDef):
                counts["n_classes"] += 1
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        counts["n_methods"] += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                counts["n_functions"] += 1
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == package_name:
                        counts["n_internal_imports"] += 1
                    else:
                        counts["n_external_imports"] += 1
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    counts["n_internal_imports"] += 1
                elif node.module:
                    if node.module.split(".")[0] == package_name:
                        counts["n_internal_imports"] += 1
                    else:
                        counts["n_external_imports"] += 1

    except (FileNotFoundError, SyntaxError, PermissionError, UnicodeDecodeError) as exc:
        print(f"Warning: Could not analyze {file_path}: {exc}")  # noqa: T201
    return counts


def collect_coverage_data(coverage_data_file: str | None = None) -> dict[str, float]:
    """Collect coverage data for packages.

    When coverage_data_file is not provided, this function attempts to run the
    'coverage' command to generate coverage data on the fly. This requires the coverage
    tool to be available in system PATH.

    Args:
        coverage_data_file: Path to coverage JSON file or URL. If provided as a local
            file and not found, raises FileNotFoundError instead of auto-generating.
            If provided as a URL, fetches the coverage data from the remote location.
            URLs must start with "https://".

    Returns:
        dict[str, float]: Mapping from file paths to coverage percentages.
    """
    coverage_map: dict[str, float] = {}

    if coverage_data_file:
        is_url = coverage_data_file.startswith("https://")

        try:
            if is_url:
                # Security: Only HTTPS URLs allowed to prevent file:// and other schemes
                with urllib.request.urlopen(coverage_data_file) as response:  # noqa: S310
                    coverage_data = json.load(response)
            else:
                with open(coverage_data_file, encoding="utf-8") as file_handle:
                    coverage_data = json.load(file_handle)
        except (urllib.error.URLError, Exception) as exc:
            if is_url:
                raise ValueError(
                    f"Failed to fetch coverage data from URL: {exc}"
                ) from exc
            raise  # Re-raise original exception for local files

        # Process coverage data
        for file_path, file_data in coverage_data.get("files", {}).items():
            summary = file_data.get("summary", {})
            if "percent_covered" in summary:
                if is_url:
                    coverage_map[file_path] = summary["percent_covered"]
                else:
                    # Convert relative paths to absolute paths for local files
                    coverage_dir = os.path.dirname(os.path.abspath(coverage_data_file))
                    abs_path = os.path.normpath(f"{coverage_dir}/{file_path}")
                    coverage_map[abs_path] = summary["percent_covered"]
    else:
        try:  # Try to collect coverage on the fly
            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False
            ) as tmp_cov_file:
                tmp_csv_path = tmp_cov_file.name

            result = subprocess.run(  # noqa: S603
                ["coverage", "json", "-o", tmp_csv_path],  # noqa: S607
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                with open(tmp_csv_path, encoding="utf-8") as f:
                    data = json.load(f)
                    for file_path, file_data in data.get("files", {}).items():
                        summary = file_data.get("summary", {})
                        if "percent_covered" in summary:
                            coverage_map[file_path] = summary["percent_covered"]
            else:
                print(f"Warning: Coverage command failed: {result.stderr}")  # noqa: T201

            if os.path.exists(tmp_csv_path):
                os.unlink(tmp_csv_path)

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            json.JSONDecodeError,
        ) as exc:
            print(f"Warning: Could not collect coverage data: {exc}")  # noqa: T201

    return coverage_map


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
            print(f"Warning: Could not find package path for {package_name}")  # noqa: T201
            continue

        python_files = []
        for root, dirs, files in os.walk(pkg_path):
            dirs[:] = [folder for folder in dirs if folder not in ignored_dirs]

            for filename in files:
                if filename.endswith(".py") and not filename.startswith("_"):
                    file_path = os.path.normpath(f"{root}/{filename}")
                    python_files.append(file_path)

        for file_path in python_files:
            line_count = count_lines(file_path)
            rel_path = os.path.relpath(file_path, os.path.dirname(pkg_path))
            module_name = (
                rel_path.replace(".py", "").replace(os.sep, ".").replace("\\", ".")
            )

            # Extract module path components (e.g. pymatviz.utils.io -> ['utils', 'io'])
            relative_module_name = (
                module_name[len(package_name) + 1 :]
                if module_name.startswith(package_name + ".")
                else module_name
            )
            module_parts = (
                relative_module_name.split(".") if relative_module_name else []
            )

            # Get top directory for directory grouping
            directory_parts = rel_path.split(os.sep)
            directory = directory_parts[0] if len(directory_parts) > 1 else "root"

            # Filename without extension for file grouping
            filename = os.path.basename(rel_path).replace(".py", "")

            # Top-level module for grouping
            top_module = module_parts[0] if module_parts else "root"

            # Calculate path segment for repository URL
            # Path relative to the immediate package root directory
            path_relative_to_pkg_root = os.path.relpath(
                file_path, start=pkg_path
            ).replace(os.sep, "/")

            # Determine repo path prefix based on package structure
            repo_path_prefix = package_name  # Default: repo has pkg_name dir at root
            path_parts = pkg_path.split(os.sep)

            if (  # Check for src-layout (e.g. .../repo/src/pkg_name)
                len(path_parts) >= 2
                and path_parts[-2] == "src"
                and path_parts[-1] == package_name
            ):
                repo_path_prefix = f"src/{package_name}"

            repo_path_segment = f"{repo_path_prefix}/{path_relative_to_pkg_root}"

            # Determine leaf node label for linking text
            leaf_label = (
                module_parts[-1] if module_parts else (filename or package_name)
            )

            analysis_results = _analyze_py_file(file_path, package_name)

            all_modules.append(
                {
                    "package": package_name,
                    "full_module": module_name,
                    "filename": filename,
                    "directory": directory,
                    "top_module": top_module,  # Needed for grouping
                    "line_count": line_count,
                    "file_path": file_path,  # Absolute path
                    "repo_path_segment": repo_path_segment,  # Path for URL
                    "leaf_label": leaf_label,  # Link text
                    "module_parts": module_parts,  # Parts for hierarchy
                    "depth": len(module_parts),
                    **analysis_results,  # Add n_classes, n_functions, etc.
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
            if pkg_name in cov_path_unix:
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

    # Create a mapping from treemap labels to DataFrame rows
    label_to_row = {}
    for _, row in df_treemap.iterrows():
        # Try to match by various criteria
        for label in labels:
            if (
                # Match by exact line count and module name
                (
                    abs(values[list(labels).index(label)] - row["line_count"]) < 1
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
    node_coverage = {}
    for label, row in label_to_row.items():
        if "color_value" in row:
            node_coverage[label] = row["color_value"]

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
                    child_idx = list(labels).index(child)
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
    base_url: str | None = "auto",
    show_counts: ShowCounts = "value+percent",
    cell_text_fn: ModuleFormatter | bool = default_module_formatter,
    group_by: GroupBy = "module",
    cell_size_fn: CellSizeFn | None = None,
    color_by: str | dict[str, float] | None = None,
    coverage_data_file: str | None = None,
    color_range: tuple[float, float] | None = None,
    cell_border: dict[str, Any] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Generate a treemap plot showing the module structure of Python packages.

    The first level shows the distribution by package, and the second level
    shows modules or files, with cell sizes typically indicating code size.

    Args:
        packages (str | ModuleType | Sequence[str | ModuleType]): Single package
            name/module or list of package names/modules to analyze.
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
        cell_size_fn (CellSizeFn | None): A callable that takes a `ModuleStats` object
            (a NamedTuple with fields like line_count, n_classes, n_functions,
            n_methods) and returns a number (int or float) to be used for the
            cell's size in the treemap. If this function returns 0, the
            corresponding module/file will be omitted from the treemap.
            If None (default), cell size is based on `line_count`.
        color_by (str | dict[str, float] | None): Controls heatmap coloring mode:
            - str: Name of a column in the module data to use for coloring
            - dict: Mapping from absolute file paths to numeric values for coloring
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
        ...     cell_size_fn=lambda cell: cell.line_count
        ...     if cell.line_count >= 50
        ...     else 0,
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
    """
    # Normalize input to handle both single and multiple strings or ModuleType objects
    pkg_list = packages if isinstance(packages, (list, tuple)) else [packages]
    package_names = [_normalize_package_input(pkg) for pkg in pkg_list]

    # Handle base_url = 'auto' using package metadata
    package_base_urls: dict[str, str | None] = {}
    if base_url == "auto":
        for package in package_names:
            github_url_found = None
            try:
                metadata = importlib.metadata.metadata(package)

                # Try Home-page first, then Project-URL entries
                homepage = metadata.get("Home-page")
                if homepage and "github.com" in homepage:
                    github_url_found = homepage
                else:
                    project_urls = metadata.get_all("Project-URL")
                    if project_urls:
                        for url_entry in project_urls:
                            url_parts = url_entry.split(",", 1)
                            if len(url_parts) == 2:
                                url = url_parts[1].strip()
                                if "github.com" in url:
                                    github_url_found = url
                                    break

                if github_url_found:  # Construct base_url (assuming main branch)
                    package_base_urls[package] = (
                        f"{github_url_found.rstrip('/')}/blob/main"
                    )
                else:
                    package_base_urls[package] = None

            except importlib.metadata.PackageNotFoundError:
                # Ignore packages not found by metadata
                package_base_urls[package] = None
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: Error processing metadata for {package}: {exc}")  # noqa: T201
                package_base_urls[package] = None

        # Check if any URL was found to enable link generation
        processed_base_url = True if any(package_base_urls.values()) else None

    elif base_url is not None:
        for package in package_names:
            package_base_urls[package] = base_url
        processed_base_url = True
    else:
        processed_base_url = None  # Links explicitly disabled

    # Process color_by parameter for heatmap mode
    color_values: dict[str, float] = {}
    if color_by == "coverage":
        color_values = collect_coverage_data(coverage_data_file)
    elif isinstance(color_by, dict):
        color_values = color_by.copy()

    # Collect module information
    df_modules = collect_package_modules(package_names)

    if df_modules.empty:
        raise ValueError(f"No Python modules found in packages: {package_names}")

    if cell_size_fn is None:  # set default cell size calculator if not provided
        cell_size_fn = lambda module: module.line_count

    # Apply cell_size_fn to get treemap sizing values
    metrics_keys = ModuleStats._fields
    df_modules["cell_value"] = df_modules.apply(
        lambda row: cell_size_fn(  # type: ignore[call-non-callable]
            ModuleStats(**{k: row[k] for k in metrics_keys if k in row})
        ),
        axis=1,
    )

    # Filter out cells where cell_value is 0
    df_modules = df_modules[df_modules["cell_value"] > 0]

    if df_modules.empty:
        raise ValueError(
            f"No Python modules found in {package_names=} after filtering by "
            "cell_size_fn(module) == 0."
        )

    # Add color values to the DataFrame if heatmap mode is enabled
    has_color_mode = bool(color_by)
    if has_color_mode:
        if isinstance(color_by, str) and color_by in df_modules.columns:
            df_modules["color_value"] = df_modules[color_by]  # Use existing column
        elif isinstance(color_by, dict) or color_by == "coverage":
            # Map file paths to color values, defaulting to 0.0 for missing values
            if color_by == "coverage":
                df_modules["color_value"] = df_modules["file_path"].apply(
                    lambda path: _match_coverage_path(
                        path, color_values, df_modules, package_names
                    )
                )
            else:
                df_modules["color_value"] = (  # For custom dict, use direct mapping
                    df_modules["file_path"].map(color_values).fillna(0.0)
                )
        else:
            df_modules["color_value"] = None  # Unknown color_by type - set to None
            has_color_mode = False
    else:
        has_color_mode = False  # No coloring - don't add the column

    df_treemap = df_modules.copy()

    # Determine the maximum depth of the module hierarchy
    max_depth = df_treemap["depth"].max() if not df_treemap.empty else 0
    level_columns = [f"level_{idx}" for idx in range(max_depth)]

    # Create columns for each level of the hierarchy from 'module_parts'
    for level_idx, col_name in enumerate(level_columns):
        df_treemap[col_name] = df_treemap["module_parts"].map(
            lambda parts, current_idx=level_idx: parts[current_idx]
            if current_idx < len(parts)
            else None
        )

    # Define the path definition based on group_by
    if group_by == "file":
        path_definition = ["package", "filename"]
    elif group_by == "directory":
        path_definition = ["package", "directory", "filename"]
    else:  # default is "module"
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
        if color_by == "coverage" and has_color_mode:
            # Calculate package-level coverage as weighted average
            package_coverage = {}
            for pkg_name in package_names:
                pkg_rows = df_treemap[df_treemap["package_name_raw"] == pkg_name]
                if not pkg_rows.empty and "color_value" in pkg_rows.columns:
                    total_pkg_lines = pkg_rows["line_count"].sum()
                    if total_pkg_lines > 0:
                        weighted_coverage = (
                            pkg_rows["color_value"] * pkg_rows["line_count"]
                        ).sum() / total_pkg_lines
                        package_coverage[pkg_name] = weighted_coverage
                    else:
                        package_coverage[pkg_name] = 0.0
                else:
                    package_coverage[pkg_name] = 0.0

            # Custom formatter for coverage mode
            def coverage_formatter(pkg: str, lines: int, _total: int) -> str:
                coverage = package_coverage.get(pkg, 0.0)
                return f"{pkg} ({lines:,} lines, {coverage:.1f}% cov)"

            df_treemap["package"] = df_treemap["package_name_raw"].map(
                lambda pkg: coverage_formatter(
                    pkg, package_totals.get(pkg, 0), total_lines
                )
            )
        else:
            df_treemap["package"] = df_treemap["package_name_raw"].map(
                lambda pkg: cell_text_fn(pkg, package_totals.get(pkg, 0), total_lines)  # type: ignore[call-non-callable]
            )

    # Calculate full file URLs for linking
    def _get_file_url(row: pd.Series) -> str | None:
        pkg_name = row["package_name_raw"]
        base = package_base_urls.get(pkg_name)
        if base:
            # Use repo_path_segment which includes src/ prefix if needed
            return f"{base}/{row['repo_path_segment']}"
        return None

    df_treemap["file_url"] = df_treemap.apply(_get_file_url, axis=1)

    # Define custom data columns for hovertemplate and linking
    # Order matters for hovertemplate %{customdata[index]} references
    analysis_cols = [
        "n_classes",
        "n_functions",
        "n_methods",
        "n_internal_imports",
        "n_external_imports",
        "n_type_checking_imports",
    ]
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

    # Prepare custom_data array
    customdata = df_treemap[custom_data_cols]

    treemap_defaults = dict(
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    # Prepare treemap parameters
    treemap_params = {
        "data_frame": df_treemap,
        "path": path_definition,
        "values": "cell_value",
        "custom_data": customdata,
        "branchvalues": "total",  # proper aggregation for parent nodes
        **treemap_defaults,
        **kwargs,
    }

    # Add color parameter if heatmap mode is enabled
    if has_color_mode and df_treemap["color_value"].notna().any():
        treemap_params["color"] = "color_value"
        # Set a continuous color scale for heatmap mode
        if "color_continuous_scale" not in kwargs:
            treemap_params["color_continuous_scale"] = "Viridis"
        # Colorbar title will be set after figure creation
        # Remove discrete color sequence when using continuous colors
        treemap_params.pop("color_discrete_sequence", None)

    # Create the treemap
    fig = px.treemap(**treemap_params)

    # Fix parent node colors for coverage heatmaps (weighted averages)
    if has_color_mode and color_by == "coverage":
        _apply_coverage_weighted_averages(fig, df_treemap)

    # Configure colorbar for heatmap mode
    if has_color_mode and df_treemap["color_value"].notna().any():
        colorbar_title = (
            "Coverage (%)"
            if color_by == "coverage"
            else (
                color_by.replace("_", " ").title()
                if isinstance(color_by, str)
                else "Color Value"
            )
        )
        fig.update_layout(
            coloraxis_colorbar=dict(
                title=colorbar_title, title_side="right", thickness=12
            )
        )

        # Set color range for heatmap mode
        if color_range is not None:  # Use manually specified range
            min_val, max_val = color_range
            fig.layout.coloraxis.update(
                cmin=min_val, cmax=max_val, cmid=(min_val + max_val) / 2
            )
        elif color_by == "coverage":  # For coverage, use data range by default
            min_cov = df_treemap["color_value"].min()
            max_cov = df_treemap["color_value"].max()
            fig.layout.coloraxis.update(
                cmin=min_cov, cmax=max_cov, cmid=(min_cov + max_cov) / 2
            )

    # Configure hover display using hovertemplate
    # Indices match the order in custom_data_cols
    base_hovertemplate = (
        "<b>%{customdata[2]}</b><br>"  # leaf_label
        "Path: %{customdata[1]}<br>"  # repo_path_segment
        "Cell Value: %{value:,}<br>"  # cell_value
        "Lines: %{customdata[11]:,}<br>"  # line_count
    )

    # Add color value to hover if heatmap mode is enabled
    if has_color_mode and not df_treemap["color_value"].isna().all():
        color_label = "Coverage" if color_by == "coverage" else "Color Value"
        color_format = ":.1f%" if color_by == "coverage" else ":,.2f"
        # Last column when in color mode
        color_value_index = len(custom_data_cols) - 1
        base_hovertemplate += (
            f"{color_label}: %{{customdata[{color_value_index}]{color_format}}}<br>"
        )

    hovertemplate = (
        base_hovertemplate +
        # percent_of_package_cell_value
        "%{customdata[10]:.1%} of %{customdata[0]} (by cell value)<br>"
        "Classes: %{customdata[4]:,}<br>"  # n_classes
        "Functions: %{customdata[5]:,}<br>"  # n_functions
        "Methods: %{customdata[9]:,}<br>"  # n_methods
        "Internal Imports: %{customdata[6]:,}<br>"  # n_internal_imports
        "External Imports: %{customdata[7]:,}<br>"  # n_external_imports
        "Type Imports: %{customdata[8]:,}<br>"  # n_type_checking_imports
        "<extra></extra>"  # Remove Plotly trace info box
    )
    # Apply hovertemplate - applies to all traces, which is fine as we have one
    fig.update_traces(hovertemplate=hovertemplate)

    # Configure text display
    def _build_text_template() -> tuple[str, str]:
        """Build text template and textinfo based on configuration."""

        def _get_coverage_text() -> str:
            """Get coverage text for display."""
            if color_by == "coverage" and has_color_mode:
                color_value_index = len(custom_data_cols) - 1
                return f"<br>%{{customdata[{color_value_index}]:,.1f}}% cov"
            return ""

        # Validate show_counts first
        valid_show_counts = ("percent", "value", "value+percent", False)
        if show_counts not in valid_show_counts:
            raise ValueError(
                f"Invalid {show_counts=}, must be one of {valid_show_counts}"
            )

        # Add coverage information if in coverage heatmap mode
        coverage_text = _get_coverage_text()

        # Define template components based on show_counts
        count_templates = {
            "percent": "%{percentEntry}",
            "value": "%{value:,}",  # %{value} refers to cell_value
            "value+percent": "%{value:,}<br>%{percentEntry}",
            False: "",
        }

        # Define textinfo fallbacks when not using custom template
        textinfo_fallbacks = {
            "percent": "label+percent entry",
            "value": "label+value",
            "value+percent": "label+value+percent entry",
            False: "label",
        }

        if processed_base_url:  # If link generation is enabled
            # Create the HTML link part using customdata[2,3] = file_url, leaf_label
            link_part = (
                "<a href='%{customdata[3]}' target='_blank'>%{customdata[2]}</a>"
            )
            count_part = count_templates[show_counts]

            # Always use custom template when links are enabled
            template_parts = [link_part]
            if count_part:
                template_parts.append(f"<br>{count_part}")
            if coverage_text:
                template_parts.append(coverage_text)

            return "none", "".join(template_parts)
        # Default behavior when base_url is not provided
        count_part = count_templates[show_counts]

        if coverage_text:
            # Use custom template when coverage is shown
            template_parts = ["%{label}"]
            if count_part:
                template_parts.append(f"<br>{count_part}")
            template_parts.append(coverage_text)
            return "none", "".join(template_parts)
        if show_counts in ("value", "value+percent"):
            # Use custom template for value formatting (maintains comma formatting)
            template_parts = ["%{label}"]
            if count_part:
                template_parts.append(f"<br>{count_part}")
            return "none", "".join(template_parts)
        # Use Plotly's built-in textinfo for simple cases
        return textinfo_fallbacks[show_counts], ""

    textinfo, texttemplate = _build_text_template()
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
