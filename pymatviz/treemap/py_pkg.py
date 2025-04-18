"""Hierarchical treemap visualizations of Python packages.

For visualizing module structure and code distribution within Python packages.
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


if TYPE_CHECKING:
    import plotly.graph_objects as go

ShowCounts = Literal["value", "percent", "value+percent", False]
ModuleFormatter = Callable[[str, int, int], str]
GroupBy: TypeAlias = Literal["file", "directory", "module"]


def default_module_formatter(module: str, count: int, total: int) -> str:
    """Default formatter showing module name, line count, and percentage on one line.

    Example output: my_package (1,234 Lines, 15.6%)
    """
    # Avoid ZeroDivisionError if total is 0
    percent_str = f", {count / total:.1%}" if total > 0 else ""
    return f"{module} ({count:,} lines{percent_str})"


def count_lines(file_path: str) -> int:
    """Count non-empty, non-comment lines in a Python file."""
    try:
        with open(file_path, encoding="utf-8") as file_handle:
            lines = []
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
        # First try with importlib.resources (Python 3.9+)
        try:
            from importlib import resources

            package_root = resources.files(package_name)
            if package_root and os.path.isdir(str(package_root)):
                return str(package_root)
        except (ImportError, ModuleNotFoundError, TypeError):
            # Fall back to importlib.util
            spec = importlib.util.find_spec(package_name)
            if spec and spec.origin:
                # Check if origin points to __init__.py
                if os.path.basename(spec.origin) == "__init__.py":
                    return os.path.dirname(spec.origin)
                # Assume it's a single module file
                return os.path.dirname(spec.origin)
    except (ImportError, ModuleNotFoundError):
        pass

    # Try common locations if importlib approach fails
    current_dir = os.path.abspath(os.getcwd())
    possible_paths = [
        f"{current_dir}/{package_name}",
        f"{current_dir}/src/{package_name}",
    ]

    # Also try site-packages
    try:
        import site

        site_packages = site.getsitepackages()[0]
        possible_paths.append(f"{site_packages}/{package_name}")
    except (ImportError, IndexError):
        pass

    for path in possible_paths:
        if os.path.isdir(path):
            return path

    # If we can't find it, return empty string
    return ""


def _analyze_py_file(file_path: str, package_name: str) -> dict[str, int]:
    """Parse Python file using AST to count classes, functions, and imports."""
    counts = {
        "n_classes": 0,
        "n_functions": 0,
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


def collect_package_modules(
    package_names: Sequence[str], min_lines: int = 0
) -> pd.DataFrame:
    """Collect information about all modules in the given packages.

    Args:
        package_names: Names of packages to analyze
        min_lines: Minimum number of code lines for a module to be included

    Returns:
        DataFrame with package structure information
    """
    all_modules = []

    for package_name in package_names:
        pkg_path = find_package_path(package_name)
        if not pkg_path:
            print(f"Warning: Could not find package path for {package_name}")  # noqa: T201
            continue

        # Find all Python files in the package
        python_files = []
        for root, _, files in os.walk(pkg_path):
            for file in files:
                if file.endswith(".py") and not file.startswith("_"):
                    file_path = f"{root}/{file}"
                    python_files.append(file_path)

        for file_path in python_files:
            # Count lines in the file
            line_count = count_lines(file_path)
            if line_count < min_lines:
                continue

            # Convert file path to module info
            rel_path = os.path.relpath(file_path, os.path.dirname(pkg_path))

            # Full module name including package prefix
            module_name = (
                rel_path.replace(".py", "").replace(os.sep, ".").replace("\\", ".")
            )

            # Create a list of relative module path components
            # e.g., for pymatviz.utils.io -> ['utils', 'io']
            relative_module_name = (
                module_name[len(package_name) + 1 :]
                if module_name.startswith(package_name + ".")
                else module_name
            )
            module_parts = (
                relative_module_name.split(".") if relative_module_name else []
            )

            # For directory grouping, get the top directory
            directory_parts = rel_path.split(os.sep)
            directory = directory_parts[0] if len(directory_parts) > 1 else "root"

            # Filename without extension for file grouping
            filename = os.path.basename(rel_path).replace(".py", "")

            # Top-level module for grouping
            top_module = module_parts[0] if len(module_parts) > 0 else "root"

            # Calculate path segment for repository URL
            # Path relative to the immediate package root directory
            path_relative_to_pkg_root = os.path.relpath(
                file_path, start=pkg_path
            ).replace(os.sep, "/")

            # Determine the likely prefix in the repo URL structure based on pkg_path
            repo_path_prefix = package_name  # Default: repo has pkg_name dir at root
            path_parts = pkg_path.split(os.sep)
            # Check if pkg_path suggests a src-layout (e.g., .../repo/src/pkg_name)
            if (
                len(path_parts) >= 2
                and path_parts[-2] == "src"
                and path_parts[-1] == package_name
            ):
                repo_path_prefix = f"src/{package_name}"  # Refined: src/pkg_name dir

            # Combine prefix and relative path for the full segment used in the URL
            repo_path_segment = f"{repo_path_prefix}/{path_relative_to_pkg_root}"

            # Determine the label for the leaf node (used for linking text)
            if len(module_parts) > 0:
                leaf_label = module_parts[-1]
            else:
                # Case for top-level modules/files directly under package root
                leaf_label = filename if filename else package_name

            # Analyze file contents
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


def py_pkg_treemap(
    packages: str | Sequence[str],
    *,
    base_url: str | None = "auto",  # Default to 'auto'
    show_counts: ShowCounts = "value+percent",
    show_module_counts: ModuleFormatter | bool = default_module_formatter,
    group_by: GroupBy = "module",
    min_lines: int = 0,
    **kwargs: Any,
) -> go.Figure:
    """Generate a treemap plot showing the module structure of Python packages.

    The first level shows the distribution by package, and the second level
    shows modules or files, with line counts indicating code size.

    Args:
        packages: Single package name or list of package names to analyze
        base_url: Base URL for the source code repository (e.g., GitHub).
            If provided, the label of each module/file cell is turned into a
            clickable link pointing to the corresponding source file. Example:
            'https://github.com/user/repo/blob/main'
            If set to "auto" (default), attempts to automatically find the GitHub URL
            from package metadata (requires package to be installed). Assumes 'main'
            branch. Only works reliably for single-package plots.
            If set to None, no links will be generated.
        show_counts: How to display counts in treemap cells:
            - "value": Show line counts
            - "percent": Show percentage of parent
            - "value+percent": Show both (default)
            - False: Don't show counts
        show_module_counts: How to display top-level names and counts:
            - Function that takes name, count, total count and returns string
            - True: Use default_module_formatter
            - False: Don't add counts to top-level names
        group_by: How to group the package modules:
            - "file": Group by filename
            - "directory": Group by top-level directory
            - "module": Group by top-level module (default)
        min_lines: Minimum number of code lines for a file to be included
        **kwargs: Additional keyword arguments passed to plotly.express.treemap

    Returns:
        Figure: The Plotly figure.

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
        >>> # Only show files with at least 50 lines of code
        >>> fig3 = pmv.py_pkg_treemap("pymatviz", min_lines=50)
        >>> # Group by top-level directory instead of module
        >>> fig4 = pmv.py_pkg_treemap("pymatviz", group_by="directory")
        >>> # Add source links for a GitHub repository
        >>> fig5 = pmv.py_pkg_treemap(
        ...     "pymatviz", base_url="https://github.com/janosh/pymatviz/blob/main"
        ... )
    """
    # Handle single package case
    if isinstance(packages, str):
        packages = [packages]

    # Handle base_url = 'auto' using package metadata
    package_base_urls: dict[str, str | None] = {}
    if base_url == "auto":
        for package in packages:
            github_url_found = None
            try:
                metadata = importlib.metadata.metadata(package)

                # Try getting Home-page first
                homepage = metadata.get("Home-page")
                if homepage and "github.com" in homepage:
                    github_url_found = homepage
                else:
                    # Fallback: check Project-URL entries
                    project_urls = metadata.get_all("Project-URL")
                    if project_urls:
                        for url_entry in project_urls:
                            url_parts = url_entry.split(",", 1)
                            if len(url_parts) == 2:
                                url = url_parts[1].strip()
                                if "github.com" in url:
                                    github_url_found = url
                                    break

                if github_url_found:
                    # Construct base_url (assuming main branch)
                    package_base_urls[package] = (
                        github_url_found.rstrip("/") + "/blob/main"
                    )
                else:
                    package_base_urls[package] = None  # Mark as not found

            except importlib.metadata.PackageNotFoundError:
                # Ignore packages not found by metadata
                package_base_urls[package] = None
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: Error processing metadata for {package}: {exc}")  # noqa: T201
                package_base_urls[package] = None  # Mark as error

        # Determine if any URL was found to enable link generation
        processed_base_url = True if any(package_base_urls.values()) else None

    elif base_url is not None:
        # Use the provided base_url for all packages
        for package in packages:
            package_base_urls[package] = base_url
        processed_base_url = True
    else:
        processed_base_url = None  # Links explicitly disabled

    # Collect module information
    df_modules = collect_package_modules(packages, min_lines)

    if df_modules.empty:
        raise ValueError(f"No Python modules found in packages: {packages}")

    df_treemap = df_modules.copy()

    # Determine the maximum depth of the module hierarchy
    max_depth = df_treemap["depth"].max() if not df_treemap.empty else 0
    level_columns = [f"level_{idx}" for idx in range(max_depth)]

    # Create columns for each level of the hierarchy from 'module_parts'
    for level_idx, col_name in enumerate(level_columns):
        df_treemap[col_name] = df_treemap["module_parts"].apply(
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

    # Keep track of raw package names for percentage calculation in hover text
    df_treemap["package_name_raw"] = df_treemap["package"]
    package_totals = df_treemap.groupby("package_name_raw")["line_count"].sum()

    # Calculate percentage of package total for hover text
    df_treemap["package_total"] = df_treemap["package_name_raw"].map(package_totals)
    df_treemap["percent_of_package"] = (
        df_treemap["line_count"] / df_treemap["package_total"]
    ).fillna(0)

    # Format package labels *after* storing raw name and calculating totals
    if show_module_counts is True:
        show_module_counts = default_module_formatter
    if show_module_counts is not False:
        total_lines = package_totals.sum()
        df_treemap["package"] = df_treemap["package_name_raw"].map(
            lambda pkg: show_module_counts(pkg, package_totals.get(pkg, 0), total_lines)
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
        "n_internal_imports",
        "n_external_imports",
        "n_type_checking_imports",
    ]
    custom_data_cols: list[str] = [
        "package_name_raw",  # 0
        "repo_path_segment",  # 1
        "leaf_label",  # 2
        "file_url",  # 3
        *analysis_cols,  # 4-8
        "percent_of_package",  # 9
    ]

    # Fill NaNs in analysis columns with 0 for customdata
    df_treemap[analysis_cols] = df_treemap[analysis_cols].fillna(0)

    # Prepare custom_data array
    customdata = df_treemap[custom_data_cols]

    treemap_defaults = dict(
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    # Create the treemap using the DataFrame and level columns
    fig = px.treemap(
        df_treemap,
        path=path_definition,
        values="line_count",
        custom_data=customdata,
        **treemap_defaults | kwargs,
    )

    # Configure hover display using hovertemplate
    # Indices match the order in custom_data_cols
    hovertemplate = (
        "<b>%{customdata[2]}</b><br>"  # leaf_label
        "Path: %{customdata[1]}<br>"  # repo_path_segment
        "Lines: %{value:,}<br>"  # line_count (from values)
        "%{customdata[9]:.1%} of %{customdata[0]}<br>"  # percent_of_package
        "Classes: %{customdata[4]:,}<br>"  # n_classes
        "Functions: %{customdata[5]:,}<br>"  # n_functions
        "Internal Imports: %{customdata[6]:,}<br>"  # n_internal_imports
        "External Imports: %{customdata[7]:,}<br>"  # n_external_imports
        "Type Imports: %{customdata[8]:,}<br>"  # n_type_checking_imports
        "<extra></extra>"  # Remove Plotly trace info box
    )
    # Apply hovertemplate - applies to all traces, which is fine as we have one
    fig.update_traces(hovertemplate=hovertemplate)

    # Configure text display: use texttemplate for links if base_url is provided
    if processed_base_url:  # If link generation is enabled
        # Create the HTML link part using customdata
        # Assumes file_url (customdata[3]) is valid if present
        link_part = "<a href='%{customdata[3]}' target='_blank'>%{customdata[2]}</a>"  # file_url, leaf_label  # noqa: E501

        # Set textinfo to none as we are defining the full template
        fig.data[0].textinfo = "none"

        # Build the texttemplate based on show_counts
        if show_counts == "percent":
            fig.data[0].texttemplate = f"{link_part}<br>%{{percentEntry}}"
        elif show_counts == "value":
            fig.data[0].texttemplate = f"{link_part}<br>%{{value:,}} lines"
        elif show_counts == "value+percent":
            fig.data[
                0
            ].texttemplate = f"{link_part}<br>%{{value:,}} lines<br>%{{percentEntry}}"
        elif show_counts is False:
            # Only show the linked label if counts are disabled
            fig.data[0].texttemplate = link_part
        else:
            # Should not be reached due to Literal type hint
            raise ValueError(
                f"Invalid {show_counts=}, must be 'value', 'percent', 'value+percent', "
                "or False"
            )
    # Default behavior when base_url is not provided
    elif show_counts == "percent":
        fig.data[0].textinfo = "label+percent entry"
    elif show_counts == "value":
        fig.data[0].textinfo = "label+value"
        fig.data[0].texttemplate = "%{{label}}<br>%{{value:,}} lines"
    elif show_counts == "value+percent":
        fig.data[0].textinfo = "label+value+percent entry"
        fig.data[
            0
        ].texttemplate = "%{{label}}<br>%{{value:,}} lines<br>%{{percentEntry}}"
    elif show_counts is False:
        # If counts are off and no URL, just show the label
        fig.data[0].textinfo = "label"
    else:
        raise ValueError(
            f"Invalid {show_counts=}, must be 'value', 'percent', 'value+percent', "
            "or False"
        )

    # Adjust margins for better appearance
    fig.layout.margin = dict(l=0, r=0, b=0, t=40, pad=0)
    fig.layout.paper_bgcolor = "rgba(0, 0, 0, 0)"

    return fig
