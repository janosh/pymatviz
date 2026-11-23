"""Hierarchical treemap visualizations of source code packages.

For visualizing module structure and code distribution within Python packages
or any language project via directory scanning.
"""

from __future__ import annotations

import ast
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from types import ModuleType
from typing import Any, Literal, NamedTuple, get_args

import numpy as np
import pandas as pd

from pymatviz.widgets.treemap import TreemapWidget


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
# ID segment of merged small modules (labeled "other"), distinct from real modules
OTHER_BUCKET = "__other__"

# Python AST counts per module (zero for non-Python sources)
_ANALYSIS_COLS: tuple[str, ...] = (
    "n_classes",
    "n_functions",
    "n_methods",
    "n_internal_imports",
    "n_external_imports",
    "n_type_checking_imports",
)
# Numeric module-data columns usable as color_by
_NUMERIC_COLOR_COLUMNS = ("line_count", "cell_value", "depth", *_ANALYSIS_COLS)


def default_module_formatter(module: str, count: int, total: int) -> str:
    """Default formatter showing module name, line count, and percentage on one line.

    Example output: my_package (1,234 Lines, 15.6%)
    """
    # Avoid ZeroDivisionError if total is 0
    percent_str = f", {count / total:.1%}" if total > 0 else ""
    return f"{module} ({count:,} lines{percent_str})"


def _top_package_name(package: str | ModuleType) -> str:
    """Top-level package name of a package name or module object."""
    return package if isinstance(package, str) else package.__name__.split(".")[0]


def _code_lines(lines: Iterable[str], comment_prefixes: tuple[str, ...]) -> int:
    """Count nonblank lines whose first token is not a comment."""
    return sum(
        1
        for line in lines
        if (code := line.lstrip()) and not code.startswith(comment_prefixes)
    )


def count_lines(file_path: str, comment_prefixes: tuple[str, ...] = ("#",)) -> int:
    """Count non-empty, non-comment lines in a source file.

    Args:
        file_path: Path to the source file.
        comment_prefixes: Comment prefix strings to skip (e.g. ``("#",)`` for
            Python, ``("//",)`` for Rust/C/JS). Defaults to Python-style.
    """
    try:
        with open(file_path, encoding="utf-8") as file_handle:
            return _code_lines(file_handle, comment_prefixes)
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return 0


def find_package_path(package_name: str) -> str:
    """Directory of an installed or sys.path-importable Python package, or "" if
    not found.
    """
    from importlib import resources

    try:
        if os.path.isdir(package_root := str(resources.files(package_name))):
            return package_root
    except (ImportError, TypeError):
        pass  # not importable (or broken): look for its directory on sys.path

    for path_entry in sys.path:
        if os.path.isdir(package_path := f"{path_entry or '.'}/{package_name}"):
            return os.path.abspath(package_path)
    return ""


def _analyze_py_file(file_path: str, package_name: str) -> dict[str, int]:
    """Count source lines and AST metrics from one read of a Python file."""
    counts = dict.fromkeys(("line_count", *_ANALYSIS_COLS), 0)
    try:
        with open(file_path, encoding="utf-8") as file_handle:
            source = file_handle.read()
        counts["line_count"] = _code_lines(source.split("\n"), ("#",))
        tree = ast.parse(source, filename=file_path)

        # Walk statements only (classes, defs, imports and ifs are statements and
        # never nest inside expressions), skipping the ~10x more expression nodes.
        # Parents are visited before their children, so imports inside
        # `if TYPE_CHECKING:` blocks and function defs in class bodies (methods) are
        # marked before they're counted
        type_checking_imports: set[int] = set()
        method_nodes: set[int] = set()
        stack: list[ast.AST] = [tree]
        while stack:
            node = stack.pop()
            for stmt_list in ("body", "orelse", "finalbody", "handlers", "cases"):
                stack += getattr(node, stmt_list, ())
            if isinstance(node, ast.If):
                test = node.test  # `if TYPE_CHECKING:` or `if typing.TYPE_CHECKING:`
                name = test.id if isinstance(test, ast.Name) else None
                name = test.attr if isinstance(test, ast.Attribute) else name
                if name == "TYPE_CHECKING":  # incl. imports nested in e.g. try blocks
                    type_checking_imports |= {
                        id(sub_node)
                        for stmt in node.body
                        for sub_node in ast.walk(stmt)
                    }
            elif isinstance(node, ast.ClassDef):
                counts["n_classes"] += 1
                method_nodes |= {id(item) for item in node.body}
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

    except (
        FileNotFoundError,
        SyntaxError,
        PermissionError,
        UnicodeDecodeError,
        # ast.parse overflows its stack on very deeply nested (e.g. generated) code
        RecursionError,
        MemoryError,
    ) as exc:
        print(f"Warning: Could not analyze {file_path}: {exc}")  # noqa: T201
    return counts


def collect_package_modules(
    package_names: Sequence[str | ModuleType],
    ignored_dirs: tuple[str, ...] = (".venv", ".git", "node_modules"),
) -> pd.DataFrame:
    """Collect information about the public modules of installed Python packages
    (files starting with an underscore like __init__.py are left out).

    Args:
        package_names (Sequence[str | ModuleType]): Names of packages to analyze or
            module objects.
        ignored_dirs (tuple[str, ...]): Directories to ignore when scanning packages.

    Returns:
        pd.DataFrame: One row per module with the ModuleStats columns.

    Raises:
        ValueError: If a package can't be found.
    """
    pkg_frames = []
    for pkg_input in package_names:
        package_name = _top_package_name(pkg_input)
        pkg_path = find_package_path(package_name)
        if not pkg_path:
            raise ValueError(f"Could not find package path for {package_name!r}")
        df_pkg = collect_source_modules_from_dir(
            pkg_path, package_name, ignored_dirs=ignored_dirs
        )
        df_pkg = df_pkg[~df_pkg["filename"].str.startswith("_")].copy()
        # repo paths of src-layout packages (repo/src/pkg_name) start at src/
        if os.path.basename(os.path.dirname(pkg_path)) == "src":
            df_pkg["repo_path_segment"] = "src/" + df_pkg["repo_path_segment"]
        pkg_frames.append(df_pkg)
    return pd.concat(pkg_frames, ignore_index=True)


def collect_source_modules_from_dir(
    source_dir: str,
    project_name: str,
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

    Walks the given source directory and builds a module hierarchy from the file
    system structure. Works for any language (Python, Rust, TypeScript, Go, ...).

    Args:
        source_dir: Root directory to scan for source files.
        project_name: Display name of the project (the package column).
        file_extensions: File extensions to include, e.g. ``(".rs",)`` for
            Rust or ``(".ts", ".tsx")`` for TypeScript.
        ignored_dirs: Directory names to skip during traversal.

    Returns:
        pd.DataFrame: One row per source file (sorted by path) with the ModuleStats
            columns.

    Raises:
        FileNotFoundError: If source_dir is not a directory.
    """
    source_dir = os.path.abspath(source_dir)
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    source_dir_name = os.path.basename(source_dir)
    # Python import root for internal-import counts, independent of the display
    # label: the outermost enclosing package if the source dir is one (so a
    # subpackage counts `import demo.subpkg` as internal), else each file's top dir
    is_py_package = os.path.isfile(f"{source_dir}/__init__.py")
    package_root = source_dir
    while os.path.isfile(f"{os.path.dirname(package_root)}/__init__.py"):
        package_root = os.path.dirname(package_root)

    ignored = set(ignored_dirs)
    all_modules: list[dict[str, Any]] = []
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = sorted(dirname for dirname in dirs if dirname not in ignored)

        for filename in sorted(files):
            ext = os.path.splitext(filename)[1]
            if ext not in file_extensions:
                continue

            file_path = os.path.normpath(f"{root}/{filename}")
            rel_path = os.path.relpath(file_path, source_dir).replace(os.sep, "/")
            rel_parts = rel_path.split("/")
            stem = os.path.splitext(rel_parts[-1])[0]
            import_root = (
                os.path.basename(package_root) if is_py_package else rel_parts[0]
            )
            if ext in (".py", ".pyi"):
                stats = _analyze_py_file(file_path, os.path.splitext(import_root)[0])
            else:
                comment_prefixes = (_COMMENT_PREFIXES.get(ext, "#"),)
                line_count = count_lines(file_path, comment_prefixes)
                stats = {"line_count": line_count} | dict.fromkeys(_ANALYSIS_COLS, 0)

            # Every file gets a unique leaf in the hierarchy. A file sharing its stem
            # with a sibling directory (Rust new-style module roots like burnside.rs
            # + burnside/) nests inside the directory to avoid leaf/branch collisions.
            module_parts = [*rel_parts[:-1], stem]
            if os.path.isdir(f"{root}/{stem}"):
                module_parts.append(stem)

            all_modules.append(
                {
                    "package": project_name,
                    "full_module": f"{project_name}.{'.'.join(module_parts)}",
                    "filename": stem,
                    "directory": rel_parts[0] if len(rel_parts) > 1 else "root",
                    "top_module": module_parts[0],
                    "file_path": file_path,
                    "repo_path_segment": f"{source_dir_name}/{rel_path}",
                    "leaf_label": module_parts[-1],
                    "module_parts": module_parts,
                    "depth": len(module_parts),
                    **stats,
                }
            )

    return pd.DataFrame(all_modules, columns=pd.Index(ModuleStats._fields))


def py_pkg_treemap(
    packages: str | ModuleType | Sequence[str | ModuleType],
    *,
    source_dir: str | None = None,
    file_extensions: tuple[str, ...] = (".py",),
    cell_text_fn: ModuleFormatter | Literal[False] = default_module_formatter,
    group_by: GroupBy = "module",
    max_module_depth: int | None = None,
    min_lines_for_split: int = 0,
    max_children_for_split: int | None = None,
    cell_size_fn: CellSizeFn | None = None,
    color_by: str | dict[str, float] | None = None,
    color_range: tuple[float, float] | None = None,
    **kwargs: Any,
) -> TreemapWidget:
    """Generate a treemap plot showing module structure of source code packages.

    Works with any language when ``source_dir`` is provided. Without it,
    defaults to Python package introspection via importlib.

    The first level shows the distribution by package (by top-level module for a
    single package), the next ones modules or files, with cell sizes typically
    indicating code size.

    Args:
        packages (str | ModuleType | Sequence[str | ModuleType]): Single package
            name/module or list of package names/modules to analyze. When
            ``source_dir`` is provided, a single name used only as display label.
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
        cell_text_fn (ModuleFormatter | False): Labels top-level names from (name,
            count, total count), or False for bare names. Defaults to
            default_module_formatter.
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
        color_by (str | dict[str, float] | None): Color cells by a value on the
            viridis scale. Branch cells get the cell-size-weighted mean of their
            leaves' values.
            - str: Name of a numeric column in the module data to use for coloring
                (e.g. "line_count", "n_classes", "n_functions")
            - dict: Numeric values keyed by absolute file path or repo_path_segment
                (e.g. test coverage). Absolute paths take precedence; missing files
                get zero.
            - None (default): one palette color per top-level module, deeper cells
                a translucent white overlay on it
        color_range (tuple[float, float] | None): Values mapped to the ends of the
            color scale, e.g. (0, 100) for coverage percentages. Values outside are
            clamped. If None, uses the data's min/max.
        **kwargs (Any): Passed to TreemapWidget, e.g. ``label_text`` (default
            ``"label+value"``), ``value_format`` (default ``".3~s"``, e.g. 54.3k
            lines), ``max_depth`` or ``style``.

    Returns:
        TreemapWidget: The MatterViz treemap.

    Example:
        >>> import pymatviz as pmv
        >>> # Analyze a single package
        >>> fig1 = pmv.py_pkg_treemap("pymatviz")
        >>> # Compare multiple packages, coloring cells by function count
        >>> fig2 = pmv.py_pkg_treemap(["pymatviz", "kemio"], color_by="n_functions")
        >>> # Rust crate treemap (language-agnostic mode, needs that crate's src dir)
        >>> fig3 = pmv.py_pkg_treemap(
        ...     "my_crate", source_dir="path/to/my_crate/src", file_extensions=(".rs",)
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
    if isinstance(color_by, dict) and not color_by:
        raise ValueError("color_by dict must map file paths to values, got empty dict")

    pkg_list = [packages] if isinstance(packages, str | ModuleType) else packages
    package_names = [_top_package_name(pkg) for pkg in pkg_list]

    if source_dir:
        if len(package_names) != 1:
            raise ValueError(
                f"source_dir scans one project, got {len(package_names)} names "
                f"{package_names}"
            )
        df_modules = collect_source_modules_from_dir(
            source_dir, package_names[0], file_extensions
        )
    else:
        df_modules = collect_package_modules(package_names)

    df_modules["cell_value"] = (
        df_modules["line_count"]
        if cell_size_fn is None
        else [
            cell_size_fn(ModuleStats(*row))
            for row in df_modules.itertuples(index=False)
        ]
    )
    # cell_size_fn returning 0 omits a module
    df_treemap = df_modules[df_modules["cell_value"] > 0].copy()
    if df_treemap.empty:
        location = source_dir or package_names
        raise ValueError(f"No source files of nonzero cell size in {location}")

    has_color_mode = color_by is not None
    if isinstance(color_by, dict):
        df_treemap["color_value"] = [
            color_by.get(path, color_by.get(segment, 0.0))
            for path, segment in zip(
                df_treemap["file_path"], df_treemap["repo_path_segment"], strict=True
            )
        ]
    elif isinstance(color_by, str) and color_by in _NUMERIC_COLOR_COLUMNS:
        df_treemap["color_value"] = df_treemap[color_by]  # Use existing column
    elif color_by is not None:
        raise ValueError(
            f"Invalid {color_by=}, must be a dict of file path to "
            f"value, or one of the numeric module columns {_NUMERIC_COLOR_COLUMNS}"
        )

    # Determine the maximum depth of the module hierarchy
    data_max_depth = effective_depth = df_treemap["depth"].max()
    if max_module_depth is not None and group_by == "module":
        effective_depth = min(data_max_depth, max_module_depth)
    level_columns = [f"level_{idx}" for idx in range(effective_depth)]

    # Create columns for each level of the hierarchy from 'module_parts'
    for level_idx, col_name in enumerate(level_columns):
        df_treemap[col_name] = [
            parts[level_idx] if level_idx < len(parts) else None
            for parts in df_treemap["module_parts"]
        ]

    # Adaptive depth pruning from the deepest level upward. A parent with
    # 1..max_children children >= min_lines_for_split is expanded (small children merge
    # into "other"), with no significant children it collapses to one cell.
    # max_children_for_split and the full line threshold only apply at the deepest
    # level, so large modules like io/analysis always show their sub-packages.
    is_pruning = min_lines_for_split > 0 or max_children_for_split is not None
    if group_by == "module" and is_pruning:
        for prune_depth in range(effective_depth - 1, 0, -1):
            is_deepest_level = prune_depth == effective_depth - 1
            child_col = f"level_{prune_depth}"
            parent_cols = [f"level_{idx}" for idx in range(prune_depth)]

            rows_to_drop: list[int] = []
            rows_to_add: list[dict] = []

            for _key, group in df_treemap.groupby(parent_cols, dropna=False):
                unique_children = group[child_col].dropna().unique()
                if len(unique_children) <= 1:
                    continue

                child_line_counts = group.groupby(child_col)["line_count"].sum()
                effective_min = (
                    min_lines_for_split
                    if is_deepest_level
                    else min(min_lines_for_split, 500)
                )
                significant = child_line_counts[child_line_counts >= effective_min]
                n_significant = len(significant)
                max_children = max_children_for_split if is_deepest_level else None
                if n_significant >= 1 and (
                    max_children is None or n_significant <= max_children
                ):
                    # EXPAND: keep significant children, merge the rest
                    small_mask = ~group[child_col].isin(significant.index)
                    merge_rows = group[small_mask]
                    if merge_rows.empty:
                        continue
                    clear_start, is_expand = prune_depth + 1, True
                else:
                    # COLLAPSE: no children are significant, merge entire group
                    merge_rows = group
                    clear_start, is_expand = prune_depth, False

                merged_row = merge_rows.iloc[0].copy()
                merged_value = merged_row["cell_value"] = merge_rows["cell_value"].sum()
                merged_row["line_count"] = merge_rows["line_count"].sum()
                if has_color_mode:
                    weighted = merge_rows["color_value"] * merge_rows["cell_value"]
                    merged_row["color_value"] = weighted.sum() / merged_value
                if is_expand:
                    merged_row[child_col] = OTHER_BUCKET
                else:
                    merged_row["depth"] = prune_depth
                for deeper in range(clear_start, effective_depth):
                    merged_row[f"level_{deeper}"] = None
                rows_to_drop.extend(merge_rows.index)
                rows_to_add.append(merged_row.to_dict())

            if rows_to_drop:
                df_treemap = pd.concat(
                    [df_treemap.drop(index=rows_to_drop), pd.DataFrame(rows_to_add)],
                    ignore_index=True,
                )

            # Drop level columns that became entirely None after pruning
            level_columns = [
                col for col in level_columns if df_treemap[col].notna().any()
            ]

    path_definition = {
        "file": ["package", "filename"],
        "directory": ["package", "directory", "filename"],
    }.get(group_by, ["package", *level_columns])

    package_totals = df_treemap.groupby("package")["cell_value"].sum()
    total_value = package_totals.sum()
    package_labels = {
        pkg: cell_text_fn(pkg, pkg_total, total_value) if cell_text_fn else pkg
        for pkg, pkg_total in package_totals.items()
    }

    roots = _module_tree(
        df_treemap[path_definition].to_numpy(dtype=object),
        df_treemap["cell_value"].to_numpy(dtype=float),
        df_treemap["color_value"].fillna(0).to_numpy(dtype=float)
        if has_color_mode
        else None,
        package_labels,
    )
    if has_color_mode:
        _color_tree(roots, color_range)
    if len(roots) == 1 and "children" in roots[0]:
        # one package: its modules are the top level, each in its own color
        roots = roots[0]["children"]
    if not has_color_mode:
        _shade_by_module(roots)
    return TreemapWidget(roots, **_SOURCE_MAP_STYLE | kwargs)


# Mid-tones that read on light and dark pages, one per top-level module
SOURCE_MAP_PALETTE = (
    *("#1f8a9e", "#d97f0a", "#5f6b78", "#7a5ea8", "#3f8f5a", "#b5523b", "#2f6db5"),
    *("#a3790f", "#8a4f7d", "#4d8a8a", "#6b7f2a", "#9a5a2e", "#53607a", "#2e8f80"),
)
# Bold module headers, then top-left names with compact line counts (e.g. 54.3k) below,
# in white on rounded cells separated by small gaps
_SOURCE_MAP_STYLE: dict[str, Any] = {
    "label_text": "label+value",
    "value_format": ".3~s",
    "label_align": "top-left",
    "label_max_font_size": 11,
    "padding_inner": 2,
    "padding_outer": 3,
    "padding_top": 22,
    "style": (
        "--treemap-label-color: white; --treemap-cell-radius: 3px; "
        "--treemap-cell-stroke: none; font-family: Inter, -apple-system, "
        "'Segoe UI', Helvetica, Arial, sans-serif"
    ),
}


def _shade_by_module(roots: list[dict[str, Any]]) -> None:
    """Color each top-level node from SOURCE_MAP_PALETTE (sorted by size, largest
    first) and draw its descendants as a translucent white overlay on that color.
    """

    def total(node: dict[str, Any]) -> float:
        """Summed leaf value of a node."""
        return node.get("value", 0) + sum(map(total, node.get("children", [])))

    def overlay(nodes: list[dict[str, Any]]) -> None:
        """Translucent white fill for nodes and their descendants."""
        for node in nodes:
            node["color"] = "rgba(255, 255, 255, 0.13)"
            overlay(node.get("children", []))

    for idx, root in enumerate(sorted(roots, key=total, reverse=True)):
        root["color"] = SOURCE_MAP_PALETTE[idx % len(SOURCE_MAP_PALETTE)]
        overlay(root.get("children", []))


# viridis at 9 evenly spaced stops (matplotlib's colormap)
_VIRIDIS = np.array(
    [
        [68, 1, 84],
        [71, 45, 123],
        [59, 82, 139],
        [44, 114, 142],
        [33, 145, 140],
        [40, 174, 128],
        [94, 201, 98],
        [173, 220, 48],
        [253, 231, 37],
    ],
    dtype=float,
)


def _viridis_hex(fraction: float) -> str:
    """Hex color at a fraction in [0, 1] (clamped) of the viridis scale."""
    pos = min(max(fraction, 0.0), 1.0) * (len(_VIRIDIS) - 1)
    lower = min(int(pos), len(_VIRIDIS) - 2)
    red, green, blue = np.rint(
        _VIRIDIS[lower] + (pos - lower) * (_VIRIDIS[lower + 1] - _VIRIDIS[lower])
    ).astype(int)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _module_tree(
    paths: np.ndarray,
    cell_values: np.ndarray,
    color_values: np.ndarray | None,
    package_labels: dict[str, str],
) -> list[dict[str, Any]]:
    """Nest rows of hierarchy levels (missing trailing levels are None or NA) into
    treemap nodes keyed by slash-joined IDs. Rows with the same path are summed.
    With color_values, ``_color`` holds each leaf's cell-value-weighted mean.

    Raises:
        ValueError: If a path is both a leaf and a branch.
    """
    roots: dict[str, dict[str, Any]] = {}
    if color_values is None:
        color_values = np.full(len(paths), np.nan)
    for row, cell_value, color_value in zip(
        paths, cell_values, color_values, strict=True
    ):
        parts = [str(part) for part in row if not pd.isna(part)]
        siblings, parent_id = roots, ""
        for depth, part in enumerate(parts):
            node_id = f"{parent_id}/{part}" if parent_id else part
            label = (
                package_labels[part]
                if depth == 0
                else "other"
                if part == OTHER_BUCKET
                else part
            )
            node = siblings.setdefault(node_id, {"id": node_id, "label": label})
            if depth == len(parts) - 1:
                if "_children" in node:
                    raise ValueError(f"Treemap path {node_id!r} is a leaf and a branch")
                weight = node.get("value", 0.0)
                node["value"] = weight + cell_value
                if not np.isnan(color_value):
                    prev_weighted = node.get("_color", 0.0) * weight
                    added_weighted = color_value * cell_value
                    node["_color"] = (prev_weighted + added_weighted) / node["value"]
            else:
                if "value" in node:
                    raise ValueError(f"Treemap path {node_id!r} is a leaf and a branch")
                siblings = node.setdefault("_children", {})
            parent_id = node_id

    def finalize(nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """Turn child dicts into lists."""
        for node in nodes.values():
            if "_children" in node:
                node["children"] = finalize(node.pop("_children"))
        return list(nodes.values())

    return finalize(roots)


def _color_tree(
    roots: list[dict[str, Any]], color_range: tuple[float, float] | None
) -> None:
    """Color nodes by viridis, branches by the value-weighted mean of their leaves,
    and drop the temporary ``_color`` keys.
    """

    def weighted(node: dict[str, Any]) -> tuple[float, float]:
        """Store branch means and return (total value, weighted color sum)."""
        if children := node.get("children"):
            total, color_sum = map(sum, zip(*map(weighted, children), strict=True))
            node["_color"] = color_sum / total
            return total, color_sum
        return node["value"], node["_color"] * node["value"]

    def walk(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten the tree."""
        return [
            desc for node in nodes for desc in (node, *walk(node.get("children", [])))
        ]

    for root in roots:
        weighted(root)
    all_nodes = walk(roots)
    if color_range is None:
        values = [node["_color"] for node in all_nodes]
        color_range = (min(values), max(values))
    min_val, max_val = color_range
    span = max_val - min_val
    for node in all_nodes:
        value = node.pop("_color")
        fraction = (value - min_val) / span if span else 0.5
        node["color"] = _viridis_hex(fraction)
