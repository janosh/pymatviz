"""Hierarchical treemap visualizations of source code packages.

For visualizing module structure and code distribution within Python packages
or any language project via directory scanning.
"""

from __future__ import annotations

import ast
import importlib.metadata
import os
import sys
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
# Numeric module-data columns usable as color_by
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
    homepage = metadata.get("Home-page")  # ty: ignore[missing-attribute] https://github.com/astral-sh/ty/issues/1134
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
    """Count source lines and AST metrics from one read of a Python file."""
    counts = dict.fromkeys(("line_count", *_ANALYSIS_COLS), 0)
    try:
        with open(file_path, encoding="utf-8") as file_handle:
            source = file_handle.read()
        counts["line_count"] = _code_lines(source.split("\n"), ("#",))
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
    # Python import root for internal-import counts, independent of the display
    # label: the outermost enclosing package if the source dir is one (so a
    # subpackage counts `import demo.subpkg` as internal), else each file's top dir
    is_py_package = os.path.isfile(f"{source_dir}/__init__.py")
    package_root = os.path.abspath(source_dir)
    while os.path.isfile(f"{os.path.dirname(package_root)}/__init__.py"):
        package_root = os.path.dirname(package_root)

    ignored = set(ignored_dirs)
    all_modules: list[dict[str, Any]] = []
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = sorted(d for d in dirs if d not in ignored)

        for filename in sorted(files):
            ext = os.path.splitext(filename)[1]
            if ext not in file_extensions:
                continue

            file_path = os.path.normpath(f"{root}/{filename}")
            rel_path = os.path.relpath(file_path, source_dir)
            rel_parts = rel_path.replace(os.sep, "/").split("/")
            stem = os.path.splitext(rel_parts[-1])[0]
            import_root = (
                os.path.basename(package_root) if is_py_package else rel_parts[0]
            )
            stats = (
                _analyze_py_file(file_path, os.path.splitext(import_root)[0])
                if ext in (".py", ".pyi")
                else {
                    "line_count": count_lines(
                        file_path, (_COMMENT_PREFIXES.get(ext, "#"),)
                    ),
                    **dict.fromkeys(_ANALYSIS_COLS, 0),
                }
            )

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
                    "file_path": file_path,
                    "repo_path_segment": repo_path_segment,
                    "leaf_label": leaf_label,
                    "module_parts": module_parts,
                    "depth": len(module_parts),
                    **stats,
                }
            )

    return pd.DataFrame(all_modules)


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
            - str: Name of a numeric column in the module data to use for coloring
                (e.g. "line_count", "n_classes", "n_functions")
            - dict: Numeric values keyed by absolute file path or repo_path_segment.
                Absolute paths take precedence; missing files get zero. Pass
                ``labels={"color_value": "Coverage (%)"}`` for coverage percentages.
            - None: No heatmap coloring (default)
        color_range (tuple[float, float] | None): Manual range for the color scale
            (min, max). Useful for consistent color mapping across plots or to
            ensure specific values (e.g., 0% coverage) map to specific colors.
            If None, uses the actual min/max values from the data.
        cell_border (dict[str, Any] | None): Cell border styling. If None, defaults to
            white borders with width 0.5 for continuous coloring, no borders otherwise.
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
        >>> # Compare multiple packages, coloring cells by function count
        >>> fig2 = pmv.py_pkg_treemap(["pymatviz", "pymatgen"], color_by="n_functions")
        >>> # Rust crate treemap (language-agnostic mode)
        >>> fig3 = pmv.py_pkg_treemap(
        ...     "my_rust_crate", source_dir="src", file_extensions=(".rs",)
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
    depth_was_clamped = effective_depth < data_max_depth
    level_columns = [f"level_{idx}" for idx in range(effective_depth)]

    # Create columns for each level of the hierarchy from 'module_parts'
    for level_idx, col_name in enumerate(level_columns):
        df_treemap[col_name] = df_treemap["module_parts"].map(
            lambda parts, current_idx=level_idx: (
                parts[current_idx] if current_idx < len(parts) else None
            )
        )

    # Adaptive depth pruning from the deepest level upward. A parent with
    # 1..max_children children >= min_lines_for_split is expanded (small children merge
    # into "other"), with no significant children it collapses to one cell.
    # max_children_for_split and the full line threshold only apply at the deepest
    # level, so large modules like io/analysis always show their sub-packages.
    rows_were_merged = False
    if group_by == "module" and (min_lines_for_split > 0 or max_children_for_split):
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
                n_sig = len(significant)
                max_kids = max_children_for_split if is_deepest_level else None
                if n_sig >= 1 and (max_kids is None or n_sig <= max_kids):
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
                if has_color_mode:
                    rep["color_value"] = (
                        merge_rows["color_value"] * merge_rows["cell_value"]
                    ).sum() / rep["cell_value"]
                if is_expand:
                    rep[child_col] = "other"
                else:
                    rep["depth"] = prune_depth
                for deeper in range(clear_start, effective_depth):
                    rep[f"level_{deeper}"] = None
                rows_to_drop.extend(merge_rows.index)
                rows_to_add.append(rep.to_dict())

            if rows_to_drop:
                rows_were_merged = True
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

    # Fill NaNs with 0 for customdata
    fill_cols = analysis_cols + ["color_value"] * has_color_mode
    df_treemap[fill_cols] = df_treemap[fill_cols].fillna(0).infer_objects(copy=False)

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
    if has_color_mode:  # color_by decides the cell colors, whatever plotly kwargs say
        treemap_params["color"] = "color_value"

    # None marks rows that end above the deepest level. plotly>=7 casts path columns
    # to str and max()es them, so object columns mixing str and None (-> NaN) raise;
    # pandas' string dtype keeps the gaps as <NA>.
    df_treemap[path_definition] = df_treemap[path_definition].astype("string")

    fig = px.treemap(**treemap_params)

    # Break long labels at underscores so plotly can wrap text instead of
    # shrinking the font to fit names like "advanced_transformations" on one line.
    trace = fig.data[0]
    if trace.labels is not None:
        wrapped = []
        for label in map(str, trace.labels):
            if len(label) > 18 and "_" in label and " " not in label:
                mid = len(label) // 2
                best_pos = min(
                    (pos for pos, char in enumerate(label) if char == "_"),
                    key=lambda pos: abs(pos - mid),
                )
                wrapped.append(f"{label[: best_pos + 1]}<br>{label[best_pos + 1 :]}")
            else:
                wrapped.append(label)
        fig.data[0].labels = wrapped

    color_hover = ""
    if has_color_mode:
        default_title = (
            color_by.replace("_", " ").title()
            if isinstance(color_by, str)
            else "Color Value"
        )
        color_title = (kwargs.get("labels") or {}).get("color_value", default_title)
        fig.update_coloraxes(
            colorbar=dict(title=color_title, title_side="right", thickness=12)
        )
        color_hover = f"{color_title}: %{{color:,.2f}}<br>"
        if color_range is not None:
            min_val, max_val = color_range
            fig.layout.coloraxis.update(
                cmin=min_val, cmax=max_val, cmid=(min_val + max_val) / 2
            )

    has_depth_limit = depth_was_clamped or rows_were_merged
    if has_depth_limit:
        # Aggregated/collapsed nodes inherit per-file customdata from the first
        # merged row, and Plotly synthetic parents get "(?) " placeholders.
        # Use only Plotly built-in fields that are correct for all nodes.
        hovertemplate = (
            "<b>%{label}</b><br>Value: %{value:,}<br>%{percentEntry}<br>"
            + color_hover
            + "<extra></extra>"
        )
    else:
        # Indices match the order in custom_data_cols
        base_hovertemplate = (
            "<b>%{customdata[2]}</b><br>"  # leaf_label
            "Path: %{customdata[1]}<br>"  # repo_path_segment
            "Cell Value: %{value:,}<br>"  # cell_value
            "Lines: %{customdata[11]:,}<br>"  # line_count
        )

        # Only show Python AST analysis fields when they have data
        analysis_hover = ""
        if df_treemap[analysis_cols].sum().sum() > 0:
            analysis_hover = "".join(
                f"{name}: %{{customdata[{idx}]:,}}<br>"
                for idx, name in enumerate(
                    (
                        "Classes",
                        "Functions",
                        "Methods",
                        "Internal Imports",
                        "External Imports",
                        "Type Imports",
                    ),
                    start=4,
                )
            )
        hovertemplate = (
            base_hovertemplate
            + color_hover
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
    use_links = has_links and not has_depth_limit
    texttemplate = ""
    if use_links or show_counts in ("value", "value+percent"):
        label_part = (
            "<a href='%{customdata[3]}' target='_blank'>%{customdata[2]}</a>"
            if use_links
            else "%{label}"
        )
        texttemplate = label_part + (f"<br>{count_part}" if count_part else "")
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
