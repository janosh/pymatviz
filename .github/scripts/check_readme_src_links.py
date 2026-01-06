# noqa: INP001
"""Auto-update Markdown links to Python function definitions.

This script rewrites Markdown links like:
    [`function(args)`](pymatviz/package/module.py#L123)

by locating the line number of:
    `def function(...)`

in the referenced Python module and updating the '#L...' anchor accordingly.
It also validates that all `.py` file links (with or without line anchors) point
to files that actually exist.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


_MD_LINK_RE: re.Pattern[str] = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_PY_LINE_ANCHOR_RE: re.Pattern[str] = re.compile(r"^(?P<path>.+\.py)#L\d+(?:-L\d+)?$")
_PY_LINK_RE: re.Pattern[str] = re.compile(r"^(?P<path>.+\.py)(?:#L\d+(?:-L\d+)?)?$")
_FUNC_NAME_IN_TEXT_RE: re.Pattern[str] = re.compile(r"`?(?P<name>[A-Za-z_]\w*)\s*\(")


def extract_func_name(link_text: str) -> str | None:
    """Extract a function name from Markdown link text.

    Args:
        link_text: The text portion of a Markdown link, e.g. "`function_name(args)`"

    Returns:
        The function name if found, otherwise None.
    """
    match = _FUNC_NAME_IN_TEXT_RE.search(link_text)
    return match.group("name") if match else None


def find_def_line(module_path: Path, func_name: str) -> int | None:
    """Find line number of function definition using AST parsing.

    Uses Python's AST module for robust parsing that handles decorators,
    comments, and edge cases better than regex.

    Args:
        module_path: Path to the Python source file.
        func_name: Name of the function to find.

    Returns:
        The 1-based line number where the function is defined, or None if not found.
    """
    code = module_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node.lineno
    return None


def update_markdown(
    md_text: str, repo_root: Path, *, check_only: bool = False
) -> tuple[str, int, list[str]]:
    """Update eligible links in the given markdown text.

    Args:
        md_text: The markdown content to process.
        repo_root: Root directory of the repository for resolving relative paths.
        check_only: If True, only report what would change without modifying.

    Returns:
        A tuple of (updated_text, update_count, missing_files_list).
    """
    updated_count = 0
    missing_files: list[str] = []

    def replace(match: re.Match[str]) -> str:
        nonlocal updated_count

        link_text = match.group(1)
        url = match.group(2)

        # Check if this is any .py link (with or without anchor)
        py_match = _PY_LINK_RE.match(url)
        if not py_match:
            return match.group(0)

        # Validate that the .py file exists
        rel_path_str = py_match.group("path")
        rel_path = Path(rel_path_str)
        abs_path = (repo_root / rel_path).resolve()

        if not abs_path.exists():
            missing_files.append(f"{url} -> {abs_path}")
            return match.group(0)

        # Only update line numbers for links that have #L anchors
        if not _PY_LINE_ANCHOR_RE.match(url):
            return match.group(0)

        func_name = extract_func_name(link_text)
        if not func_name:
            return match.group(0)

        lineno = find_def_line(abs_path, func_name)
        if lineno is None:
            return match.group(0)

        new_url = f"{rel_path.as_posix()}#L{lineno}"
        if new_url == url:
            return match.group(0)

        updated_count += 1
        if check_only:
            return match.group(0)  # Don't modify in check mode
        return f"[{link_text}]({new_url})"

    updated_text = _MD_LINK_RE.sub(replace, md_text)

    return updated_text, updated_count, missing_files


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the script.

    Args:
        argv: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Auto-update Markdown .py#L... links and validate .py links."
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path("."), help="Repository root."
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("readme.md"),
        help="Markdown file to update.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: exit 1 if updates are needed, without modifying files.",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    readme_path = (repo_root / args.readme).resolve()

    original = readme_path.read_text(encoding="utf-8")
    updated, n_updated, missing_files = update_markdown(
        original, repo_root, check_only=args.check
    )

    # Report missing files
    if missing_files:
        msg = "Missing target .py files referenced by Markdown links:\n" + "\n".join(
            f"  - {item}" for item in sorted(set(missing_files))
        )
        print(msg)  # noqa: T201
        return 1

    if args.check:
        if n_updated > 0:
            print(f"Check failed: {n_updated} link(s) need updating.")  # noqa: T201
            return 1
        print("All links are up to date.")  # noqa: T201
        return 0

    if updated == original:
        return 0

    readme_path.write_text(updated, encoding="utf-8")
    print(f"Updated {n_updated} link(s).")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
