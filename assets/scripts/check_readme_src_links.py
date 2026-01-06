"""Auto-update Markdown links to Python function definitions.

This script rewrites Markdown links like:
    [`structure_3d(hea_structure)`](pymatviz/structure/plotly.py#L318)

by locating the current line number of:
    `def structure_3d(...)`

in the referenced Python module and updating the '#L...' anchor accordingly.

TODO:
- add to link check workflow
- non-existent func def doesn't raise
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


_MD_LINK_RE: re.Pattern[str] = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_PY_LINE_ANCHOR_RE: re.Pattern[str] = re.compile(r"^(?P<path>.+\.py)#L\d+(?:-L\d+)?$")
_FUNC_NAME_IN_TEXT_RE: re.Pattern[str] = re.compile(r"`?(?P<name>[A-Za-z_]\w*)\s*\(")


def extract_func_name(link_text: str) -> str | None:
    """Extract a function name from Markdown link text."""
    match = _FUNC_NAME_IN_TEXT_RE.search(link_text)
    return match.group("name") if match else None


def find_def_line(module_path: Path, func_name: str) -> int | None:
    """Find the 1-based line number of 'def <func_name>' in a Python file."""
    pattern = re.compile(rf"^\s*def\s+{re.escape(func_name)}\b")
    for idx, line in enumerate(
        module_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if pattern.search(line):
            return idx
    return None


def update_markdown(md_text: str, repo_root: Path) -> tuple[str, int]:
    """Update eligible Markdown links in the given text."""
    updated_count = 0
    warned: set[tuple[Path, str]] = set()

    def replace(match: re.Match[str]) -> str:
        nonlocal updated_count

        link_text = match.group(1)
        url = match.group(2)

        if not _PY_LINE_ANCHOR_RE.match(url):
            return match.group(0)

        func_name = extract_func_name(link_text)
        if not func_name:
            return match.group(0)

        rel_path = Path(url.split("#", 1)[0])
        abs_path = (repo_root / rel_path).resolve()
        if not abs_path.exists():
            return match.group(0)

        lineno = find_def_line(abs_path, func_name)
        if lineno is None:
            key = (abs_path, func_name)
            if key not in warned:
                # TODO: this doesn't seem to work
                print(
                    f"WARNING: function '{func_name}' not found in "
                    f"{abs_path.relative_to(repo_root)}",
                    file=sys.stderr,
                )
                warned.add(key)
            return match.group(0)

        new_url = f"{rel_path.as_posix()}#L{lineno}"
        if new_url == url:
            return match.group(0)

        updated_count += 1
        return f"[{link_text}]({new_url})"

    updated_text = _MD_LINK_RE.sub(replace, md_text)
    return updated_text, updated_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-update Markdown .py#L... links.")
    parser.add_argument(
        "--repo-root", type=Path, default=Path("."), help="Repository root."
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Markdown file to update.",
    )

    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    readme_path = (repo_root / args.readme).resolve()

    original = readme_path.read_text(encoding="utf-8")
    updated, n_updated = update_markdown(original, repo_root)

    if updated == original:
        return 0

    readme_path.write_text(updated, encoding="utf-8")
    print(f"Updated {n_updated} link(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
