"""Auto-update Markdown links to Python function definitions.

This script rewrites Markdown links like:
    [`function(args)`](pymatviz/package/module.py#L123)

by locating the line number of:
    `def function(...)`

in the referenced Python module and updating the '#L...' anchor accordingly.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


_MD_LINK_RE: re.Pattern[str] = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_PY_LINE_ANCHOR_RE: re.Pattern[str] = re.compile(r"^(?P<path>.+\.py)#L\d+(?:-L\d+)?$")
_FUNC_NAME_IN_TEXT_RE: re.Pattern[str] = re.compile(r"`?(?P<name>[A-Za-z_]\w*)\s*\(")


def update_markdown(md_text: str, repo_root: Path) -> tuple[str, int]:
    """Update eligible links in the given markdown text."""

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

    updated_count = 0
    missing_files: list[str] = []

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
            missing_files.append(f"{url} -> {abs_path} (func: {func_name})")
            return match.group(0)

        lineno = find_def_line(abs_path, func_name)
        if lineno is None:
            return match.group(0)

        new_url = f"{rel_path.as_posix()}#L{lineno}"
        if new_url == url:
            return match.group(0)

        updated_count += 1
        return f"[{link_text}]({new_url})"

    updated_text = _MD_LINK_RE.sub(replace, md_text)

    # Fail at the end if any missing files were encountered
    if missing_files:
        msg = "Missing target .py files referenced by Markdown links:\n" + "\n".join(
            f"- {item}" for item in sorted(set(missing_files))
        )
        raise FileNotFoundError(msg)

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
