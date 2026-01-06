# noqa: INP001
"""Auto-update Markdown links to Python function definitions.

Rewrites [`func(args)`](path/module.py#L123) by finding `def func` line numbers.
Also validates all .py links point to files that exist.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_PY_ANCHOR_RE = re.compile(r"^(?P<path>.+\.py)#L\d+(?:-L\d+)?$")
_PY_LINK_RE = re.compile(r"^(?P<path>.+\.py)(?:#L\d+(?:-L\d+)?)?$")
_FUNC_NAME_RE = re.compile(r"`?(?P<name>[A-Za-z_]\w*)\s*\(")


def find_def_line(path: Path, func_name: str) -> int | None:
    """Find line number of a function definition using AST."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ):
            return node.lineno
    return None


def update_markdown(
    text: str, root: Path, *, check_only: bool = False
) -> tuple[str, int, list[str]]:
    """Update .py#L links and validate .py files exist."""
    count = 0
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        link_text, url = match.group(1), match.group(2)

        py_match = _PY_LINK_RE.match(url)
        if not py_match:
            return match.group(0)

        rel_path = Path(py_match.group("path"))
        abs_path = (root / rel_path).resolve()
        if not abs_path.exists():
            missing.append(f"{url} -> {abs_path}")
            return match.group(0)

        if not _PY_ANCHOR_RE.match(url):
            return match.group(0)

        func_match = _FUNC_NAME_RE.search(link_text)
        if not func_match:
            return match.group(0)

        lineno = find_def_line(abs_path, func_match.group("name"))
        if lineno is None:
            return match.group(0)

        new_url = f"{rel_path.as_posix()}#L{lineno}"
        if new_url == url:
            return match.group(0)

        count += 1
        return match.group(0) if check_only else f"[{link_text}]({new_url})"

    return _MD_LINK_RE.sub(replace, text), count, missing


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--readme", type=Path, default=Path("readme.md"))
    parser.add_argument("--check", action="store_true", help="Exit 1 if updates needed")
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    readme = (root / args.readme).resolve()
    original = readme.read_text(encoding="utf-8")
    updated, n_updated, missing = update_markdown(original, root, check_only=args.check)

    if missing:
        print("Missing .py files:\n" + "\n".join(f"  - {m}" for m in missing))  # noqa: T201
        return 1

    if args.check:
        if n_updated:
            print(f"{n_updated} link(s) need updating.")  # noqa: T201
            return 1
        print("All links up to date.")  # noqa: T201
        return 0

    if updated != original:
        readme.write_text(updated, encoding="utf-8")
        print(f"Updated {n_updated} link(s).")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
