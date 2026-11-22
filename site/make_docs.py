"""Generate markdown API docs from docstrings with the standard library's ast module.

Also converts all notebooks in the examples folder to HTML and adds a
language class to the <pre> tag so that syntax highlighting works.
"""  # noqa: INP001

from __future__ import annotations

import ast
import doctest
import json
import os
import re
import subprocess
from glob import glob

import pymatviz as pmv


os.chdir(pmv.utils.ROOT)
with open(f"{pmv.utils.ROOT}/site/package.json") as file:
    pkg = json.load(file)
route = "site/api-docs"


def prose_to_md(doc: str) -> str:
    """Convert the prose of a Google-style docstring (no doctest lines) to markdown."""
    # turn `name (type): desc` and `name: desc` entries into bullets
    doc = re.sub(
        r"^ {4}(\*{0,2}\w+) \((.+?)\):", r"- `\1` (`\2`):", doc, flags=re.MULTILINE
    )
    doc = re.sub(r"^ {4}(\*{0,2}\w+):", r"- `\1`:", doc, flags=re.MULTILINE)
    # bare **kwargs breaks bold (not powers like x**2)
    doc = re.sub(r"(?<![\w`*])\*\*([a-zA-Z_]\w*)", r"`**\1`", doc)
    doc = re.sub(r"\n {6,}", " ", doc)  # join each entry's continuation lines
    return re.sub(r"^(\w+):$", r"**\1**", doc, flags=re.MULTILINE)


def docstring_to_md(node: ast.FunctionDef | ast.ClassDef) -> str:
    """Convert a Google-style docstring to markdown. Each run of doctest examples
    becomes a fenced Python code block (prompts stripped, expected output as
    ``# `` comments), and prose after an example is dedented so markdown doesn't
    render it as an indented code block.
    """
    parts, code_lines, seen_example = [], [], False
    for piece in doctest.DocTestParser().parse(ast.get_docstring(node) or ""):
        if isinstance(piece, doctest.Example):
            code_lines += piece.source.splitlines()
            code_lines += [f"# {line}" for line in piece.want.splitlines()]
            seen_example = True
        elif piece.strip() or not code_lines:  # blank strings separate examples
            if code_lines:
                parts.append("\n```python\n" + "\n".join(code_lines) + "\n```\n\n")
                code_lines = []
            prose = prose_to_md(piece)
            if seen_example:
                prose = re.sub(r"^ +", "", prose, flags=re.MULTILINE)
            parts.append(prose)
    if code_lines:
        parts.append("\n```python\n" + "\n".join(code_lines) + "\n```\n")
    return "".join(parts)


def signature(func: ast.FunctionDef) -> str:
    """Format a function signature without its self argument."""
    args = ast.unparse(func.args).removeprefix("self, ").removeprefix("self")
    returns = f" -> {ast.unparse(func.returns)}" if func.returns else ""
    return f"{func.name}({args}){returns}"


def module_to_md(path: str) -> str:
    """Render the public functions and classes in a module, "" if it has none."""
    with open(path, encoding="utf-8") as file:
        tree = ast.parse(file.read())

    sections = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.ClassDef) or node.name[0] == "_":
            continue
        title = node.name if isinstance(node, ast.ClassDef) else signature(node)
        src_link = f"[source]({pkg['repository']}/blob/main/{path}#L{node.lineno})"
        sections.append(f"## `{title}`\n\n{src_link}\n\n{docstring_to_md(node)}")
        if isinstance(node, ast.ClassDef):
            sections += [
                f"### `{signature(method)}`\n\n{docstring_to_md(method)}"
                for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name[0] != "_"
            ]
    module = path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
    return f"# module `{module}`\n\n" + "\n\n".join(sections) if sections else ""


for path in glob(f"{route}/*.md"):
    os.remove(path)
os.makedirs(route, exist_ok=True)

for pattern in ("[!_]*.py", "__init__.py"):
    for path in glob(f"pymatviz/**/{pattern}", recursive=True):
        if markdown := module_to_md(path):
            md_name = path.replace("/", ".").removesuffix(".py")
            with open(f"{route}/{md_name}.md", "w", encoding="utf-8") as file:
                file.write(markdown)

if not glob(f"{route}/*.md"):
    raise FileNotFoundError(f"no API docs generated in {route}")


# --- Notebook to HTML conversion ---
pattern = "examples/*.ipynb"
notebooks = glob(pattern)
if len(notebooks) == 0:
    raise FileNotFoundError(f"no notebooks found matching {pattern!r}")

cmd = f"jupyter nbconvert --to html {pattern} --no-prompt --template basic"
subprocess.run(cmd, shell=True, check=True)  # noqa: S602

html_paths = glob("examples/*.html")
if len(html_paths) != len(notebooks):
    raise ValueError(
        f"expected {len(notebooks)} HTML files but found {len(html_paths)}"
    )


for file_path in html_paths:
    with open(file_path, encoding="utf-8") as file:
        html = file.read()

    html = html.replace("<pre>", '<pre class="language-python">')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(html)
