"""Generate markdown API docs from docstrings with the standard library's ast module.

Also converts all notebooks in the examples folder to HTML and adds a
language class to the <pre> tag so that syntax highlighting works.
"""  # noqa: INP001

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from glob import glob

import pymatviz as pmv


os.chdir(pmv.ROOT)
with open(f"{pmv.ROOT}/site/package.json") as file:
    pkg = json.load(file)
route = "site/api-docs"


def docstring_to_md(node: ast.FunctionDef | ast.ClassDef) -> str:
    """Convert a Google-style docstring to markdown."""
    doc = ast.get_docstring(node) or ""
    # turn `name (type): desc` and `name: desc` entries into bullets
    doc = re.sub(
        r"^ {4}(\*{0,2}\w+) \((.+?)\):", r"- `\1` (`\2`):", doc, flags=re.MULTILINE
    )
    doc = re.sub(r"^ {4}(\*{0,2}\w+):", r"- `\1`:", doc, flags=re.MULTILINE)
    doc = re.sub(r"(?<![`*])\*\*(\w+)", r"`**\1`", doc)  # bare **kwargs breaks bold
    doc = re.sub(r"\n {6,}", " ", doc)  # join each entry's continuation lines
    return re.sub(r"^(\w+):$", r"**\1**", doc, flags=re.MULTILINE)


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
