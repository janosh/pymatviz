"""This script auto-generates markdown files from Python docstrings using lazydocs
and tweaks the output for
- prettier badges linking to source code on GitHub
- remove bold tags since they break inline code.

It also converts all notebooks in the examples folder to HTML and adds a
language class to the <pre> tag so that syntax highlighting works.
"""

import json
import os
import subprocess
from glob import glob


ROOT = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT)
pkg = json.load(open("site/package.json"))
route = "site/src/routes/api"

for path in glob(f"{route}/*.md"):
    os.remove(path)

subprocess.run(
    f"lazydocs {pkg['name']} --output-path {route} "
    f"--no-watermark --src-base-url {pkg['repository']}/blob/main",
    shell=True,
)

for path in glob(f"{route}/*.md"):
    markdown = open(path).read()
    # remove <b> tags from generated markdown as they break inline code
    markdown = markdown.replace("<b>", "").replace("</b>", "")
    # improve style of badges linking to source code on GitHub
    markdown = markdown.replace(
        'src="https://img.shields.io/badge/-source-cccccc?style=flat-square"',
        'src="https://img.shields.io/badge/source-blue?style=flat" alt="source link"',
    )
    open(path, "w").write(markdown)


# --- Notebook to HTML conversion ---
pattern = "examples/*.ipynb"
notebooks = glob(pattern)
if len(notebooks) == 0:
    raise FileNotFoundError(f"no notebooks found matching {pattern!r}")

cmd = f"jupyter nbconvert --to html {pattern} --no-prompt --template basic"
subprocess.run(cmd, shell=True, check=True)

html_paths = glob("examples/*.html")
if len(html_paths) != len(notebooks):
    raise ValueError(
        f"expected {len(notebooks)} HTML files but found {len(html_paths)}"
    )


for file_path in html_paths:
    with open(file_path) as file:
        html = file.read()

    html = html.replace("<pre>", '<pre class="language-python">')
    with open(file_path, "w") as file:
        file.write(html)
