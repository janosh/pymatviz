import json
import os
from glob import glob
from subprocess import run


# Update auto-generated API docs. Also tweak lazydocs's markdown output for
# - prettier badges linking to source code on GitHub
# - remove bold tags since they break inline code

pkg = json.load(open("site/package.json"))
route = "site/src/routes/api"

for path in glob(f"{route}/*.md"):
    os.remove(path)

run(
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
