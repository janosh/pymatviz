"""Demonstrate treemap visualizations of Python package structures.

This script shows how to create interactive treemaps visualizing module
structures of Python packages using the pmv.py_pkg_treemap function.
"""

# /// script
# dependencies = [
#     "pymatgen>=2026.9.24",
# ]
# ///

# %%
from __future__ import annotations

import json
import os
from urllib.request import urlopen

import numpy as np
import pymatgen

import pymatviz as pmv
from pymatviz.treemap import py_pkg


# %% Single packages with default settings
for pkg in ("pymatviz", np):  # Mix strings and module objects
    widget = pmv.py_pkg_treemap(pkg, style="height: 500px")
    widget.show()
    pkg_name = str(getattr(pkg, "__name__", pkg)).replace("_", "-")
    if pkg_name == "pymatviz":  # in readme
        widget.to_img(
            f"{pmv.utils.ROOT}/assets/svg/py-pkg-treemap-{pkg_name}.svg",
            width=800,
            transparent=True,
        )


# %% Compare multiple packages, only including files with at least 50 lines
widget = pmv.py_pkg_treemap(
    ("pymatviz", np),
    cell_size_fn=lambda cell: cell.line_count if cell.line_count >= 50 else 0,
    label_text="label+value",
    padding_inner=2,
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/py-pkg-treemap-multiple.svg",
    width=800,
    transparent=True,
)


# %% Different ways to group packages
for group_by in ("file", "directory", "module"):
    pmv.py_pkg_treemap(
        "pymatviz",
        group_by=group_by,
        cell_size_fn=lambda cell: cell.line_count if cell.line_count >= 20 else 0,
    ).show()


# %% Custom formatting of package names and line counts
def custom_module_formatter(module: str, count: int, _total: int) -> str:
    """Custom formatter that emphasizes the module name."""
    return f"{module.upper()} [{count:,} lines]"


pmv.py_pkg_treemap(("numpy", "pymatviz"), cell_text_fn=custom_module_formatter).show()


# %% Custom cell sizing by functions, classes, and methods
pmv.py_pkg_treemap(
    "pymatviz",
    cell_size_fn=lambda cell: cell.n_functions + cell.n_classes + cell.n_methods,
    label_text="label+value",
).show()


# %% Color files from published coverage.py JSON reports
def coverage_percentages(url: str, package: str) -> dict[str, float]:
    """Map each reported file to its installed path for this gallery example."""
    with urlopen(url, timeout=30) as response:  # noqa: S310
        files = json.load(response)["files"]
    pkg_dir = py_pkg.find_package_path(package)
    prefix = f"{package}/"
    return {
        # normpath matches the collector's file_path (e.g. backslashes on Windows)
        os.path.normpath(f"{pkg_dir}/{path.partition(prefix)[2]}"): record["summary"][
            "percent_covered"
        ]
        for path, record in files.items()
        if prefix in path
    }


coverage_url = "https://github.com/janosh/pymatviz/releases/download/v0.17.5/2026-03-31-pymatviz-coverage.json"

widget = pmv.py_pkg_treemap(
    "pymatviz",
    color_by=coverage_percentages(coverage_url, "pymatviz"),
    color_range=(0, 100),  # viridis from 0% (purple) to 100% (yellow) coverage
    cell_size_fn=lambda cell: cell.line_count if cell.line_count >= 20 else 0,
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/py-pkg-treemap-pymatviz-coverage.svg",
    width=800,
    transparent=True,
)


# %% pymatgen treemap with coverage heatmap and manual color range (0-100%)
coverage_url = "https://github.com/user-attachments/files/21545087/2025-07-31-pymatgen-coverage.json"

# Namespace packages have their own repos and coverage reports, so they show
# as 0% coverage here which is misleading. Exclude them.
NAMESPACE_PKGS = {"analysis.defects", "analysis.diffusion"}


def _pmg_cell_size(cell: py_pkg.ModuleStats) -> int:
    """Filter out namespace packages and tiny files."""
    dotted = f".{cell.full_module}."
    if any(f".{ns_pkg}." in dotted for ns_pkg in NAMESPACE_PKGS):
        return 0
    return cell.line_count if cell.line_count >= 20 else 0


widget = pmv.py_pkg_treemap(
    pymatgen,
    color_by=coverage_percentages(coverage_url, "pymatgen"),
    color_range=(0, 100),
    cell_size_fn=_pmg_cell_size,
    max_module_depth=3,
    min_lines_for_split=500,  # only expand if every child has >=500 lines
    max_children_for_split=5,  # only expand if parent has <=5 children
)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/py-pkg-treemap-pymatgen-coverage.svg",
    width=800,
    transparent=True,
)
