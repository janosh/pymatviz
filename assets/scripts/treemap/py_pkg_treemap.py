"""Demonstrate treemap visualizations of Python package structures.

This script shows how to create interactive treemaps visualizing module
structures of Python packages using the py_pkg_treemap function.
"""

# %%
from __future__ import annotations

import plotly.express as px

import pymatviz as pmv


pmv.set_plotly_template("plotly_white")


# %% single package with default settings
for package in ("pymatviz", "numpy", "pymatgen"):
    fig = pmv.py_pkg_treemap(package)
    fig.layout.title.update(text=f"{package} Package Structure", font_size=20, x=0.5)
    fig.show()
    pmv.io.save_and_compress_svg(fig, f"py-pkg-treemap-{package.replace('_', '-')}")


# %% Compare multiple packages
fig = pmv.py_pkg_treemap(
    packages := ("pymatviz", "numpy", "pymatgen"),
    show_counts="value+percent",
    min_lines=50,  # Only include files with at least 50 lines
)
# Add customizations: rounded corners and custom hover
fig.update_traces(
    marker=dict(cornerradius=5),
    hovertemplate="<b>%{label}</b><br>Lines: %{value}<br>Percentage: "
    "%{percentRoot:.1%} of total<extra></extra>",
)
title = f"Comparing Package Structure: {', '.join(packages)}"
fig.layout.title.update(text=title, x=0.5, font_size=20)
fig.show()
pmv.io.save_and_compress_svg(fig, "py-pkg-treemap-multiple")


# %% Different ways to group packages
for group_by, clr_scheme in {
    "file": px.colors.qualitative.Set2,
    "directory": px.colors.qualitative.Pastel1,
    "module": px.colors.qualitative.Set3,
}.items():
    fig = pmv.py_pkg_treemap(
        "pymatviz",
        group_by=group_by,  # type: ignore[arg-type]
        show_counts="value",
        min_lines=20,
        color_discrete_sequence=clr_scheme,
    )
    title = f"pymatviz Package Structure - Grouped by {group_by}"
    fig.layout.title.update(text=title, x=0.5, y=0.97, font_size=18)
    fig.show()


# %% Custom formatting of package names and line counts
def custom_module_formatter(module: str, count: int, _total: int) -> str:
    """Custom formatter that emphasizes the module name."""
    return f"{module.upper()} [{count:,} lines]"


fig = pmv.py_pkg_treemap(
    ("numpy", "pymatviz"),
    show_module_counts=custom_module_formatter,
    color_discrete_sequence=px.colors.qualitative.Bold,
)
fig.update_layout(
    title=dict(text="Custom Package Name Formatting", x=0.5, y=0.97, font_size=18),
)
fig.show()
