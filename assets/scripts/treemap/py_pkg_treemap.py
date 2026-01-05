"""Demonstrate treemap visualizations of Python package structures.

This script shows how to create interactive treemaps visualizing module
structures of Python packages using the pmv.py_pkg_treemap function.
"""

# %%
from __future__ import annotations

import numpy as np
import plotly.express as px
import pymatgen

import pymatviz as pmv


pmv.set_plotly_template("plotly_white")


# %% Single packages with default settings
for pkg in ("pymatviz", np):  # Mix strings and module objects
    fig = pmv.py_pkg_treemap(pkg)
    pkg_name = str(getattr(pkg, "__name__", pkg))
    fig.layout.title.update(text=f"{pkg_name} Package Structure", font_size=20, x=0.5)
    fig.show()
    pmv.io.save_and_compress_svg(fig, f"py-pkg-treemap-{pkg_name.replace('_', '-')}")


# %% Compare multiple packages (with file/module filtering via cell_size_fn)
fig = pmv.py_pkg_treemap(
    packages := ("pymatviz", np),
    show_counts="value+percent",
    # Only include files with at least 50 lines
    cell_size_fn=lambda cell: cell.line_count if cell.line_count >= 50 else 0,
)
# Add customizations: rounded corners and custom hover
fig.update_traces(
    marker=dict(cornerradius=5),
    hovertemplate="<b>%{label}</b><br>Lines: %{value}<br>Percentage: "
    "%{percentRoot:.1%} of total<extra></extra>",
)
pkg_names = [getattr(pkg, "__name__", pkg) for pkg in packages]
title = f"Comparing Package Structure: {', '.join(map(str, pkg_names))}"
fig.layout.title.update(text=title, x=0.5, font_size=20)
fig.show()
pmv.io.save_and_compress_svg(fig, "py-pkg-treemap-multiple")


# %% Different ways to group packages
for group_by, clr_scheme in (
    ("file", px.colors.qualitative.Set2),
    ("directory", px.colors.qualitative.Pastel1),
    ("module", px.colors.qualitative.Set3),
):
    fig = pmv.py_pkg_treemap(
        "pymatviz",
        group_by=group_by,
        show_counts="value",
        color_discrete_sequence=clr_scheme,
        # Only include files with at least 20 lines
        cell_size_fn=lambda cell: cell.line_count if cell.line_count >= 20 else 0,
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
    cell_text_fn=custom_module_formatter,
    color_discrete_sequence=px.colors.qualitative.Bold,
)
fig.update_layout(
    title=dict(text="Custom Package Name Formatting", x=0.5, y=0.97, font_size=18),
)
fig.show()


# %% Custom cell sizing by functions, classes, and methods
fig_custom_size = pmv.py_pkg_treemap(
    "pymatviz",
    cell_size_fn=lambda cell: cell.n_functions + cell.n_classes + cell.n_methods,
    show_counts="value",  # Show the custom calculated value
)
title = "pymatviz: Cell size by (functions + classes + methods)"
fig_custom_size.layout.title.update(text=title, x=0.5, y=0.97, font_size=18)
fig_custom_size.update_traces(
    hovertemplate="<b>%{label}</b><br>"  # Use label from path
    "Classes: %{customdata[4]:,}<br>"  # n_classes is at customdata[4]
    "Functions: %{customdata[5]:,}<br>"  # n_functions is at customdata[5]
    "Methods: %{customdata[9]:,}<br>"  # n_methods is at customdata[9]
    "Lines: %{customdata[11]:,}<br>"  # line_count is at customdata[11]
    "Cell Value (Func+Class+Meth): %{value:,}<br>"  # value is the result of calculator
    "<extra></extra>",
)
fig_custom_size.show()


# %% pymatviz treemap with coverage heatmap (cell size by lines, color by test coverage)
# coverage_data_file=f"{pmv.ROOT}/tmp/2025-07-31-pymatviz-coverage.json"
coverage_data_file = "https://github.com/user-attachments/files/21545088/2025-07-31-pymatviz-coverage.json"

fig_cov = pmv.py_pkg_treemap(
    "pymatviz",
    color_by="coverage",
    coverage_data_file=coverage_data_file,
    cell_size_fn=lambda cell: cell.line_count if cell.line_count >= 20 else 0,
    show_counts="value",
    color_continuous_scale="RdYlGn",  # Red-Yellow-Green scale for coverage
)
title = "pymatviz: Coverage Heatmap (Cell size by lines, Color by test coverage)"
fig_cov.layout.title.update(text=title, x=0.5, y=0.97, font_size=18)
fig_cov.show()
pmv.io.save_and_compress_svg(fig_cov, "py-pkg-treemap-pymatviz-coverage")


# %% pymatgen treemap with coverage heatmap and manual color range (0-100%)
coverage_data_file = "https://github.com/user-attachments/files/21545087/2025-07-31-pymatgen-coverage.json"
# coverage_data_file = f"{pmv.ROOT}/tmp/2025-07-31-pymatgen-coverage.json"

fig_cov_range = pmv.py_pkg_treemap(
    pymatgen,
    color_by="coverage",
    coverage_data_file=coverage_data_file,  # Use existing coverage data
    color_range=(0, 100),  # Manual range: 0% (red) to 100% (green)
    cell_size_fn=lambda cell: cell.line_count if cell.line_count >= 20 else 0,
    show_counts="value",
    color_continuous_scale="RdYlGn",  # Red-Yellow-Green scale for coverage
)
# title_range = "pymatgen: Coverage Heatmap with Manual Range (0-100%)"
# fig_cov_range.layout.title.update(text=title_range, x=0.5, y=0.97, font_size=18)
fig_cov_range.layout.margin = dict(l=0, r=0, b=0, t=0)
fig_cov_range.show()
pmv.io.save_and_compress_svg(fig_cov_range, "py-pkg-treemap-pymatgen-coverage")
