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
for package in ("pymatviz", "numpy"):
    fig = pmv.py_pkg_treemap(package)
    fig.layout.title.update(text=f"{package} Package Structure", font_size=20, x=0.5)
    fig.show()
    pmv.io.save_and_compress_svg(fig, f"py-pkg-treemap-{package.replace('_', '-')}")


# %% Compare multiple packages
fig = pmv.py_pkg_treemap(
    packages := ("pymatviz", "numpy"),
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


# %% Custom cell sizing based on number of functions + classes
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
