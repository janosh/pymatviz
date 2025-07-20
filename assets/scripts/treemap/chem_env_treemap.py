"""Coordination Number (CN) and Coordination Environment (CE) treemap examples."""

# %%
from __future__ import annotations

import gzip
import json
import os
import time
from glob import glob

from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.utils import ROOT
from pymatviz.utils.testing import TEST_FILES


# %% Load structures
structure_dir = f"{TEST_FILES}/structures"
structures = [
    Structure.from_file(file_name) for file_name in glob(f"{structure_dir}/*.json.gz")
]


# %% Example 1: Method Comparison with Performance Analysis
# Compare CrystalNN (fast) vs ChemEnv (accurate) methods with timing
test_structure = structures[2]

# Time ChemEnv method
start_time = time.perf_counter()
fig_chem_env = pmv.chem_env_treemap(test_structure, chem_env_settings="chemenv")
chem_env_time = time.perf_counter() - start_time

# Time CrystalNN method
start_time = time.perf_counter()
fig_crystal_nn = pmv.chem_env_treemap(test_structure, chem_env_settings="crystal_nn")
crystal_nn_time = time.perf_counter() - start_time

print(f"ChemEnv time: {chem_env_time:.3f} seconds")
print(f"CrystalNN time: {crystal_nn_time:.3f} seconds")
print(f"Speed improvement: {chem_env_time / crystal_nn_time:.1f}x faster")

fig_chem_env.layout.title = f"ChemEnv Analysis (Accurate) - {test_structure.formula}"
fig_crystal_nn.layout.title = f"CrystalNN Analysis (Fast) - {test_structure.formula}"

fig_chem_env.show()
fig_crystal_nn.show()


# %% Example 2: Advanced Features - Normalization, Limiting, and Custom Formatting
# Demonstrate key functional features in one comprehensive example


def custom_cn_formatter(coord_num: int | str, count: float, total: float) -> str:
    """Custom formatter showing detailed statistics."""
    return f"CN-{coord_num}: {count:.1f} sites ({count / total:.1%})"


fig_advanced = pmv.chem_env_treemap(
    structures[:4],
    normalize=True,  # Normalize counts per structure
    max_cells_cn=3,  # Limit to top 3 coordination numbers
    max_cells_ce=2,  # Limit to top 2 environments per CN
    cn_formatter=custom_cn_formatter,  # Custom label formatting
    show_counts="value+percent",  # Show both values and percentages
    chem_env_settings="crystal_nn",  # Use fast method for demo
)

title = (
    "<b>Advanced Features Demo</b><br>"
    "<sub>Normalized + Limited (Top 3 CNs, Top 2 CEs) + Custom Formatting</sub>"
)
fig_advanced.layout.title = dict(text=title, x=0.5, y=0.95, font_size=16)
fig_advanced.show()


# %% Example 3: Real-world Data - Elemental Carbon from Materials Project
# Demonstrate usage with actual crystallographic data
json_path = f"{ROOT}/tmp/mp-carbon-structures.json.gz"

if os.path.isfile(json_path):
    with gzip.open(json_path, mode="rt") as file:
        docs = json.load(file)
    for doc in docs:
        doc["structure"] = Structure.from_dict(doc["structure"])
else:
    try:
        from mp_api.client import MPRester
    except ImportError:
        # Create dummy data if Materials Project API not available
        docs = [{"structure": structures[0]} for _ in range(3)]
    else:
        with MPRester(use_document_model=False) as mpr:
            docs = mpr.materials.summary.search(
                elements=["C"],
                num_elements=[1, 1],
                fields=[Key.mat_id, Key.structure, Key.formula_pretty],
            )
        with gzip.open(json_path, mode="wt") as file:
            json.dump(
                docs,
                file,
                default=lambda x: x.as_dict() if hasattr(x, "as_dict") else x,
            )

fig_carbon = pmv.chem_env_treemap(
    [doc["structure"] for doc in docs],
    max_cells_cn=5,
    max_cells_ce=4,
    show_counts="value+percent",
    chem_env_settings="crystal_nn",
)

title = (
    "<b>Real-world Example: Elemental Carbon Structures</b><br>"
    "<sub>Coordination environments in carbon allotropes from Materials Project</sub>"
)
fig_carbon.layout.title = dict(text=title, x=0.5, y=0.95, font_size=16)
fig_carbon.layout.update(height=600, width=800)
fig_carbon.show()


# %% Example 4: Styling and Interactivity
# Showcase visual customization and enhanced user interaction
fig_styled = pmv.chem_env_treemap(
    structures,
    max_cells_cn=4,
    max_cells_ce=3,
    show_counts="value",
    chem_env_settings="crystal_nn",
)

# Apply comprehensive styling and interactivity
fig_styled.update_traces(
    marker=dict(
        cornerradius=8,
        colorscale="Viridis",
        line=dict(color="white", width=2),
    ),
    textfont=dict(size=11, color="white"),
    textinfo="label+value",
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Count: %{value}<br>"
        "% of parent: %{percentEntry:.1%}<br>"
        "% of total: %{percentRoot:.1%}<br>"
        "<extra></extra>"
    ),
    maxdepth=2,
)

title = (
    "<b>Styled & Interactive Treemap</b><br>"
    "<sub>Custom colors, rounded corners, enhanced hover info</sub>"
)
fig_styled.layout.title = dict(text=title, x=0.5, y=0.95, font_size=16)
fig_styled.layout.update(
    height=700, width=900, font=dict(size=12), plot_bgcolor="rgba(248,249,250,0.8)"
)
fig_styled.show()
pmv.io.save_and_compress_svg(fig_styled, "chem-env-treemap-styled")
