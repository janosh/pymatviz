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
module_dir = os.path.dirname(__file__)
structure_dir = f"{TEST_FILES}/structures"
structures = [
    Structure.from_file(file_name) for file_name in glob(f"{structure_dir}/*.json.gz")
]


# %% Fast Chem Env Treemap Plot - CrystalNN
# Using CrystalNN for faster analysis
# Now uses pymatgen's LocalStructOrderParams for proper order parameter calculation
# (Zimmerman et al. 2017, DOI: 10.3389/fmats.2017.00034)
fig_fast = pmv.chem_env_treemap(structures[2], chem_env_settings="crystal_nn")
fig_fast.layout.title = (
    f"Coordination Numbers and Environments (Fast) - {structures[2].formula}"
)
fig_fast.show()
# pmv.io.save_and_compress_svg(fig_basic, "chem-env-treemap-basic")


# %% Time CrystalNN vs ChemEnv Analysis
# Load the structure we'll analyze in detail
structures = [
    Structure.from_file(file_name) for file_name in glob(f"{structure_dir}/*.json.gz")
]
test_structure = structures[2]

# Time ChemEnv
start_time = time.time()
fig_chem_env = pmv.chem_env_treemap(structures[2], chem_env_settings="chemenv")
chem_env_time = time.time() - start_time

# Time CrystalNN
start_time = time.time()
fig_crystal_nn = pmv.chem_env_treemap(structures[2], chem_env_settings="crystal_nn")
crystal_nn_time = time.time() - start_time

print("\nPerformance comparison:")
print(f"ChemEnv time: {chem_env_time:.3f} seconds")
print(f"CrystalNN time: {crystal_nn_time:.3f} seconds")
print(f"Speed improvement: {chem_env_time / crystal_nn_time:.1f}x faster")

fig_chem_env = pmv.chem_env_treemap(test_structure, chem_env_settings="chemenv")
fig_crystal_nn = pmv.chem_env_treemap(test_structure, chem_env_settings="crystal_nn")

fig_chem_env.layout.title = f"ChemEnv Analysis - {test_structure.formula}"
fig_chem_env.show()

fig_crystal_nn.layout.title = f"CrystalNN Analysis - {test_structure.formula}"
fig_crystal_nn.show()


# %% Normalized Chem Env Treemap Plot (per-structure normalization)
# Show relative proportions within each structure
fig_normalized = pmv.chem_env_treemap(
    structures[:3],
    normalize=True,
    chem_env_settings="crystal_nn",  # Use fast analysis for demonstration
)
fig_normalized.layout.title = "Normalized Chem Env Distribution (CrystalNN)"
fig_normalized.show()


# %% Limited CN and CE Treemap Plot with Fast Analysis
# Limit to top 3 CNs and top 2 CEs per CN, using fast CrystalNN
fig_limited = pmv.chem_env_treemap(
    structures[:4], max_cells_cn=3, max_cells_ce=2, chem_env_settings="crystal_nn"
)
fig_limited.layout.title = (
    "Limited Chem Env Treemap (Top 3 CNs, Top 2 CEs per CN) - CrystalNN"
)
fig_limited.show()


# %% Combined Limiting and Normalization with Fast Analysis
# Show normalized data with limits applied, using CrystalNN
fig_combined = pmv.chem_env_treemap(
    structures[:5],
    max_cells_cn=4,
    max_cells_ce=3,
    normalize=True,
    chem_env_settings="crystal_nn",
)
fig_combined.layout.title = (
    "Combined: Limited + Normalized Chem Env Treemap (CrystalNN)"
)
fig_combined.show()


# %% Method Comparison: ChemEnv vs CrystalNN
# Compare results from both methods side by side

# Create comparison for a single structure
test_structure = structures[1]

# Get data from both methods
fig_chem_env = pmv.chem_env_treemap(test_structure, chem_env_settings="chemenv")
fig_crystal_nn = pmv.chem_env_treemap(test_structure, chem_env_settings="crystal_nn")

# Display both results
print(f"\nComparison for {test_structure.formula}:")
print("ChemEnv (slower):")
fig_chem_env.layout.title = f"ChemEnv Analysis - {test_structure.formula}"
fig_chem_env.show()

print("CrystalNN (faster):")
fig_crystal_nn.layout.title = f"CrystalNN Analysis - {test_structure.formula}"
fig_crystal_nn.show()


# %% Large Dataset Analysis with Fast Method
# For analyzing many structures, use the fast CrystalNN method
if len(structures) > 5:
    print(f"\nAnalyzing {len(structures)} structures with CrystalNN (fast method)...")

    start_time = time.time()
    fig_large = pmv.chem_env_treemap(
        structures, chem_env_settings="crystal_nn", max_cells_cn=5, max_cells_ce=3
    )
    total_time = time.time() - start_time

    fig_large.layout.title = (
        f"Large Dataset Analysis: {len(structures)} Structures (CrystalNN)"
    )
    print(f"Analysis completed in {total_time:.2f} seconds")
    fig_large.show()


# %% Custom CN Formatter with Fast Analysis
# Custom formatter for coordination number labels, using fast CrystalNN
# The formatter now displays cleaner decimal formatting (e.g., 4.33 instead of 4.333333)
def custom_cn_formatter(coord_num: int | str, count: int, total: int) -> str:
    """Custom formatter showing detailed statistics."""
    return f"CN-{coord_num}: {count} sites ({count / total:.1%} of total)"


fig_custom = pmv.chem_env_treemap(
    structures[:3], cn_formatter=custom_cn_formatter, chem_env_settings="crystal_nn"
)
fig_custom.layout.title = "Custom CN Formatter with CrystalNN"
fig_custom.show()


# %% Clean Labels (no CN counts) with Fast Analysis
# Hide coordination number counts in labels
fig_clean = pmv.chem_env_treemap(
    structures[0], cn_formatter=False, chem_env_settings="crystal_nn"
)
fig_clean.layout.title = "Clean Labels (No Counts) - CrystalNN"
fig_clean.show()


# %% Normalized Chem Env Treemap Plot: CE counts within each structure to sum to 1
fig_normalized = pmv.chem_env_treemap(structures, normalize=True)
title = "<b>Chem Env Treemap (Normalized per Structure)</b>"
fig_normalized.layout.title = dict(text=title, x=0.5, y=0.95, font_size=18)
fig_normalized.show()
# pmv.io.save_and_compress_svg(fig_normalized, "chem-env-treemap-normalized")


# %% Limit CN cells (max_cells_cn): only top 2 CNs, combine the rest into "Other CNs"
max_cn = 2
fig_limit_cn = pmv.chem_env_treemap(structures, max_cells_cn=max_cn)
title = f"<b>Chem Env Treemap (Top {max_cn} CNs - Other Mode)</b>"
fig_limit_cn.layout.title = dict(text=title, x=0.5, y=0.95, font_size=18)
fig_limit_cn.show()
# pmv.io.save_and_compress_svg(fig_limit_cn, "chem-env-treemap-limit-cn")


# %% Limit CE cells (max_cells_ce)
# For each CN, show only the top 3 CEs, combine the rest into "Other CEs"
max_ce = 3
fig_limit_ce = pmv.chem_env_treemap(structures, max_cells_ce=max_ce)
title = f"<b>Chem Env Treemap (Top {max_ce} CEs per CN - Other Mode)</b>"
fig_limit_ce.layout.title = dict(text=title, x=0.5, y=0.95, font_size=18)
fig_limit_ce.show()
# pmv.io.save_and_compress_svg(fig_limit_ce, "chem-env-treemap-limit-ce")


# %% Combined CN and CE cell limiting with normalization
max_cn_comb = 2
max_ce_comb = 2
fig_combined_limit = pmv.chem_env_treemap(
    structures,
    normalize=True,
    max_cells_cn=max_cn_comb,
    max_cells_ce=max_ce_comb,
)
title = (
    f"<b>Normalized Chem Env Treemap (Top {max_cn_comb} CNs, "
    f"Top {max_ce_comb} CEs - Other Mode)</b>"
)
fig_combined_limit.layout.title = dict(text=title, x=0.5, y=0.95, font_size=18)
fig_combined_limit.show()
# pmv.io.save_and_compress_svg(fig_combined_limit, "chem-env-treemap-combined-limit")


# %% Treemap with custom CN formatter and styling
fig_custom_formatter = pmv.chem_env_treemap(
    structures, cn_formatter=custom_cn_formatter, show_counts="value+percent"
)
# Add styling: rounded corners and custom colors
fig_custom_formatter.update_traces(marker=dict(cornerradius=8))
title = "<b>Chem Env Treemap with Custom CN Formatter and Rounded Corners</b>"
fig_custom_formatter.layout.title = dict(text=title, x=0.5, y=0.95, font_size=18)
fig_custom_formatter.show()
# pmv.io.save_and_compress_svg(
#     fig_custom_formatter, "chem-env-treemap-custom-formatter"
# )


# %% Treemap with disabled CN counts (clean labels)
fig_clean_labels = pmv.chem_env_treemap(
    structures, cn_formatter=False, show_counts="percent"
)
# Add styling: patterns for different sections
fig_clean_labels.update_traces(
    marker=dict(cornerradius=5),
    textinfo="label+percent entry",
    root_color="lightgrey",
)
title = "<b>Chem Env Treemap with Clean Labels (Percentages Only)</b>"
fig_clean_labels.layout.title = dict(text=title, x=0.5, y=0.95, font_size=18)
fig_clean_labels.show()
# pmv.io.save_and_compress_svg(fig_clean_labels, "chem-env-treemap-clean-labels")


# %% Chem Env treemap diagram for elemental carbon from Materials Project
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
        raise SystemExit(0) from None

    with MPRester(use_document_model=False) as mpr:
        # Query for elemental carbon structures
        docs = mpr.materials.summary.search(
            elements=["C"],  # Only carbon
            num_elements=[1, 1],  # Elemental (unary) systems
            fields=[Key.mat_id, Key.structure, Key.formula_pretty],
        )
    with gzip.open(json_path, mode="wt") as file:
        json.dump(
            docs, file, default=lambda x: x.as_dict() if hasattr(x, "as_dict") else x
        )

fig_carbon = pmv.chem_env_treemap(
    [doc["structure"] for doc in docs],
    show_counts="value+percent",
    max_cells_cn=5,  # Limit to top 5 CNs for clarity
    max_cells_ce=4,  # Limit to top 4 CEs per CN for clarity
)

# Add comprehensive styling
fig_carbon.update_traces(
    marker=dict(
        cornerradius=6,
        line=dict(color="white", width=2),  # Add white borders
    ),
    textinfo="label+value+percent entry",
    hovertemplate=(
        "<b>%{label}</b><br>Count: %{value}<br>"
        "%{percentEntry:.1%} of parent<extra></extra>"
    ),
)

title = (
    "<b>Chem Env Treemap: Elemental Carbon (Materials Project)</b><br>"
    "<sub>First level: Coordination Numbers, "
    "Second level: Coordination Environments</sub>"
)
fig_carbon.layout.title = dict(text=title, x=0.5, y=0.95, font_size=16)
fig_carbon.layout.update(height=600, width=800)
fig_carbon.show()
# pmv.io.save_and_compress_svg(fig_carbon, "chem-env-treemap-mp-carbon")


# %% Advanced customization: Custom color mapping and patterns
# Create a treemap to customize
fig_advanced = pmv.chem_env_treemap(
    structures[:3],  # Use subset for clearer example
    show_counts="value",
    max_cells_cn=3,
    max_cells_ce=3,
)

# Apply advanced customizations
fig_advanced.update_traces(
    marker=dict(
        cornerradius=10,
        line=dict(color="darkblue", width=1),
    ),
    textfont=dict(size=12, color="white"),
    textinfo="label+value",
    maxdepth=2,  # Limit treemap depth for better visibility
)

title = (
    "<b>Advanced Chem Env Treemap with Custom Styling</b><br>"
    "<sub>Custom colors, borders, and text formatting</sub>"
)
fig_advanced.layout.title = dict(text=title, x=0.5, y=0.95, font_size=16)
fig_advanced.layout.update(
    height=600,
    width=800,
    font=dict(size=14),
    plot_bgcolor="rgba(240,240,240,0.8)",
)
fig_advanced.show()
# pmv.io.save_and_compress_svg(fig_advanced, "chem-env-treemap-advanced")


# %% Comparison: Multiple structures side by side
fig_comparison = pmv.chem_env_treemap(
    structures[:2],  # Compare first two structures
    normalize=True,  # Normalize for fair comparison
    show_counts="percent",
    max_cells_cn=4,
    max_cells_ce=3,
)

# Style for comparison
fig_comparison.update_traces(
    marker=dict(cornerradius=5),
    textinfo="label+percent entry",
)

formulas = [struct.formula for struct in structures[:2]]
title = f"<b>Normalized Chem Env Comparison: {' vs '.join(formulas)}</b>"
fig_comparison.layout.title = dict(text=title, x=0.5, y=0.95, font_size=16)
fig_comparison.show()
# pmv.io.save_and_compress_svg(fig_comparison, "chem-env-treemap-comparison")


# %% Interactive features demonstration
fig_interactive = pmv.chem_env_treemap(
    structures,
    show_counts="value+percent",
    max_cells_cn=3,
    max_cells_ce=4,
)

# Enhanced interactivity
fig_interactive.update_traces(
    marker=dict(
        cornerradius=8,
        colorscale="Viridis",  # Use color scale for visual appeal
        line=dict(color="white", width=1.5),
    ),
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Count: %{value}<br>"
        "Percentage of parent: %{percentEntry:.2%}<br>"
        "Percentage of total: %{percentRoot:.2%}<br>"
        "<extra></extra>"
    ),
)

title = (
    "<b>Interactive Chem Env Treemap</b><br>"
    "<sub>Enhanced hover information and visual appeal</sub>"
)
fig_interactive.layout.title = dict(text=title, x=0.5, y=0.95, font_size=16)
fig_interactive.layout.update(height=700, width=900)
fig_interactive.show()
pmv.io.save_and_compress_svg(fig_interactive, "chem-env-treemap-interactive")


# %% Performance demonstration with larger dataset
# This example shows how the treemap handles larger datasets
fig_large = pmv.chem_env_treemap(
    structures,  # Use all available structures
    normalize=False,  # Keep absolute counts for this example
    max_cells_cn=6,  # Allow more CNs to show diversity
    max_cells_ce=5,  # Allow more CEs per CN
    show_counts="value",
)

# Optimize for larger datasets
fig_large.update_traces(
    marker=dict(cornerradius=3),  # Smaller radius for more cells
    textfont=dict(size=10),  # Smaller text
    textinfo="label+value",
)

title = f"<b>Chem Env Treemap: Large Dataset ({len(structures)} structures)</b>"
fig_large.layout.title = dict(text=title, x=0.5, y=0.95, font_size=16)
fig_large.layout.update(height=800, width=1000)
fig_large.show()
pmv.io.save_and_compress_svg(fig_large, "chem-env-treemap-large-dataset")
