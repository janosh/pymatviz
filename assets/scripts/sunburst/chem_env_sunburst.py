"""Coordination Number (CN) and Coordination Environment (CE) sunburst examples."""

# %%
from __future__ import annotations

import gzip
import json
import os
from glob import glob
from typing import Literal

from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.enums import Key
from pymatviz.utils import ROOT
from pymatviz.utils.testing import TEST_FILES


def linkify_mp_id(mat_id: str, color: str = "#0066cc") -> str:
    """Create a clickable link for Materials Project IDs."""
    if mat_id.startswith("mp-"):
        href = f"https://materialsproject.org/materials/{mat_id}"
        return f'<a href="{href}" target="_blank" style="color:{color}">{mat_id}</a>'
    return mat_id


# %% Load gzipped test structures
module_dir = os.path.dirname(__file__)
structure_dir = f"{TEST_FILES}/structures"
structures = [
    Structure.from_file(file_name) for file_name in glob(f"{structure_dir}/*.json.gz")
]


# %% Basic chem env Sunburst Plot - ChemEnv vs CrystalNN Comparison
# Default behavior: absolute counts, no slice limits
spg_num = structures[2].get_space_group_info()[1]
mat_id = structures[2].properties[Key.mat_id]
mat_id_link = linkify_mp_id(mat_id)

# ChemEnv method (comprehensive but slow)
fig = pmv.chem_env_sunburst(structures[2], chem_env_settings="chemenv")
title_chem_env = (
    f"<b>Chem Env Sunburst of {structures[2].formula} (ChemEnv)</b><br>"
    f"ID: {mat_id_link}, space group: {spg_num}"
)
fig.layout.title = dict(text=title_chem_env, x=0.5, y=0.96, font_size=18)
fig.show()
# pmv.io.save_and_compress_svg(fig, "chem-env-sunburst-basic-chemenv")

# CrystalNN method (fast but less detailed)
fig_basic = pmv.chem_env_sunburst(structures[2], chem_env_settings="crystal_nn")
title = (
    f"<b>Chem Env Sunburst of {structures[2].formula} (CrystalNN)</b><br>"
    f"ID: {mat_id_link}, space group: {spg_num}"
)
fig_basic.layout.title = dict(text=title, x=0.5, y=0.96, font_size=18)
fig_basic.show()
# pmv.io.save_and_compress_svg(fig_basic, "chem-env-sunburst-basic-crystal-nn")

fig_struct = pmv.structure_3d(structures[2], scale=0.5)
struct_title = (
    f"<b>3D Structure: {structures[2].formula}</b><br>"
    f"ID: {mat_id_link}, space group: {spg_num}"
)
fig_struct.layout.title = dict(text=struct_title, x=0.5, y=0.95, font_size=16)
fig_struct.show()


# %% Normalized Chem Env Sunburst Plot - ChemEnv vs CrystalNN Comparison
# CE counts within each structure to sum to 1

# ChemEnv method
fig_normalized_ch_emenv = pmv.chem_env_sunburst(
    structures, normalize=True, chem_env_settings="chemenv"
)
title_chem_env = "<b>Chem Env Sunburst (Normalized per Structure - ChemEnv)</b>"
fig_normalized_ch_emenv.layout.title = dict(
    text=title_chem_env, x=0.5, y=0.96, font_size=18
)
fig_normalized_ch_emenv.show()
# pmv.io.save_and_compress_svg(
#     fig_normalized_ch_emenv, "chem-env-sunburst-normalized-chemenv"
# )

# CrystalNN method
fig = pmv.chem_env_sunburst(structures, normalize=True, chem_env_settings="crystal_nn")
title = "<b>Chem Env Sunburst (Normalized per Structure - CrystalNN)</b>"
fig.layout.title = dict(text=title, x=0.5, y=0.96, font_size=18)
fig.show()
# pmv.io.save_and_compress_svg(
#     fig, "chem-env-sunburst-normalized-crystal-nn"
# )

# Show 3D structures for the first few structures in the dataset as subplots
num_struct_show = min(3, len(structures))
print(f"Showing 3D structures for {num_struct_show} structures in the analysis:")

structures_dict = {}
for idx in range(num_struct_show):
    struct = structures[idx]
    spg = struct.get_space_group_info()[1]
    mat_id = struct.properties.get(Key.mat_id, f"struct-{idx + 1}")
    mat_id_display = linkify_mp_id(mat_id)
    key = f"Structure {idx + 1}: {struct.formula} ({mat_id_display}, SG: {spg})"
    structures_dict[key] = struct

if structures_dict:
    fig_struct_multi = pmv.structure_3d(structures_dict, scale=0.5)
    title = "<b>3D Structures in Chem Env Analysis</b>"
    fig_struct_multi.layout.title = dict(text=title, x=0.5, y=0.95, font_size=16)
    fig_struct_multi.show()


# %% Limit CN slices - ChemEnv vs CrystalNN Comparison
# Only top 2 CNs, combine the rest into "Other CNs"
max_cn = 2
max_slices_mode: Literal["other", "drop"]
for max_slices_mode in ("other", "drop"):
    # ChemEnv method
    fig_limit_cn_chem_env = pmv.chem_env_sunburst(
        structures,
        max_slices_cn=max_cn,
        max_slices_mode=max_slices_mode,
        chem_env_settings="chemenv",
    )
    title_chem_env = (
        f"<b>Chem Env Sunburst (Top {max_cn} CNs - "
        f"{max_slices_mode} Mode - ChemEnv)</b>"
    )
    fig_limit_cn_chem_env.layout.title = dict(
        text=title_chem_env, x=0.5, y=0.96, font_size=18
    )
    fig_limit_cn_chem_env.show()
    svg_path_chem_env = f"chem-env-sunburst-limit-cn-{max_slices_mode}-chemenv.svg"
    # pmv.io.save_and_compress_svg(fig_limit_cn_chem_env, svg_path_chem_env)

    # CrystalNN method
    fig = pmv.chem_env_sunburst(
        structures,
        max_slices_cn=max_cn,
        max_slices_mode=max_slices_mode,
        chem_env_settings="crystal_nn",
    )
    title = (
        f"<b>Chem Env Sunburst (Top {max_cn} CNs - "
        f"{max_slices_mode} Mode - CrystalNN)</b>"
    )
    fig.layout.title = dict(text=title, x=0.5, y=0.96, font_size=18)
    fig.show()
    svg_path = f"chem-env-sunburst-limit-cn-{max_slices_mode}-crystal-nn.svg"
    # pmv.io.save_and_compress_svg(fig, svg_path)


# %% Limit CE slices - ChemEnv vs CrystalNN Comparison
# For each CN, show only the top 3 CEs, combine the rest into "Other CEs"
max_ce = 3
for max_slices_mode in ("other", "drop"):
    # ChemEnv method
    fig_limit_ce_chem_env = pmv.chem_env_sunburst(
        structures,
        max_slices_ce=max_ce,
        max_slices_mode=max_slices_mode,
        chem_env_settings="chemenv",
    )
    title_chem_env = (
        f"<b>Chem Env Sunburst (Top {max_ce} CEs per CN - "
        f"{max_slices_mode} Mode - ChemEnv)</b>"
    )
    fig_limit_ce_chem_env.layout.title = dict(
        text=title_chem_env, x=0.5, y=0.96, font_size=18
    )
    fig_limit_ce_chem_env.show()
    svg_path_chem_env = f"chem-env-sunburst-limit-ce-{max_slices_mode}-chemenv.svg"
    # pmv.io.save_and_compress_svg(fig_limit_ce_chem_env, svg_path_chem_env)

    # CrystalNN method
    fig = pmv.chem_env_sunburst(
        structures,
        max_slices_ce=max_ce,
        max_slices_mode=max_slices_mode,
        chem_env_settings="crystal_nn",
    )
    title = (
        f"<b>Chem Env Sunburst (Top {max_ce} CEs per CN - "
        f"{max_slices_mode} Mode - CrystalNN)</b>"
    )
    fig.layout.title = dict(text=title, x=0.5, y=0.96, font_size=18)
    fig.show()
    svg_path = f"chem-env-sunburst-limit-ce-{max_slices_mode}-crystal-nn.svg"
    # pmv.io.save_and_compress_svg(fig, svg_path)


# %% Combined CN and CE slicing with normalization - ChemEnv vs CrystalNN Comparison
max_cn_comb = max_ce_comb = 2

# ChemEnv method
fig_combined_limi_t_chem_env = pmv.chem_env_sunburst(
    structures,
    normalize=True,
    max_slices_cn=max_cn_comb,
    max_slices_ce=max_ce_comb,
    max_slices_mode="other",
    chem_env_settings="chemenv",
)
title_chem_env = (
    f"<b>Normalized Chem Env (Top {max_cn_comb} CNs, "
    f"Top {max_ce_comb} CEs - Other Mode - ChemEnv)</b>"
)
fig_combined_limi_t_chem_env.layout.title = dict(
    text=title_chem_env, x=0.5, y=0.96, font_size=18
)
fig_combined_limi_t_chem_env.show()
# pmv.io.save_and_compress_svg(
#     fig_combined_limi_t_chem_env, "chem-env-sunburst-combined-limit-chemenv"
# )

# CrystalNN method
fig = pmv.chem_env_sunburst(
    structures,
    normalize=True,
    max_slices_cn=max_cn_comb,
    max_slices_ce=max_ce_comb,
    max_slices_mode="other",
    chem_env_settings="crystal_nn",
)
title = (
    f"<b>Normalized Chem Env (Top {max_cn_comb} CNs, "
    f"Top {max_ce_comb} CEs - Other Mode - CrystalNN)</b>"
)
fig.layout.title = dict(text=title, x=0.5, y=0.96, font_size=18)
fig.show()
# pmv.io.save_and_compress_svg(fig, "chem-env-sunburst-combined-limit-crystal-nn")


# %% Chem Env sunburst diagram for elemental carbon from Materials Project.
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

# ChemEnv method for elemental carbon
fig_chem_env = pmv.chem_env_sunburst(
    [doc["structure"] for doc in docs],
    show_counts="value+percent",
    chem_env_settings="chemenv",
)

title_chem_env = (
    "<b>Chem Env Sunburst: Elemental Carbon (ChemEnv)</b><br>"
    "<sub>Inner ring: Coordination Numbers, "
    "Outer ring: Coordination Environments</sub>"
)
fig_chem_env.layout.title = dict(text=title_chem_env, x=0.5, y=0.96, font_size=16)
fig_chem_env.layout.update(height=600, width=600)
fig_chem_env.show()
pmv.io.save_and_compress_svg(fig_chem_env, "chem-env-sunburst-mp-carbon-chemenv")

# CrystalNN method for elemental carbon
fig = pmv.chem_env_sunburst(
    [doc["structure"] for doc in docs],
    show_counts="value+percent",
    chem_env_settings="crystal_nn",
)

title = (
    "<b>Chem Env Sunburst: Elemental Carbon (CrystalNN)</b><br>"
    "<sub>Inner ring: Coordination Numbers, "
    "Outer ring: Coordination Environments</sub>"
)
fig.layout.title = dict(text=title, x=0.5, y=0.96, font_size=16)
fig.layout.update(height=600, width=600)
fig.show()
pmv.io.save_and_compress_svg(fig, "chem-env-sunburst-mp-carbon-crystal-nn")

# Show 3D structures for a few representative carbon polymorphs as subplots
num_carbon_show = min(3, len(docs))
print(f"Showing 3D structures for {num_carbon_show} representative carbon structures:")

carbon_structures_dict = {}
for idx in range(num_carbon_show):
    doc = docs[idx]
    struct = doc["structure"]
    spg = struct.get_space_group_info()[1]
    mat_id = doc.get("material_id", f"carbon-{idx + 1}")
    mat_id_display = linkify_mp_id(mat_id)
    key = f"{mat_id_display}: {struct.formula} (SG: {spg})"
    carbon_structures_dict[key] = struct

if carbon_structures_dict:
    fig = pmv.structure_3d(carbon_structures_dict, scale=0.5)
    fig.layout.title = dict(
        text="<b>3D Carbon Polymorphs from Materials Project</b>",
        x=0.5,
        y=0.95,
        font_size=16,
    )
    fig.show()
