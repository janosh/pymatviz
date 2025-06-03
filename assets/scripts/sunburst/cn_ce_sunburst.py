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
from pymatviz.utils.testing import TEST_FILES


# %% Load gzipped test structures
module_dir = os.path.dirname(__file__)
structure_dir = f"{TEST_FILES}/structures"
structures = [
    Structure.from_file(file_name) for file_name in glob(f"{structure_dir}/*.json.gz")
]


# %% Basic CN-CE Sunburst Plot
# Default behavior: absolute counts, no slice limits
fig_basic = pmv.cn_ce_sunburst(structures[2])
spg_num = structures[2].get_space_group_info()[1]
title = (
    f"<b>CN-CE Sunburst of {structures[2].formula}</b><br>"
    f"ID: {structures[2].properties[Key.mat_id]}, space group: {spg_num}"
)
fig_basic.layout.title = dict(text=title, x=0.5, y=0.96, font_size=18)
fig_basic.show()
# pmv.io.save_and_compress_svg(fig_basic, "cn-ce-sunburst-basic")


# %% Normalized CN-CE Sunburst Plot: CE counts within each structure to sum to 1
fig_normalized = pmv.cn_ce_sunburst(structures, normalize=True)
title = "<b>CN-CE Sunburst (Normalized per Structure)</b>"
fig_normalized.layout.title = dict(text=title, x=0.5, y=0.96, font_size=18)
fig_normalized.show()
# pmv.io.save_and_compress_svg(fig_normalized, "cn-ce-sunburst-normalized")


# %% Limit CN slices (max_slices_cn): only top 2 CNs, combine the rest into "Other CNs"
max_cn = 2
max_slices_mode: Literal["other", "drop"]
for max_slices_mode in ("other", "drop"):
    fig_limit_cn = pmv.cn_ce_sunburst(
        structures, max_slices_cn=max_cn, max_slices_mode=max_slices_mode
    )
    title = f"<b>CN-CE Sunburst (Top {max_cn} CNs - {max_slices_mode} Mode)</b>"
    fig_limit_cn.layout.title = dict(text=title, x=0.5, y=0.96, font_size=18)
    fig_limit_cn.show()
    svg_path = f"cn-ce-sunburst-limit-cn-{max_slices_mode}.svg"
    # pmv.io.save_and_compress_svg(fig_limit_cn, svg_path)


# %% Limit CE slices (max_slices_ce)
# For each CN, show only the top 3 CEs, combine the rest into "Other CEs"
max_ce = 3
for max_slices_mode in ("other", "drop"):
    fig_limit_ce = pmv.cn_ce_sunburst(
        structures, max_slices_ce=max_ce, max_slices_mode=max_slices_mode
    )
    title = f"<b>CN-CE Sunburst (Top {max_ce} CEs per CN - {max_slices_mode} Mode)</b>"
    fig_limit_ce.layout.title = dict(text=title, x=0.5, y=0.96, font_size=18)
    fig_limit_ce.show()
    svg_path = f"cn-ce-sunburst-limit-ce-{max_slices_mode}.svg"
    # pmv.io.save_and_compress_svg(fig_limit_ce, svg_path)


# %% Combined CN and CE slicing with normalization
max_cn_comb = 2
max_ce_comb = 2
fig_combined_limit = pmv.cn_ce_sunburst(
    structures,
    normalize=True,
    max_slices_cn=max_cn_comb,
    max_slices_ce=max_ce_comb,
    max_slices_mode="other",
)
title = (
    f"<b>Normalized CN-CE (Top {max_cn_comb} CNs, "
    f"Top {max_ce_comb} CEs - Other Mode)</b>"
)
fig_combined_limit.layout.title = dict(text=title, x=0.5, y=0.96, font_size=18)
fig_combined_limit.show()
# pmv.io.save_and_compress_svg(fig_combined_limit, "cn-ce-sunburst-combined-limit")


# %% CN-CE sunburst diagram for elemental carbon from Materials Project.
json_path = f"{module_dir}/mp-carbon-structures.json.gz"

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

fig = pmv.cn_ce_sunburst(
    [doc["structure"] for doc in docs], show_counts="value+percent"
)

title = (
    "<b>CN-CE Sunburst: Elemental Carbon</b><br>"
    "<sub>Inner ring: Coordination Numbers, Outer ring: Coordination Environments</sub>"
)
fig.layout.title = dict(text=title, x=0.5, y=0.96, font_size=16)
fig.layout.update(height=600, width=600)
fig.show()
pmv.io.save_and_compress_svg(fig, "cn-ce-sunburst-mp-carbon")
