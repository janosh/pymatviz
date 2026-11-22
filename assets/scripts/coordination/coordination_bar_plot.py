"""Coordination histogram examples."""

# /// script
# dependencies = [
#     "pymatgen>=2026.9.24",
# ]
# ///

# %%
from glob import glob

from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.coordination import CnSplitMode
from pymatviz.utils.testing import TEST_FILES, formula_spg_str


# %%
structures = {
    formula_spg_str(struct := Structure.from_file(file)): struct
    for file in (
        glob(f"{TEST_FILES}/structures/*.json.gz") + glob(f"{TEST_FILES}/xrd/*.cif")
    )
}
key1, key2, key3, *_ = structures
three_structs = {key: structures[key] for key in (key1, key2, key3)}


# %% Single structure with a 4 Å cutoff
widget = pmv.coordination_bar_plot(structures[key1], strategy=4.0, annotate_bars=True)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/coordination-hist-single.svg",
    width=700,
    transparent=True,
)


# %% CrystalNN and Voronoi neighbors (as pymatgen's CrystalNN and VoronoiNN)
for strategy in ("crystal_nn", "voronoi"):
    pmv.coordination_bar_plot(structures[key1], strategy=strategy).show()


# %% MatterViz bonding in the browser, the default (drop more structure files onto it)
pmv.coordination_bar_plot(three_structs).show()


# %% Multiple structures: one series per element (summed), per structure, or all sites
for split_mode in CnSplitMode:
    widget = pmv.coordination_bar_plot(
        three_structs, strategy=3.0, split_mode=split_mode
    )
    widget.show()
    if split_mode == CnSplitMode.by_element:  # shown in the readme
        widget.to_img(
            f"{pmv.utils.ROOT}/assets/svg/coordination-hist-by-element.svg",
            width=700,
            transparent=True,
        )


# %% Side-by-side bars per structure with custom element colors
pmv.coordination_bar_plot(
    three_structs, strategy=3.0, split_mode="by_structure", mode="grouped"
).show()
pmv.coordination_bar_plot(
    three_structs,
    strategy=3.0,
    element_colors={"Fe": "red", "O": "blue", "H": "green"},
).show()
