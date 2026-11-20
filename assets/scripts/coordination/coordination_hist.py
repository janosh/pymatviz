"""Coordination histogram examples."""

# %%
from glob import glob

from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.coordination import CnSplitMode
from pymatviz.utils.testing import TEST_FILES, formula_spg_str
from pymatviz.widgets import CoordinationBarPlotWidget


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
widget = pmv.coordination_hist(structures[key1], strategy=4.0, annotate_bars=True)
widget.show()
widget.to_img(f"{pmv.ROOT}/assets/svg/coordination-hist-single.svg", width=700)


# %% pymatgen NearNeighbors strategies
for strategy in (CrystalNN(), VoronoiNN()):
    pmv.coordination_hist(structures[key1], strategy=strategy).show()


# %% MatterViz bonding computed in the frontend (drop more structure files onto it)
CoordinationBarPlotWidget(three_structs, strategy="electroneg_ratio").show()


# %% Multiple structures: one series per element (summed), per structure, or all sites
for split_mode in CnSplitMode:
    widget = pmv.coordination_hist(three_structs, split_mode=split_mode)
    widget.show()
    svg_name = f"coordination-hist-{split_mode.value.replace('_', '-')}"
    widget.to_img(f"{pmv.ROOT}/assets/svg/{svg_name}.svg", width=700)


# %% Side-by-side bars per structure with custom element colors
pmv.coordination_hist(three_structs, split_mode="by_structure", mode="grouped").show()
pmv.coordination_hist(
    three_structs, element_color_scheme={"Fe": "red", "O": "blue", "H": "green"}
).show()
