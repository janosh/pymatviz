# %%
import numpy as np
from pymatgen.core.periodic_table import Element

import pymatviz as pmv


# %%
np_rng = np.random.default_rng(seed=0)


# %% Histograms laid out in as a periodic table
# Generate random parity data with y \approx x with some noise
data_dict = {
    elem.symbol: np_rng.standard_normal(100) + np_rng.standard_normal(100)
    for elem in Element
}
fig = pmv.ptable_hists(
    data_dict,
    colormap="coolwarm",
    cbar_title="Periodic Table Histograms",
    cbar_axis="x",
    color_elem_strategy="background",
    add_elem_type_legend=True,
    symbol_pos=(0.25, 0.8),
    anno_text={elem.symbol: str(idx - 1) for idx, elem in enumerate(Element)},
    anno_pos=(0.75, 0.8),
)
pmv.io.save_and_compress_svg(fig, "ptable-hists", transparent_bg=False)
