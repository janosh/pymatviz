# %%
import numpy as np
from pymatgen.core.periodic_table import Element

import pymatviz as pmv


# %% Evenly-split tile plots laid out as a periodic table
rng = np.random.default_rng(seed=0)
for n_splits in (2, 3, 4):
    data_dict = {
        elem.symbol: rng.integers(10 * n_splits, 20 * (n_splits + 1), size=n_splits)
        for elem in Element
    }

    fig = pmv.ptable_heatmap_splits(
        data=data_dict,
        colormap="coolwarm",
        start_angle=135 if n_splits % 2 == 0 else 90,
        cbar_title="Periodic Table Evenly-Split Heatmap Plots",
        hide_f_block=True,
    )
    fig.show()
    pmv.io.save_and_compress_svg(fig, f"ptable-heatmap-splits-{n_splits}")
