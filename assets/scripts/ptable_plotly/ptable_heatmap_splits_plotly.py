# %%
import itertools

import numpy as np
from pymatgen.core.periodic_table import Element

import pymatviz as pmv


np_rng = np.random.default_rng(seed=0)


# %% Examples of ptable_heatmap_splits_plotly with different numbers of splits
for n_splits, orientation in itertools.product(
    range(2, 5),
    ("diagonal", "horizontal", "vertical", "grid"),
):
    if orientation == "grid" and n_splits != 4:
        continue

    data_dict = {
        elem.symbol: np_rng.integers(10 * n_splits, 20 * (n_splits + 1), size=n_splits)
        for elem in Element
    }

    cbar_title = f"Periodic Table Heatmap with {n_splits}-fold split"
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data_dict,
        orientation=orientation,  # type: ignore[arg-type]
        colorscale="RdYlBu",
        colorbar=dict(title=cbar_title),
    )

    fig.show()
    if orientation == "diagonal":
        pmv.io.save_and_compress_svg(fig, f"ptable-heatmap-splits-plotly-{n_splits}")
