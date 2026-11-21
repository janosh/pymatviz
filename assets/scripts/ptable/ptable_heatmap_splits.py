"""Periodic table heatmaps with element tiles split into 2-4 segments."""

# %%
import numpy as np
from pymatgen.core import Element

import pymatviz as pmv


np_rng = np.random.default_rng(seed=0)


# %% 2 (diagonal), 3 (horizontal strips) and 4 (quadrants) values per element
for n_splits in range(2, 5):
    data_dict = {
        elem.symbol: np_rng.integers(10, 20, size=n_splits).tolist() for elem in Element
    }
    widget = pmv.ptable_heatmap_splits(
        data_dict,
        color_scale="interpolateRdYlBu",
        color_bar={"title": f"Periodic table heatmap with {n_splits}-fold split"},
    )
    widget.show()
    if n_splits < 4:  # readme shows the 2- and 3-fold splits
        widget.to_img(
            f"{pmv.ROOT}/assets/png/ptable-heatmap-splits-{n_splits}.png",
            width=1000,
            transparent=True,
        )


# %% Sparse data with NaN segments: absent elements and NaN splits stay uncolored
sparse_data = {
    elem.symbol: [np_rng.normal(), np.nan if elem.Z % 3 == 0 else np_rng.normal()]
    for elem in list(Element)[:56]
}
widget = pmv.ptable_heatmap_splits(
    sparse_data, missing={"color": "#ddd"}, color_bar={"title": "Metric"}
)
widget.show()


# %% 4 values per element as triangles (top, right, bottom, left), 1-decimal labels
widget = pmv.ptable_heatmap_splits(
    {elem.symbol: np_rng.normal(size=4) for elem in list(Element)[:56]},
    split_layout="triangular",
    tile_props={"float_fmt": ".1f", "show_number": False},
    color_bar={"title": "Metric"},
)
widget.show()
