# %%
import itertools
from collections.abc import Callable, Sequence

import numpy as np
from pymatgen.core import Element

import pymatviz as pmv
import pymatviz.colors as pmv_colors
from pymatviz.typing import RgbColorType


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


# %% Visualize multiple element color schemes on a split periodic table heatmap
def make_color_scale(
    color_schemes: Sequence[dict[str, RgbColorType]],
) -> Callable[[str, float, int], str]:
    """Return element colors in different palettes based on split index."""

    def elem_color_scale(element: str, _val: float, split_idx: int) -> str:
        color = color_schemes[split_idx].get(element)
        if color is None:
            raise ValueError(f"no color for {element=} in {split_idx=}")
        return f"rgb{color}"

    return elem_color_scale


palettes_3 = (
    pmv_colors.ELEM_COLORS_ALLOY,
    pmv_colors.ELEM_COLORS_JMOL,
    pmv_colors.ELEM_COLORS_VESTA,
)

fig = pmv.ptable_heatmap_splits_plotly(
    # Use dummy values for all elements
    {str(elem): list(range(len(palettes_3))) for elem in Element},
    orientation="diagonal",  # could also use "grid"
    colorscale=make_color_scale(palettes_3),
    hover_data=dict.fromkeys(
        map(str, Element), "top left: JMOL<br>top right: VESTA, bottom: ALLOY"
    ),
)
title = (
    "<b>Element color schemes</b><br>top left: JMOL, top right: VESTA, bottom: ALLOY"
)
fig.layout.title.update(text=title, x=0.4, y=0.8)
fig.show()
pmv.io.save_and_compress_svg(fig, "ptable-heatmap-splits-plotly-3-color-schemes")


# %% Visualize multiple element color schemes on a split periodic table heatmap
palettes_2 = (pmv_colors.ELEM_COLORS_ALLOY, pmv_colors.ELEM_COLORS_VESTA)

fig = pmv.ptable_heatmap_splits_plotly(
    # Use dummy values for all elements
    {str(elem): list(range(len(palettes_2))) for elem in Element},
    orientation="vertical",
    colorscale=make_color_scale(palettes_2),
    hover_data=dict.fromkeys(map(str, Element), "left: VESTA<br>right: ALLOY"),
)
title = "<b>Element color schemes</b><br>left: VESTA, right: ALLOY"
fig.layout.title.update(text=title, x=0.4, y=0.8)
fig.show()
pmv.io.save_and_compress_svg(fig, "ptable-heatmap-splits-plotly-2-color-schemes")
