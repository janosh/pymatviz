"""Periodic table heatmap splits plotly examples."""

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
for idx, (n_splits, orientation) in enumerate(
    itertools.product(range(2, 5), ("diagonal", "horizontal", "vertical", "grid"))
):
    if orientation == "grid" and n_splits != 4:
        continue
    if idx > 5:  # running all n_split/orientation combos takes long
        break

    # Example 1: Single colorscale with single colorbar
    data_dict = {
        elem.symbol: np_rng.integers(10, 20, size=n_splits) for elem in Element
    }
    cbar_title = f"Periodic Table Heatmap with {n_splits}-fold split"
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data_dict,
        orientation=orientation,
        colorscale="RdYlBu",  # Single colorscale will be used for all splits
        colorbar=dict(title=cbar_title),
    )
    fig.show()

    # Example 2: Multiple colorscales with vertical colorbars
    colorscales = ["Viridis", "Plasma", "Inferno", "Magma"][:n_splits]
    colorbars = [
        dict(title=f"Metric {idx + 1}", orientation="v") for idx in range(n_splits)
    ]
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data_dict,
        orientation=orientation,
        colorscale=colorscales,
        colorbar=colorbars,
    )
    fig.show()

    # Example 3: Multiple colorscales with horizontal colorbars
    # Use sequential colors from the same family
    sequential_colors = [
        [(0, "rgb(255,220,220)"), (1, "rgb(255,0,0)")],  # Red scale
        [(0, "rgb(220,220,255)"), (1, "rgb(0,0,255)")],  # Blue scale
        [(0, "rgb(220,255,220)"), (1, "rgb(0,255,0)")],  # Green scale
        [(0, "rgb(255,220,255)"), (1, "rgb(128,0,128)")],  # Purple scale
    ][:n_splits]
    colorbars = [
        dict(title=f"Metric {idx + 1}", orientation="h") for idx in range(n_splits)
    ]
    fig = pmv.ptable_heatmap_splits_plotly(
        data=data_dict,
        orientation=orientation,
        colorscale=sequential_colors,
        colorbar=colorbars,
    )
    fig.show()

    # if orientation == "diagonal":
    #     pmv.io.save_and_compress_svg(fig, f"ptable-heatmap-splits-plotly-{n_splits}")


# %% Example 4: Custom color schemes with multiple colorbars
def make_color_scale(
    color_schemes: Sequence[dict[str, RgbColorType]],
) -> Callable[[str, float, int], str]:
    """Return element colors in different palettes based on split index."""

    def elem_color_scale(element: str, _val: float, split_idx: int) -> str:
        # Default to gray for elements without defined colors
        color = color_schemes[split_idx].get(element, "(128, 128, 128)")
        return f"rgb{color}"

    return elem_color_scale


palettes_3 = (
    pmv_colors.ELEM_COLORS_ALLOY,
    pmv_colors.ELEM_COLORS_JMOL,
    pmv_colors.ELEM_COLORS_VESTA,
)

# Example with vertical colorbars
fig = pmv.ptable_heatmap_splits_plotly(
    # Use dummy values for all elements
    {str(elem): list(range(len(palettes_3))) for elem in Element},
    orientation="diagonal",  # could also use "grid"
    colorscale=make_color_scale(palettes_3),
    colorbar=[
        dict(title="ALLOY Colors", orientation="v"),
        dict(title="JMOL Colors", orientation="v"),
        dict(title="VESTA Colors", orientation="v"),
    ],
    hover_data={
        el.symbol: "top left: JMOL<br>top right: VESTA, bottom: ALLOY" for el in Element
    },
)
title = (
    "<b>Element color schemes</b><br>top left: JMOL, top right: VESTA, bottom: ALLOY"
)
fig.layout.title.update(text=title, x=0.4, y=0.8)
fig.show()
# pmv.io.save_and_compress_svg(fig, "ptable-heatmap-splits-plotly-3-color-schemes")


# %% Example 5: Two color schemes with horizontal colorbars
palettes_2 = (pmv_colors.ELEM_COLORS_ALLOY, pmv_colors.ELEM_COLORS_VESTA)

fig = pmv.ptable_heatmap_splits_plotly(
    # Use dummy values for all elements
    {str(elem): list(range(len(palettes_2))) for elem in Element},
    orientation="vertical",
    colorscale=make_color_scale(palettes_2),
    colorbar=[
        dict(title="VESTA Colors", orientation="h"),
        dict(title="ALLOY Colors", orientation="h"),
    ],
    hover_data={el.symbol: "left: VESTA<br>right: ALLOY" for el in Element},
)
title = "<b>Element color schemes</b><br>left: VESTA, right: ALLOY"
fig.layout.title.update(text=title, x=0.4, y=0.8)
fig.show()
# pmv.io.save_and_compress_svg(fig, "ptable-heatmap-splits-plotly-2-color-schemes")


# %% Example 6: Mixed colorbar orientations
# Create data with 4 splits
data_dict = {el.symbol: list(np_rng.integers(0, 100, size=4)) for el in Element}

# Use grid orientation with 4 different colorscales and mixed colorbar orientations
fig = pmv.ptable_heatmap_splits_plotly(
    data=data_dict,
    orientation="grid",
    # Use colorscale names directly
    colorscale=["Viridis", "Plasma", "Inferno", "Magma"],
    colorbar=[
        dict(title="Top Left", orientation="v", x=-0.05, y=0, len=0.4),
        dict(title="Top Right", orientation="v", x=0.05, y=0, len=0.4),
        dict(title="Bottom Left", orientation="h"),
        dict(title="Bottom Right", orientation="h"),
    ],
)
title = "<b>Mixed Colorbar Orientations</b><br>Grid Layout Example"
fig.layout.title.update(text=title, x=0.4, y=0.9)
fig.show()
# pmv.io.save_and_compress_svg(fig, "ptable-heatmap-splits-plotly-mixed-colorbars")
