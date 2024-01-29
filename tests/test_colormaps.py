"""Unit tests for colormaps."""
# TODO: refine result colormap with precise comparison (e.g. from colormap[start:end])


from __future__ import annotations

import matplotlib
import pytest
from matplotlib.colors import Colormap

from pymatviz.colormaps import combine_two, truncate


@pytest.mark.parametrize(
    "cmap1, cmap2, node, N, reverse",
    [
        (
            matplotlib.colormaps["Reds"],
            matplotlib.colormaps["Blues_r"],
            0.5,
            256,
            False,
        ),
        ("Reds", "Blues_r", 0.25, 128, True),
    ],
)
def test_combine_two(cmap1, cmap2, node, N, reverse):
    result_cmap = combine_two(
        [cmap1, cmap2], node=node, n_rgb_levels=N, reverse=reverse
    )

    assert isinstance(result_cmap, Colormap)


@pytest.mark.parametrize(
    "cmap1, cmap2, node, N, reverse",
    [
        (-1, "Blues_r", 0.5, 256, False),
    ],
)
def test_combine_two_colormaps_invalid(cmap1, cmap2, node, N, reverse):
    with pytest.raises(TypeError) as excinfo:
        combine_two([cmap1, cmap2], node=node, n_rgb_levels=N, reverse=reverse)

    assert (
        str(excinfo.value)
        == "Invalid datatype. Expect either all Colormaps or all strings."
    )
